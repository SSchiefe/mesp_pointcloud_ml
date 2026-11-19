#%%
import os
import sys

# Make the shared package importable whether run as a file or interactively.
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

# Import the rdkit-backed package before torch to avoid a GLIBCXX libstdc++ conflict.
from pointcloud_gnn import (
    build_dataloader,
    build_pointcloud_dataset,
    estimate_radius_candidates_from_training_data,
    extract_global_embeddings,
    log_run_config,
    prepare_datasets,
    run_grid_search,
    save_embeddings,
    setup_logger,
    train_with_config,
)

import torch
from torch.nn import Linear as Lin
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_max_pool,
    global_mean_pool,
    knn,
    radius_graph,
)
from torch_geometric.utils import scatter

# ---- Configuration ----
RESULTS_DIR = os.path.join(_MODELS_DIR, "PointTransformer_radius/results")  # log file and model state dict are written here
ACTIVITY_CSV = ""
POINTCLOUD_DIR = ""
CLUSTERSPLITS_SDF = ""
CLUSTERSPLIT_COL = ""
ACTIVITY_COL = "pKi"
EMBEDDINGS_DIR = ""
RESULT_SUFFIX = ""
LOG_FILENAME = ""
MODEL_FILENAME = ""

RADIUS_ESTIMATION_K = 16
RADIUS_ESTIMATION_QUANTILE = 0.5
RADIUS_ESTIMATION_MAX_GRAPHS = 200
RADIUS_ESTIMATION_MAX_POINTS_PER_GRAPH = 512
RADIUS_CANDIDATE_MULTIPLIERS = (0.8, 1.0, 1.2)


#%%
# ---- Model ----
class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """Sample the point cloud by a ratio to reduce cardinality and augment feature dim."""

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        id_clusters = fps(pos, ratio=self.ratio, batch=batch, random_start=self.training)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        x = self.mlp(x)
        x_out = scatter(
            x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce="max"
        )
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformerRegressor(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim_model=(32, 64, 128, 256, 512),
        k=16,
        radius: float = 1.0,
        max_num_neighbors: int = 64,
        head_hidden_dim=64,
        pooling="mean",
        down_ratio: float = 0.25,
    ):
        super().__init__()
        if pooling not in {"mean", "max"}:
            raise ValueError("pooling must be one of {'mean', 'max'}")
        self.k = k
        self.radius = float(radius)
        self.max_num_neighbors = int(max_num_neighbors)
        self.head_hidden_dim = head_hidden_dim
        self.pooling = pooling
        self.down_ratio = float(down_ratio)

        in_channels = max(in_channels, 1)

        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])

        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()
        self.global_embedding_dim = dim_model[-1]

        for i in range(len(dim_model) - 1):
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i], out_channels=dim_model[i + 1], ratio=self.down_ratio, k=self.k
                )
            )
            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1], out_channels=dim_model[i + 1])
            )

        self.mlp_output = MLP([dim_model[-1], self.head_hidden_dim, out_channels], norm=None)

    def forward(self, data, return_global_embeddings: bool = False):
        x, pos, batch = data.x, data.pos, data.batch

        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.device)
        if x.dim() == 1:
            x = x.view(-1, 1)

        x = self.mlp_input(x)
        edge_index = radius_graph(
            pos, r=self.radius, batch=batch, loop=False, max_num_neighbors=self.max_num_neighbors
        )
        x = self.transformer_input(x, pos, edge_index)

        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            # fps keeps `down_ratio` of the points; on a surface the spacing grows by 1/sqrt(ratio).
            level_radius = self.radius * (1.0 / self.down_ratio) ** ((i + 1) / 2)
            edge_index = radius_graph(
                pos, r=level_radius, batch=batch, loop=False, max_num_neighbors=self.max_num_neighbors
            )
            x = self.transformers_down[i](x, pos, edge_index)

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        else:
            x = global_max_pool(x, batch)
        global_embeddings = x

        out = self.mlp_output(x).view(-1)

        if return_global_embeddings:
            return out, global_embeddings

        return out


def build_model(config):
    return PointTransformerRegressor(
        in_channels=1,
        out_channels=1,
        dim_model=tuple(config.get("dim_model", (64, 128, 256))),
        k=int(config.get("k", 16)),
        radius=float(config.get("radius", 1.0)),
        max_num_neighbors=int(config.get("max_num_neighbors", 64)),
        head_hidden_dim=int(config.get("head_hidden_dim", 128)),
        pooling=config.get("pooling", "mean"),
    )


#%%
# ---- Main ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)
log_file_path = os.path.join(RESULTS_DIR, LOG_FILENAME)
logger = setup_logger("pointtransformer_radiusgraph_revised_grid_search", log_file_path)
logger.info("Starting revised PointTransformer radius-graph grid search run")

datasets = prepare_datasets(
    activity_csv=ACTIVITY_CSV,
    pointcloud_dir=POINTCLOUD_DIR,
    clustersplits_sdf=CLUSTERSPLITS_SDF,
    clustersplit_col=CLUSTERSPLIT_COL,
    activity_col=ACTIVITY_COL,
)
train_dataset = datasets["train_dataset"]
val_dataset = datasets["val_dataset"]
test_dataset = datasets["test_dataset"]
log_run_config(
    logger,
    datasets=datasets,
    activity_csv=ACTIVITY_CSV,
    pointcloud_dir=POINTCLOUD_DIR,
    clustersplits_sdf=CLUSTERSPLITS_SDF,
    clustersplit_col=CLUSTERSPLIT_COL,
    activity_col=ACTIVITY_COL,
    embeddings_dir=EMBEDDINGS_DIR,
    results_dir=RESULTS_DIR,
    model_file=MODEL_FILENAME,
    device=device,
)

estimated_radius, estimated_radius_candidates = estimate_radius_candidates_from_training_data(
    train_dataset=train_dataset,
    k_for_estimation=RADIUS_ESTIMATION_K,
    quantile=RADIUS_ESTIMATION_QUANTILE,
    max_graphs=RADIUS_ESTIMATION_MAX_GRAPHS,
    max_points_per_graph=RADIUS_ESTIMATION_MAX_POINTS_PER_GRAPH,
    radius_multipliers=RADIUS_CANDIDATE_MULTIPLIERS,
)
radius_info_msg = (
    f"Estimated base radius from training data: {estimated_radius:.6f} "
    f"| radius candidates: {estimated_radius_candidates}"
)
print(radius_info_msg)
logger.info(radius_info_msg)

param_grid = {
    "batch_size": [32],
    "lr": [1e-3, 3e-4],
    "weight_decay": [1e-5, 1e-4],
    "k": [16],
    "radius": estimated_radius_candidates,
    "max_num_neighbors": [64],
    "head_hidden_dim": [64],
    "dim_model": [
        (64, 128, 256),
        (64, 128, 256, 512),
        (32, 64, 128, 256),
    ],
    "pooling": ["mean", "max"],
}
logger.info(f"Parameter grid: {param_grid}")

GRID_EARLY_STOPPING_PATIENCE = 5  # None to disable
GRID_EARLY_STOPPING_MIN_DELTA = 1e-4
FINAL_EARLY_STOPPING_PATIENCE = 10  # None to disable
FINAL_EARLY_STOPPING_MIN_DELTA = 1e-4

print("running hyperparameter grid search")
best_config, best_val_mse, best_state = run_grid_search(
    model_factory=build_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    param_grid=param_grid,
    epochs=25,
    num_workers=0,
    logger=logger,
    early_stopping_patience=GRID_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=GRID_EARLY_STOPPING_MIN_DELTA,
)
print(f"Best grid-search config: {best_config}")
print(f"Best grid-search val MSE: {best_val_mse:.6f}")
logger.info(f"Best grid-search config: {best_config}")
logger.info(f"Best grid-search val MSE: {best_val_mse:.6f}")

print("training PointTransformer model with best configuration")
model, train_loader, val_loader, test_loader = train_with_config(
    model_factory=build_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    device=device,
    config=best_config,
    epochs=50,
    num_workers=0,
    logger=logger,
    early_stopping_patience=FINAL_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=FINAL_EARLY_STOPPING_MIN_DELTA,
)

# Extract embeddings for the whole dataset.
full_dataset = build_pointcloud_dataset(datasets["all_files"], datasets["df"], activity_col=ACTIVITY_COL)
full_loader = build_dataloader(full_dataset, shuffle=False, num_workers=0)
embeddings, targets, mol_indices, predictions = extract_global_embeddings(
    model, full_loader, device, return_predictions=True
)

save_embeddings(
    EMBEDDINGS_DIR,
    embeddings.detach().cpu().numpy(),
    targets.detach().cpu().numpy(),
    mol_indices.detach().cpu().numpy(),
    RESULT_SUFFIX,
    predictions=predictions.detach().cpu().numpy(),
)
print("Extracted global pool embeddings")
logger.info("Extracted global pool embeddings")

model_save_path = os.path.join(RESULTS_DIR, MODEL_FILENAME)
torch.save(model.state_dict(), model_save_path)
logger.info(f"Saved model state dict to: {model_save_path}")
# %%
