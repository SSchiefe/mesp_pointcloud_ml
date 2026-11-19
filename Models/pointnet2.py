#%%
# References:
# https://deepwiki.com/charlesq34/pointnet2/1-overview
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
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
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, global_mean_pool, radius

# ---- Configuration ----
RESULTS_DIR = os.path.join(_MODELS_DIR, "PointNet2/results")  # log file and model state dict are written here
ACTIVITY_CSV = ""
POINTCLOUD_DIR = ""
CLUSTERSPLITS_SDF = ""
CLUSTERSPLIT_COL = ""
ACTIVITY_COL = "pKi"
EMBEDDINGS_DIR = ""
RESULT_SUFFIX = ""
LOG_FILENAME = ""
MODEL_FILENAME = ""

# Coordinates are in Å (not rescaled to a unit sphere), so the SA radii are derived from the data.
RADIUS_ESTIMATION_K = 16
RADIUS_ESTIMATION_QUANTILE = 0.5
RADIUS_ESTIMATION_MAX_GRAPHS = 200
RADIUS_ESTIMATION_MAX_POINTS_PER_GRAPH = 512
RADIUS_CANDIDATE_MULTIPLIERS = (1.0, 1.5, 2.0)


#%%
# ---- Model ----
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn, pooling="max"):
        super().__init__()
        self.nn = nn
        self.pooling = pooling

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        if self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2Regressor(torch.nn.Module):
    def __init__(
        self,
        sa1_ratio: float = 0.5,
        sa1_radius: float = 2.0,  # Å
        sa2_ratio: float = 0.25,
        sa2_radius: float = 4.0,  # Å
        mlp_hidden_dims: tuple = (512, 256),
        mlp_dropout: float = 0.3,
        pooling: str = "max",
    ):
        super().__init__()
        if pooling not in {"max", "mean"}:
            raise ValueError("pooling must be one of {'max', 'mean'}")

        # First SA layer input = ESP scalar feature (1) + relative xyz (3).
        self.sa1_module = SAModule(sa1_ratio, sa1_radius, MLP([1 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(sa2_ratio, sa2_radius, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]), pooling=pooling)
        self.global_embedding_dim = 1024

        if len(mlp_hidden_dims) == 0:
            raise ValueError("mlp_hidden_dims must contain at least one hidden layer size.")
        head_dims = [1024, *list(mlp_hidden_dims), 1]
        self.mlp = MLP(head_dims, dropout=mlp_dropout, norm=None)

    def forward(self, data, return_global_embeddings: bool = False):
        x = data.x
        if x is not None and x.dim() == 1:
            x = x.view(-1, 1)
        sa0_out = (x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        global_embeddings = x

        pred = self.mlp(x).view(-1)

        if return_global_embeddings:
            return pred, global_embeddings

        return pred


def build_model(config):
    sa1_radius = float(config["sa1_radius"])
    return PointNet2Regressor(
        sa1_ratio=float(config.get("sa1_ratio", 0.5)),
        sa1_radius=sa1_radius,
        sa2_ratio=float(config.get("sa2_ratio", 0.25)),
        sa2_radius=sa1_radius * float(config.get("sa2_radius_factor", 2.0)),
        mlp_hidden_dims=tuple(config.get("mlp_hidden_dims", (512, 256))),
        mlp_dropout=float(config.get("mlp_dropout", 0.3)),
        pooling=config.get("pooling", "max"),
    )


#%%
# ---- Main ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)
log_file_path = os.path.join(RESULTS_DIR, LOG_FILENAME)
logger = setup_logger("pointnet2_revised_grid_search", log_file_path)
logger.info("Starting revised PointNet2 grid search run")

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
    f"Estimated base SA1 radius from training data: {estimated_radius:.6f} "
    f"| SA1 radius candidates: {estimated_radius_candidates}"
)
print(radius_info_msg)
logger.info(radius_info_msg)

param_grid = {
    "batch_size": [32],
    "lr": [1e-3, 3e-4],
    "weight_decay": [1e-5, 1e-4],
    "mlp_hidden_dims": [(512, 256)],
    "mlp_dropout": [0.2, 0.3],
    "sa1_ratio": [0.25, 0.5],
    "sa1_radius": estimated_radius_candidates,
    "sa2_ratio": [0.125, 0.25],
    "sa2_radius_factor": [2.0],
    "pooling": ["max", "mean"],
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

print("training PointNet2 model with best configuration")
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
print("Extracted GlobalSA embeddings")
logger.info("Extracted GlobalSA embeddings")

model_save_path = os.path.join(RESULTS_DIR, MODEL_FILENAME)
torch.save(model.state_dict(), model_save_path)
logger.info(f"Saved model state dict to: {model_save_path}")
# %%
