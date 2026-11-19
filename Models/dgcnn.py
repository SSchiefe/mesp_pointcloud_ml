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
    extract_global_embeddings,
    log_run_config,
    prepare_datasets,
    run_grid_search,
    save_embeddings,
    setup_logger,
    train_with_config,
)

import torch
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool

# ---- Configuration ----
RESULTS_DIR = os.path.join(_MODELS_DIR, "DGCNN/results")  # log file and model state dict are written here
ACTIVITY_CSV = ""
POINTCLOUD_DIR = ""
CLUSTERSPLITS_SDF = ""
CLUSTERSPLIT_COL = ""
ACTIVITY_COL = "pKi"
EMBEDDINGS_DIR = ""
RESULT_SUFFIX = ""
LOG_FILENAME = ""
MODEL_FILENAME = ""


#%%
# ---- Model ----
class DGCNNRegressor(torch.nn.Module):
    def __init__(
        self,
        k=20,
        aggr="max",  # 'add', 'mean', 'max'
        pooling="max",  # 'max', 'mean', 'max_mean'
        mlp_hidden_dims=(512, 256),
        mlp_dropout=0.3,
    ):
        super().__init__()

        if pooling not in {"max", "mean", "max_mean"}:
            raise ValueError("pooling must be one of {'max', 'mean', 'max_mean'}")
        self.pooling = pooling

        self.conv1 = DynamicEdgeConv(MLP([2 * 4, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        # Norm + activation before pooling, as in the original DGCNN embedding layer.
        self.lin1 = MLP([128 + 64, 1024], plain_last=False)

        pooled_dim = 2048 if pooling == "max_mean" else 1024
        self.mlp = MLP([pooled_dim, *mlp_hidden_dims, 1], dropout=mlp_dropout, norm=None)

        self.global_embedding_dim = pooled_dim

    def forward(self, data, return_global_embeddings=False):
        pos, x, batch = data.pos, data.x, data.batch
        if x is None:
            raise ValueError("Expected `data.x` with ESP charges, but found None.")
        if x.dim() == 1:
            x = x.view(-1, 1)

        node_features = torch.cat([pos, x], dim=1)

        x1 = self.conv1(node_features, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))

        if self.pooling == "max":
            out = global_max_pool(out, batch)
        elif self.pooling == "mean":
            out = global_mean_pool(out, batch)
        else:  # max_mean
            out_max = global_max_pool(out, batch)
            out_mean = global_mean_pool(out, batch)
            out = torch.cat([out_max, out_mean], dim=1)

        global_embeddings = out
        out = self.mlp(out).view(-1)

        if return_global_embeddings:
            return out, global_embeddings

        return out


def build_model(config):
    return DGCNNRegressor(
        k=int(config.get("k", 20)),
        aggr=config.get("aggr", "max"),
        pooling=config.get("pooling", "max"),
        mlp_hidden_dims=tuple(config.get("mlp_hidden_dims", (512, 256))),
        mlp_dropout=float(config.get("mlp_dropout", 0.3)),
    )


#%%
# ---- Main ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)
log_file_path = os.path.join(RESULTS_DIR, LOG_FILENAME)
logger = setup_logger("DGCNN_revised_grid_search", log_file_path)
logger.info("Starting revised DGCNN grid search run")

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

param_grid = {
    "batch_size": [32],
    "lr": [1e-3, 3e-4],
    "weight_decay": [1e-5, 1e-4],
    "k": [10, 20, 30],
    "aggr": ["max"],
    "pooling": ["max", "max_mean"],
    "mlp_hidden_dims": [(512, 256)],
    "mlp_dropout": [0.3, 0.5],
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

print("training DGCNN model with best configuration")
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
print("generating dataset for extraction of global embeddings")
full_dataset = build_pointcloud_dataset(datasets["all_files"], datasets["df"], activity_col=ACTIVITY_COL)
full_loader = build_dataloader(full_dataset, shuffle=False, num_workers=0)
print("extracting global embeddings of dataset")
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
