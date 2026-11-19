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
import torch.nn as nn
from torch_geometric.nn import knn_graph, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter

# ---- Configuration ----
RESULTS_DIR = os.path.join(_MODELS_DIR, "EGNN_knn/results")  # log file and model state dict are written here
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
class EGNNLayer(nn.Module):
    """One EGNN block: updates node features h and coordinates x."""

    def __init__(self, hidden_dim: int, edge_dim: int = 0, act=nn.SiLU()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.act = act

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        # flow="source_to_target": edge_index[0] = neighbour j, edge_index[1] = centre i.
        src, dst = edge_index

        x_diff = x[dst] - x[src]
        dist2 = (x_diff ** 2).sum(dim=-1, keepdim=True)

        if edge_attr is None:
            edge_input = torch.cat([h[dst], h[src], dist2], dim=-1)
        else:
            edge_input = torch.cat([h[dst], h[src], dist2, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        # Normalising by (|x_i - x_j| + 1) keeps coordinate steps bounded.
        delta_x_ij = x_diff / (dist2.sqrt() + 1.0) * self.coord_mlp(m_ij)
        delta_x = scatter(delta_x_ij, dst, dim=0, dim_size=x.size(0), reduce="mean")
        x = x + delta_x

        m_i = scatter(m_ij, dst, dim=0, dim_size=h.size(0), reduce="sum")
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h, x


class EGNNRegressor(nn.Module):
    """EGNN for graph-level pKi regression on ESP point clouds using a kNN graph."""

    def __init__(
        self,
        in_charge_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        k: int = 16,
        pooling: str = "mean",  # "mean" or "max"
        dropout: float = 0.1,
    ):
        super().__init__()
        assert pooling in {"mean", "max"}, "pooling must be 'mean' or 'max'"
        self.k = k
        self.pooling = pooling
        self.global_embedding_dim = hidden_dim

        # Only the ESP value: absolute coordinates would break E(3) invariance.
        self.input_proj = nn.Sequential(
            nn.Linear(in_charge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim=hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data, return_global_embeddings: bool = False):
        pos = data.pos
        batch = data.batch

        x_charge = data.x
        if x_charge is None:
            raise ValueError("Expected `data.x` with ESP charges, but got None.")
        if x_charge.dim() == 1:
            x_charge = x_charge.unsqueeze(-1)

        h = self.input_proj(x_charge)

        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)

        x = pos
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        if self.pooling == "mean":
            g = global_mean_pool(h, batch)
        else:
            g = global_max_pool(h, batch)

        out = self.head(g).view(-1)

        if return_global_embeddings:
            return out, g
        return out


def build_model(config):
    return EGNNRegressor(
        in_charge_dim=1,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 4)),
        k=int(config.get("k", 16)),
        dropout=float(config.get("dropout", 0.1)),
        pooling=config.get("pooling", "mean"),
    )


#%%
# ---- Main ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)
log_file_path = os.path.join(RESULTS_DIR, LOG_FILENAME)
logger = setup_logger("EGNN_revised_grid_search", log_file_path)
logger.info("Starting revised EGNN grid search run")

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
    "batch_size": [16],
    "lr": [1e-3, 3e-4],
    "weight_decay": [1e-5, 1e-4],
    "k": [8, 16],
    "hidden_dim": [64, 128],
    "num_layers": [3, 4],
    "dropout": [0.1, 0.3],
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

print("training EGNN model with best configuration")
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
