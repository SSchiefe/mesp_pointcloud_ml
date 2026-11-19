#%%
import os
import sys

# Make the shared package importable whether run as a file or interactively.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_MODELS_DIR = _SCRIPT_DIR
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

import math

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from torch_geometric.nn import MLP, global_max_pool, global_mean_pool, knn_graph
from torch_geometric.utils import scatter

# ---- Configuration ----
RESULTS_DIR = os.path.join(_MODELS_DIR, "TFN_knn/results")  # log file and model state dict are written here
ACTIVITY_CSV = ""
POINTCLOUD_DIR = ""
CLUSTERSPLITS_SDF = ""
CLUSTERSPLIT_COL = ""
ACTIVITY_COL = "pKi"
EMBEDDINGS_DIR = ""
RESULT_SUFFIX = ""
LOG_FILENAME = ""
MODEL_FILENAME = ""
RADIAL_CUTOFF_PERCENTILE = 99

# Filled from the training set before the grid search: {k: radial_cutoff}.
RADIAL_CUTOFF_BY_K = {}


#%%
# ---- Model ----
class TFNConvLayer(nn.Module):
    def __init__(
        self,
        in_irreps: str,
        out_irreps: str,
        sh_irreps: o3.Irreps,
        radial_hidden_dim: int = 64,
        num_radial_basis: int = 16,
        radial_cutoff: float = 5.0,
        aggr: str = "mean",
        activate: bool = True,
    ):
        super().__init__()
        # Element-wise max over vector components is not rotation-equivariant.
        if aggr not in {"add", "sum", "mean"}:
            raise ValueError("aggr must be one of {'add', 'sum', 'mean'}")

        self.aggr = "sum" if aggr == "add" else aggr
        self.sh_irreps = o3.Irreps(sh_irreps)
        self.num_radial_basis = int(num_radial_basis)
        self.radial_cutoff = float(radial_cutoff)

        out_irreps = o3.Irreps(out_irreps)
        self.gate = None
        self.scalar_act = None
        if activate:
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l == 0])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l > 0])
            if irreps_gated.num_irreps > 0:
                irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e")
                self.gate = Gate(
                    irreps_scalars, [torch.nn.functional.silu],
                    irreps_gates, [torch.sigmoid],
                    irreps_gated,
                )
                tp_out_irreps = self.gate.irreps_in
            else:
                self.scalar_act = nn.SiLU()
                tp_out_irreps = out_irreps
        else:
            tp_out_irreps = out_irreps

        self.tp = o3.FullyConnectedTensorProduct(
            o3.Irreps(in_irreps),
            self.sh_irreps,
            tp_out_irreps,
            shared_weights=False,
        )
        self.self_connection = o3.Linear(o3.Irreps(in_irreps), tp_out_irreps)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.num_radial_basis, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, self.tp.weight_numel),
        )

    def forward(self, x, pos, edge_index):
        # knn_graph (flow="source_to_target"): edge_index[0] = neighbour, edge_index[1] = centre.
        src, dst = edge_index

        rel = pos[src] - pos[dst]
        dist = rel.norm(dim=-1)
        sh = o3.spherical_harmonics(self.sh_irreps, rel, normalize=True, normalization="component")
        rbf = soft_one_hot_linspace(
            dist,
            start=0.0,
            end=self.radial_cutoff,
            number=self.num_radial_basis,
            basis="gaussian",
            cutoff=False,
        ) * self.num_radial_basis**0.5

        edge_weight = self.edge_mlp(rbf)
        message = self.tp(x[src], sh, edge_weight)
        aggregated = scatter(message, dst, dim=0, dim_size=x.size(0), reduce=self.aggr)

        out = self.self_connection(x) + aggregated
        if self.gate is not None:
            out = self.gate(out)
        elif self.scalar_act is not None:
            out = self.scalar_act(out)
        return out


class TFNRegressor(torch.nn.Module):
    def __init__(
        self,
        k: int = 16,
        lmax: int = 2,
        num_layers: int = 3,
        hidden_scalar_channels: int = 32,
        hidden_vector_channels: int = 16,
        readout_scalar_channels: int = 64,
        aggr: str = "mean",
        pooling: str = "max",  # 'max', 'mean', 'max_mean'
        radial_hidden_dim: int = 64,
        num_radial_basis: int = 16,
        radial_cutoff: float = 5.0,
        mlp_hidden_dims=(512, 256),
        mlp_dropout=0.3,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if hidden_scalar_channels < 1:
            raise ValueError("hidden_scalar_channels must be >= 1")
        if readout_scalar_channels < 1:
            raise ValueError("readout_scalar_channels must be >= 1")
        if k < 1:
            raise ValueError("k must be >= 1")
        if lmax < 0:
            raise ValueError("lmax must be >= 0")
        if pooling not in {"max", "mean", "max_mean"}:
            raise ValueError("pooling must be one of {'max', 'mean', 'max_mean'}")

        self.k = int(k)
        self.pooling = pooling

        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_scalar_channels),
            nn.SiLU(),
            nn.Linear(hidden_scalar_channels, hidden_scalar_channels),
        )

        input_irreps = f"{hidden_scalar_channels}x0e"
        if hidden_vector_channels > 0:
            hidden_irreps = f"{hidden_scalar_channels}x0e + {hidden_vector_channels}x1o"
        else:
            hidden_irreps = f"{hidden_scalar_channels}x0e"

        sh_irreps = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TFNConvLayer(
                    in_irreps=input_irreps if i == 0 else hidden_irreps,
                    out_irreps=hidden_irreps,
                    sh_irreps=sh_irreps,
                    radial_hidden_dim=radial_hidden_dim,
                    num_radial_basis=num_radial_basis,
                    radial_cutoff=radial_cutoff,
                    aggr=aggr,
                    activate=i < num_layers - 1,
                )
            )

        self.to_scalar = o3.Linear(o3.Irreps(hidden_irreps), o3.Irreps(f"{readout_scalar_channels}x0e"))

        pooled_dim = 2 * readout_scalar_channels if pooling == "max_mean" else readout_scalar_channels
        self.mlp = MLP([pooled_dim, *mlp_hidden_dims, 1], dropout=mlp_dropout, norm=None)

        self.global_embedding_dim = pooled_dim

    def forward(self, data, return_global_embeddings=False):
        pos, x, batch = data.pos, data.x, data.batch
        if x is None:
            raise ValueError("Expected `data.x` with ESP charges, but found None.")
        if x.dim() == 1:
            x = x.view(-1, 1)

        x = self.input_proj(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)

        for layer in self.layers:
            x = layer(x, pos, edge_index)

        out = self.to_scalar(x)

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


def compute_radial_cutoff(dataset, k, percentile=RADIAL_CUTOFF_PERCENTILE):
    kth_dists = []
    for data in dataset:
        if data.pos.size(0) <= k:
            continue
        src, dst = knn_graph(data.pos, k=k, loop=False)
        d = (data.pos[src] - data.pos[dst]).norm(dim=-1)
        # Edges are grouped by centre node, k per node.
        kth_dists.append(d.view(-1, k).max(dim=1).values)
    value = float(np.percentile(torch.cat(kth_dists).numpy(), percentile))
    return math.ceil(value * 10) / 10


def _channels(config):
    """(scalar, vector) hidden channels; a "channels" entry pairs them so the grid varies capacity as one setting."""
    if "channels" in config:
        return config["channels"]
    return config.get("hidden_scalar_channels", 32), config.get("hidden_vector_channels", 16)


def build_model(config):
    k = int(config.get("k", 16))
    radial_cutoff = config.get("radial_cutoff")
    if radial_cutoff is None:
        radial_cutoff = RADIAL_CUTOFF_BY_K[k]
    return TFNRegressor(
        k=k,
        lmax=int(config.get("lmax", 2)),
        num_layers=int(config.get("num_layers", 3)),
        hidden_scalar_channels=int(_channels(config)[0]),
        hidden_vector_channels=int(_channels(config)[1]),
        readout_scalar_channels=int(config.get("readout_scalar_channels", 64)),
        aggr=config.get("aggr", "mean"),
        pooling=config.get("pooling", "max"),
        radial_hidden_dim=int(config.get("radial_hidden_dim", 64)),
        num_radial_basis=int(config.get("num_radial_basis", 16)),
        radial_cutoff=float(radial_cutoff),
        mlp_hidden_dims=tuple(config.get("mlp_hidden_dims", (512, 256))),
        mlp_dropout=float(config.get("mlp_dropout", 0.3)),
    )


#%%
# ---- Main ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)
log_file_path = os.path.join(RESULTS_DIR, LOG_FILENAME)
logger = setup_logger("TFN_knn_grid_search", log_file_path)
logger.info("Starting TFN kNN grid search run")

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
    "lmax": [1, 2],  # hidden irreps are 0e+1o, so l>2 harmonics have no valid tensor-product path
    "num_layers": [2, 3],
    "channels": [(16, 8), (32, 16)],  # (scalar, vector) hidden channels
    "readout_scalar_channels": [64],
    "aggr": ["mean", "sum"],
    "pooling": ["max", "max_mean"],
    "radial_hidden_dim": [64],
    "num_radial_basis": [16],
    "mlp_hidden_dims": [(512, 256)],
    "mlp_dropout": [0.3],
}
logger.info(f"Parameter grid: {param_grid}")

for k in param_grid["k"]:
    RADIAL_CUTOFF_BY_K[int(k)] = compute_radial_cutoff(train_dataset, int(k))
print(f"Radial cutoff per k (p{RADIAL_CUTOFF_PERCENTILE} of k-th neighbour distance): {RADIAL_CUTOFF_BY_K}")
logger.info(f"Radial cutoff per k (p{RADIAL_CUTOFF_PERCENTILE} of k-th neighbour distance): {RADIAL_CUTOFF_BY_K}")

MAX_TRIALS = 96  # random subset of the grid; None runs every configuration
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
    max_trials=MAX_TRIALS,
    logger=logger,
    early_stopping_patience=GRID_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=GRID_EARLY_STOPPING_MIN_DELTA,
)
print(f"Best grid-search config: {best_config}")
print(f"Best grid-search val MSE: {best_val_mse:.6f}")
logger.info(f"Best grid-search config: {best_config}")
logger.info(f"Best grid-search val MSE: {best_val_mse:.6f}")

print("training TFN model with best configuration")
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
