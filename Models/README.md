# ESP Point-Cloud 3D-GNN Models

Graph/point-cloud neural networks that regress ligand activity (`pKi`) directly
from molecular **electrostatic-potential (ESP) point clouds**. Each point carries
an ESP value at a 3D coordinate; every model reads a point cloud and predicts a
single scalar per molecule.

All model scripts share the same data pipeline, training loop, grid search, and
embedding extraction, which live in the [`pointcloud_gnn/`](pointcloud_gnn) package.
Each model script only defines its **architecture** and its **hyperparameter grid**.
Short descriptions of every architecture are in [MODELS.md](MODELS.md).

## Layout

```
Models/
├── pointcloud_gnn/            # shared, reusable package
│   ├── __init__.py            # public API (re-exports the helpers below)
│   ├── logging_utils.py       # setup_logger
│   ├── data.py                # dataset building, splits, dataloaders, saving
│   ├── geometry.py            # radius estimation for radius-graph models
│   └── training.py            # train/eval loops, grid search, embeddings
│
├── dgcnn.py                        # DGCNN (dynamic kNN edge conv)
├── egnn_knn.py                     # EGNN, kNN graph
├── egnn_radius_graph.py            # EGNN, radius graph
├── pointnet2.py                    # PointNet++ set-abstraction
├── point_transformer_knn.py        # Point Transformer, kNN graph
├── point_transformer_radius_graph.py  # Point Transformer, radius graph
├── TFN_knn.py                      # Tensor Field Network, kNN graph
├── TFN_radius_graph.py             # Tensor Field Network, radius graph
└── TFN_attention.py                # Tensor Field Network with multi-head attention, kNN graph
```

## Requirements

- Python with PyTorch and PyTorch Geometric (`torch`, `torch_geometric` incl.
  `torch-scatter`/`torch-cluster`), `numpy`, `pandas`, RDKit.
- `e3nn` is additionally required for the equivariant models
  (`TFN_knn.py`, `TFN_radius_graph.py`, `TFN_attention.py`).
- A CUDA GPU is used automatically when available, otherwise CPU.

Activate the conda environment that has these installed (e.g. the GNN/`rdkit`
environment) before running or before selecting the VS Code interpreter.

## Running a model

Each script is self-contained and can be run either way:

```bash
python egnn_radius_graph.py
```

or cell-by-cell in the VS Code Interactive Window (the files use `#%%` cells).
Each script inserts its own directory on `sys.path`, so `from pointcloud_gnn import ...`
resolves in both cases.

Every run performs the same pipeline:

1. Load activities + ESP point clouds and build a cluster-based train/test split,
   carving a validation set out of the training data.
2. (radius-graph models only) Estimate neighbourhood radius candidates from the
   training geometry.
3. Grid-search the hyperparameters with early stopping on validation MSE.
4. Retrain the best configuration and report test MSE/MAE.
5. Extract per-molecule global embeddings for the **whole** dataset and save them.
6. Save the trained model `state_dict`.

Outputs written per run:

- A `.log` file in `RESULTS_DIR` (grid-search / training / test metrics, preceded by a
  run-configuration block).
- `embeddings_<suffix>.npy`, `targets_<suffix>.npy`, `molecule_indices_<suffix>.npy` and
  `predictions_<suffix>.npy` under the script's `EMBEDDINGS_DIR`.
- `<MODEL_FILENAME>.pth` (model `state_dict`, including the target mean/std) in `RESULTS_DIR`.

## Configuring a run

The paths and run identifiers are constants near the top of each script:

| Constant | Meaning |
| --- | --- |
| `ACTIVITY_CSV` | CSV with a `Title` column and the activity column |
| `POINTCLOUD_DIR` | Folder of per-molecule ESP point-cloud CSVs (`x,y,z,esp`) |
| `CLUSTERSPLITS_SDF` | SDF providing `mol_id` and the cluster-split assignment |
| `CLUSTERSPLIT_COL` | Which cluster-split column to use (e.g. `clustersplit_3`) |
| `ACTIVITY_COL` | Target column to regress (default `pKi`) |
| `EMBEDDINGS_DIR` | Where the extracted embeddings are saved |
| `RESULTS_DIR` | Where the log file and the model `state_dict` are saved (default: `results/` next to the scripts) |
| `RESULT_SUFFIX` | Suffix appended to the saved `.npy` filenames |
| `LOG_FILENAME` / `MODEL_FILENAME` | Log and model output names |

The hyperparameter search space is the `param_grid` dict in each script.

## Shared package API

Import from `pointcloud_gnn`:

- `prepare_datasets(activity_csv, pointcloud_dir, clustersplits_sdf, clustersplit_col, ...)`
  → dict with `train_dataset`, `val_dataset`, `test_dataset`, `df`, `all_files`.
- `build_pointcloud_dataset`, `build_dataloader` — dataset / loader construction.
- `estimate_radius_candidates_from_training_data(train_dataset, ...)`
  → `(base_radius, [candidate radii])` for radius-graph models.
- `run_grid_search(model_factory, train_dataset, val_dataset, device, param_grid, ...)`
  → `(best_config, best_val_mse, best_state)`.
- `train_with_config(model_factory, train_dataset, val_dataset, test_dataset, device, config, ...)`
  → `(model, train_loader, val_loader, test_loader)`.
- `extract_global_embeddings(model, loader, device)` → `(embeddings, targets, mol_indices)`.
- `save_embeddings(out_dir, embeddings, targets, mol_indices, suffix)`.
- `setup_logger(name, log_file_path)`.

### Adding a new model

1. Define a `torch.nn.Module` whose `forward(self, data, return_global_embeddings=False)`
   returns the prediction (and the pooled embedding when requested), and set
   `self.global_embedding_dim`.
2. Provide a factory `build_model(config) -> nn.Module` that reads hyperparameters
   from a config dict.
3. Pass `build_model` to `run_grid_search` / `train_with_config`.

The shared training loop calls `model(data)` and reads `data.mol_id`, so any model
following this contract works without touching the shared code.
