"""Shared utilities for the ESP point-cloud 3D-GNN regression models.

This package centralises the data loading, dataset construction, geometry
helpers, and training/grid-search loops that are common to every model script
so the individual model files only need to define their architecture and
hyperparameter grid.
"""

from .logging_utils import log_run_config, setup_logger
from .data import (
    build_dataloader,
    build_pointcloud_dataset,
    get_activity_data_from_dataframe,
    get_molecule_index_from_pointcloud_filename,
    list_pointcloud_files,
    load_activity_dataframe,
    load_pointcloud_data_from_single_mol,
    prepare_datasets,
    save_embeddings,
    split_files_by_clustersplit,
)
from .geometry import estimate_radius_candidates_from_training_data
from .training import (
    compute_target_stats,
    evaluate,
    extract_global_embeddings,
    iter_grid_configs,
    run_grid_search,
    train_one_epoch,
    train_with_config,
)

__all__ = [
    "setup_logger",
    "log_run_config",
    "build_dataloader",
    "build_pointcloud_dataset",
    "get_activity_data_from_dataframe",
    "get_molecule_index_from_pointcloud_filename",
    "list_pointcloud_files",
    "load_activity_dataframe",
    "load_pointcloud_data_from_single_mol",
    "prepare_datasets",
    "save_embeddings",
    "split_files_by_clustersplit",
    "estimate_radius_candidates_from_training_data",
    "compute_target_stats",
    "evaluate",
    "extract_global_embeddings",
    "iter_grid_configs",
    "run_grid_search",
    "train_one_epoch",
    "train_with_config",
]
