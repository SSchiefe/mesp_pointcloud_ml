import logging
import re


def setup_logger(name: str, log_file_path: str) -> logging.Logger:
    """Create a file logger that writes INFO-level messages to ``log_file_path``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_run_config(logger: logging.Logger, datasets: dict = None, **config) -> None:
    """Write the run's inputs, outputs and dataset sizes as one block into the log."""
    logger.info("Run configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    pointcloud_dir = config.get("pointcloud_dir")
    if pointcloud_dir:
        match = re.search(r"voxel_?size[_-]?\d+(\.\d+)?", str(pointcloud_dir), flags=re.IGNORECASE)
        logger.info(f"  voxel_size_tag (from pointcloud_dir): {match.group(0) if match else 'not found'}")

    if datasets is not None:
        n_files = len(datasets["all_files"])
        n_train = len(datasets["train_dataset"])
        n_val = len(datasets["val_dataset"])
        n_test = len(datasets["test_dataset"])
        logger.info(f"  point_cloud_files: {n_files}")
        logger.info(f"  train / val / test molecules: {n_train} / {n_val} / {n_test}")
        logger.info(
            f"  files not used (no activity or no train/test assignment): {n_files - n_train - n_val - n_test}"
        )
