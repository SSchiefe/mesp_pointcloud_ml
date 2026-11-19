import os
from typing import List

# Import rdkit before torch: importing torch first loads its bundled libstdc++,
# after which rdkit resolves against the system libstdc++ and fails on GLIBCXX.
from rdkit.Chem import PandasTools

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def get_molecule_index_from_pointcloud_filename(file: str) -> int:
    return int(os.path.basename(file).split("_")[0])


def get_activity_data_from_dataframe(df: pd.DataFrame, molecule_index: int, activity_col: str = "pKi") -> float:
    values = df[df["molecule_index"] == molecule_index][activity_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(f"No valid activity value found for molecule index {molecule_index}")
    return float(values[0])


def load_pointcloud_data_from_single_mol(file: str, df: pd.DataFrame, activity_col: str = "pKi"):
    molecule_index = get_molecule_index_from_pointcloud_filename(file)
    activity_data = get_activity_data_from_dataframe(df, molecule_index=molecule_index, activity_col=activity_col)
    pointcloud_data = pd.read_csv(file)
    coordinates = pointcloud_data[["x", "y", "z"]].values
    esp_values = pointcloud_data["esp"].values
    return molecule_index, coordinates, esp_values, activity_data


def build_pointcloud_dataset(filepath_list: list, df: pd.DataFrame, activity_col: str = "pKi") -> List[Data]:
    dataset = []
    skipped = 0
    for file in filepath_list:
        try:
            molecule_index, coordinates, esp_values, activity_value = load_pointcloud_data_from_single_mol(
                file, df, activity_col
            )
        except ValueError:
            skipped += 1
            continue

        # "mol_id" avoids the "index" suffix that PyG auto-increments during batching.
        mol_data = Data(
            x=torch.from_numpy(esp_values.astype(np.float32)).view(-1, 1),
            pos=torch.from_numpy(coordinates.astype(np.float32)),
            y=torch.tensor([activity_value], dtype=torch.float32),
            mol_id=torch.tensor([molecule_index], dtype=torch.long),
        )
        dataset.append(mol_data)

    if len(dataset) == 0:
        raise RuntimeError("No samples could be built for regression dataset.")
    if skipped:
        print(f"Skipped {skipped} of {len(filepath_list)} point clouds without a valid activity value")

    return dataset


def build_dataloader(dataset: list, batch_size: int = 32, shuffle: bool = True, num_workers: int = 10) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_activity_dataframe(csv_path: str, title_col: str = "Title") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["molecule_index"] = df[title_col].apply(lambda title: int(title.split(":")[1].split("_")[0]))
    return df


def list_pointcloud_files(directory: str) -> list:
    return [os.path.join(directory, f) for f in os.listdir(directory)]


def split_files_by_clustersplit(file_paths: list, clustersplits_sdf: str, clustersplit_col: str):
    mol_splits = PandasTools.LoadSDF(clustersplits_sdf)
    train_ids = set(mol_splits[mol_splits[clustersplit_col] == "train"]["mol_id"].astype(int).values)
    test_ids = set(mol_splits[mol_splits[clustersplit_col] == "test"]["mol_id"].astype(int).values)

    train_files = [f for f in file_paths if get_molecule_index_from_pointcloud_filename(f) in train_ids]
    test_files = [f for f in file_paths if get_molecule_index_from_pointcloud_filename(f) in test_ids]
    return train_files, test_files


def prepare_datasets(
    activity_csv: str,
    pointcloud_dir: str,
    clustersplits_sdf: str,
    clustersplit_col: str,
    activity_col: str = "pKi",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Load activities and point clouds, split by cluster, and carve out a validation set."""
    df = load_activity_dataframe(activity_csv)
    all_files = list_pointcloud_files(pointcloud_dir)
    train_files, test_files = split_files_by_clustersplit(all_files, clustersplits_sdf, clustersplit_col)

    train_and_val = build_pointcloud_dataset(train_files, df, activity_col)
    test_dataset = build_pointcloud_dataset(test_files, df, activity_col)

    n_total = len(train_and_val)
    n_train = int((1.0 - val_fraction) * n_total)
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_and_val, [n_train, n_val], generator=generator)

    return {
        "df": df,
        "all_files": all_files,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


def save_embeddings(out_dir: str, embeddings, targets, mol_indices, suffix: str, predictions=None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"embeddings_{suffix}.npy"), embeddings)
    np.save(os.path.join(out_dir, f"targets_{suffix}.npy"), targets)
    np.save(os.path.join(out_dir, f"molecule_indices_{suffix}.npy"), mol_indices)
    if predictions is not None:
        np.save(os.path.join(out_dir, f"predictions_{suffix}.npy"), predictions)
