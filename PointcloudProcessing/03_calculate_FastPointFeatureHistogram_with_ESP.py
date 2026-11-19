#%%
"""
Fast Point Feature Histogram (FPFH) geometric descriptors combined with an
ESP-difference point feature histogram.

For each downsampled point cloud (columns: x, y, z, esp) this computes:
  - fpfh_feature_vector : 33 FPFH dimensions x num_bins_fpfh, pooled as
                          per-dimension histograms (pure geometry).
  - esp_diff_histogram  : histogram of |esp_i - esp_j| over local neighborhoods,
                          pooled over the whole cloud, describing local
                          electrostatic surface texture.
  - esp_grad_*          : per-point mean-|delta esp| statistics (a local ESP
                          gradient proxy).

Fixed, dataset-wide bin edges (see estimate_esp_absdiff_max) make the ESP
histograms comparable across molecules.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree

# Open3D FPFH: each of the three 11-bin blocks sums to 200, so every value lies in [0, 200].
FPFH_VALUE_MAX = 200.0


def load_pointcloud(csv_file: str):
    """Read x, y, z, esp by name (the CSVs carry an unnamed index column)."""
    df = pd.read_csv(csv_file)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df = df.dropna(subset=["x", "y", "z", "esp"])
    coords = df[["x", "y", "z"]].to_numpy(np.float64)
    esp = df["esp"].to_numpy(np.float64)
    return coords, esp


def compute_fpfh_histogram(coords: np.ndarray, voxel_size: float, num_bins: int = 10):
    """Geometric FPFH pooled into per-dimension histograms over a fixed value range."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    histograms = []
    for i in range(33):
        values = np.clip(pcd_fpfh.data[i, :], 0.0, FPFH_VALUE_MAX)
        hist, _ = np.histogram(values, bins=num_bins, range=(0.0, FPFH_VALUE_MAX))
        histograms.append(hist / pcd_fpfh.data.shape[1])
    feature_vector = np.concatenate(histograms)  # shape: (33 * num_bins,)
    return feature_vector, pcd_fpfh.data


def compute_esp_difference_histogram(
    coords: np.ndarray,
    esp: np.ndarray,
    radius: float,
    absdiff_max: float,
    num_bins: int = 20,
    max_nn: int = 100,
):
    """
    ESP analog of FPFH: for every point, histogram the absolute ESP differences
    |esp_i - esp_j| to its spatial neighbors, then pool over the whole cloud.
    Absolute (not signed) differences avoid the built-in symmetry of counting
    each pair in both directions. Bin edges are fixed via absdiff_max so the
    descriptor is comparable across molecules.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    if absdiff_max <= 0:
        absdiff_max = 1.0
    bin_edges = np.linspace(0.0, absdiff_max, num_bins + 1)
    hist_accum = np.zeros(num_bins, dtype=np.float64)
    per_point_grad = []

    for i in range(coords.shape[0]):
        k, idx, _ = kdtree.search_hybrid_vector_3d(pcd.points[i], radius, max_nn)
        if k <= 1:
            continue
        neigh = list(idx[1:])  # drop the query point itself
        abs_d = np.abs(esp[neigh] - esp[i])
        # Tail beyond the shared upper edge goes into the last bin instead of being dropped.
        hist, _ = np.histogram(np.minimum(abs_d, absdiff_max), bins=bin_edges)
        hist_accum += hist
        per_point_grad.append(abs_d.mean())

    total = hist_accum.sum()
    if total > 0:
        hist_accum = hist_accum / total  # normalize to a distribution

    grad = np.asarray(per_point_grad) if per_point_grad else np.zeros(1)
    grad_stats = {
        "esp_grad_mean": float(grad.mean()),
        "esp_grad_std": float(grad.std()),
        "esp_grad_max": float(grad.max()),
    }
    return hist_accum, grad_stats


def estimate_esp_absdiff_max(
    csv_files,
    radius: float,
    max_nn: int = 100,
    upper_pct: float = 99.0,
    max_samples_per_file: int = 20000,
    seed: int = 0,
):
    """
    Dataset-wide upper bin edge for |delta esp|: the upper_pct percentile of the
    differences between each point and its neighbours (same radius / max_nn as the
    histogram), pooled over all clouds. Every molecule then shares one histogram support.
    """
    rng = np.random.default_rng(seed)
    pooled = []
    for f in csv_files:
        coords, esp = load_pointcloud(f)
        n = coords.shape[0]
        if n < 2:
            continue
        k = min(max_nn + 1, n)
        _, idx = cKDTree(coords).query(coords, k=k, distance_upper_bound=radius)
        idx = idx[:, 1:]  # first column is the point itself
        valid = idx < n  # missing neighbours are reported as index n
        diffs = np.abs(esp[np.where(valid, idx, 0)] - esp[:, None])[valid]
        if diffs.size > max_samples_per_file:
            diffs = rng.choice(diffs, size=max_samples_per_file, replace=False)
        pooled.append(diffs)
    pooled = np.concatenate(pooled)
    return float(np.percentile(pooled, upper_pct))


def calculate_combined_features(
    csv_file: str,
    voxel_size: float,
    absdiff_max: float,
    num_bins_fpfh: int = 10,
    num_bins_esp: int = 20,
):
    mol_idx = os.path.basename(csv_file).replace(".csv", "").split("_")[0]
    coords, esp = load_pointcloud(csv_file)

    fpfh_vector, fpfh_data = compute_fpfh_histogram(
        coords, voxel_size, num_bins=num_bins_fpfh
    )
    esp_hist, esp_grad_stats = compute_esp_difference_histogram(
        coords, esp, radius=voxel_size * 2, absdiff_max=absdiff_max, num_bins=num_bins_esp
    )

    result = {
        "fpfh_feature_vector": [float(v) for v in fpfh_vector],
        "esp_diff_histogram": [float(v) for v in esp_hist],
        "fpfh_mean": float(np.mean(fpfh_data)),
        "fpfh_max": float(np.max(fpfh_data)),
        "fpfh_min": float(np.min(fpfh_data)),
        "esp_absdiff_max": float(absdiff_max),
    }
    result.update(esp_grad_stats)
    return mol_idx, result


def main():
    parser = argparse.ArgumentParser(
        description="Compute FPFH geometric features combined with an ESP-difference histogram."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with downsampled point-cloud CSVs.")
    parser.add_argument("--output-file", required=True,
                        help="Path for the combined output JSON.")
    parser.add_argument("--failed-mols-file", default="fpfh_esp_featurization_failed_mols.json",
                        help="Path to write the list of molecules that failed processing.")
    parser.add_argument("--voxel-size", type=float, default=1,
                        help="Voxel size scaling the FPFH/ESP search radii (default: 1).")
    args = parser.parse_args()

    csv_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".csv")
    ]

    # One dataset-wide pass so ESP-difference histograms share fixed bin edges.
    absdiff_max = estimate_esp_absdiff_max(csv_files, radius=args.voxel_size * 2)
    print(f"Using |delta esp| histogram upper edge: {absdiff_max:.4f}")

    results_dict = {}
    failed_mols = []
    for csv_file in csv_files:
        print(f"Processing file: {os.path.basename(csv_file)}")
        try:
            mol_idx, result = calculate_combined_features(
                csv_file, voxel_size=args.voxel_size, absdiff_max=absdiff_max
            )
            results_dict[mol_idx] = result
        except Exception as exc: 
            mol_idx = os.path.basename(csv_file).split("_")[0]
            failed_mols.append(mol_idx)
            print(f"  could not process mol {mol_idx}: {exc}")

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results_dict, f, indent=4)

    with open(args.failed_mols_file, "w") as f:
        json.dump(failed_mols, f, indent=4)


if __name__ == "__main__":
    main()
