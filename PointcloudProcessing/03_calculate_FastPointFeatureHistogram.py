#%%
import open3d as o3d
import pandas as pd
import numpy as np
import os
import argparse
import json
#%%

# Open3D FPFH: each of the three 11-bin blocks sums to 200, so every value lies in [0, 200].
FPFH_VALUE_MAX = 200.0

def calculate_fast_point_feature_histogram_features(csv_file, voxel_size=1):

    mol_idx = csv_file.split("/")[-1].replace(".csv", "").split("_")[0]
    df = pd.read_csv(csv_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df = df.dropna(subset=['x', 'y', 'z'])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].to_numpy())

    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    num_bins = 10
    histograms = []
    n_points = pcd_fpfh.data.shape[1]

    for i in range(33):
        # Fixed range so bins mean the same thing for every molecule; clip guards float overshoot.
        values = np.clip(pcd_fpfh.data[i, :], 0.0, FPFH_VALUE_MAX)
        hist, _ = np.histogram(values, bins=num_bins, range=(0.0, FPFH_VALUE_MAX))
        histograms.append(hist / n_points)

    feature_vector = np.concatenate(histograms)  # shape: (33 * num_bins,)
    mean = np.mean(pcd_fpfh.data)
    max = np.max(pcd_fpfh.data)
    min = np.min(pcd_fpfh.data)

    return feature_vector, mol_idx, mean, max, min


def main():
    parser = argparse.ArgumentParser(
        description="Compute Fast Point Feature Histogram (FPFH) features from point-cloud CSVs."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with downsampled point-cloud CSVs.")
    parser.add_argument("--output-file", required=True,
                        help="Path for the combined output JSON.")
    parser.add_argument("--voxel-size", type=float, default=1,
                        help="Voxel size scaling the FPFH search radii (default: 1).")
    args = parser.parse_args()

    results_dict = {}
    for file in os.listdir(args.input_dir):
        if not file.endswith(".csv"):
            continue
        print(f"Processing file: {file}")
        feature_vector, mol_idx, mean, max, min = calculate_fast_point_feature_histogram_features(
            os.path.join(args.input_dir, file), voxel_size=args.voxel_size
        )
        results_dict[mol_idx] = {
            'feature_vector': [float(f) for f in feature_vector],
            'mean': float(mean),
            'max': float(max),
            'min': float(min)
        }

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    main()
