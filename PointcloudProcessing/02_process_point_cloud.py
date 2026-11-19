#%%
import os
import argparse
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.stats import skew
import json


def process_pointcloud_csv(csv_file, downsampled_dir, features_dir, voxel_size=1.0):
    """
    processes pointcloud files labeled as "molecule_id".csv
    csvs should have columns 'x', 'y', 'z', and 'esp' and were prepared with the 
    01_esp_pdb_to_csv.sh script from Multiwfn exported pdb files.
    """
    mol_idx = os.path.basename(csv_file).replace(".csv", "")

    df = pd.read_csv(csv_file)
    df['nan_y'] = [0]*df.shape[0]
    df['nan_z'] = [0]*df.shape[0]
    df = df.dropna()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].to_numpy())
    point_cloud.colors = o3d.utility.Vector3dVector(df[['esp', 'nan_y', 'nan_z']].values)

    center = point_cloud.get_center()
    point_cloud_centered = point_cloud.translate(-center)
    sampled_pcd = point_cloud_centered.voxel_down_sample(voxel_size = voxel_size)
    # Open3D's convex hull / bounding box can segfault on degenerate clouds, which try/except cannot catch.
    if len(sampled_pcd.points) < 4:
        raise ValueError(f"only {len(sampled_pcd.points)} points after downsampling")

    points_df = pd.DataFrame(sampled_pcd.points)
    esp_df = pd.DataFrame(sampled_pcd.colors).iloc[:, 0]

    df_downsampled = pd.concat([points_df, esp_df], axis=1)
    df_downsampled.columns = ['x', 'y', 'z', 'esp']
    df_downsampled.to_csv(os.path.join(downsampled_dir, f"{mol_idx}_downsampled.csv"), index=False)

    features = dict()

    covariance_matrix = np.cov(points_df.T)
    eig_val, eig_vec = np.linalg.eigh(covariance_matrix)
    sorted_indexes = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_indexes]
    eig_vec = eig_vec[:, sorted_indexes]
    if eig_val[0] <= 0 or eig_val[2] / eig_val[0] < 1e-6:
        raise ValueError("degenerate (planar or linear) point cloud")

    planarity = (eig_val[1] - eig_val[2]) / eig_val[0]
    linearity = (eig_val[0] - eig_val[1]) / eig_val[0]
    omnivariance = (eig_val[0] * eig_val[1] * eig_val[2])**(1/3)
    normal = eig_vec[:, 2]  # eigenvectors are columns
    # Eigenvector sign is arbitrary; fix it so normal_z >= 0.
    if normal[2] < 0:
        normal = -normal
    verticality = 1 - normal[2]
    normal_x = normal[0]
    normal_y = normal[1]
    normal_z = normal[2]

    ch = sampled_pcd.compute_convex_hull()
    volume = ch[0].get_volume()
    area = ch[0].get_surface_area()

    features['planarity'] = round(planarity, 5)
    features['linearity'] = round(linearity, 5)
    features['omnivariance'] = round(omnivariance, 5)
    features['verticality'] = round(verticality, 5)
    features['normal_x'] = round(normal_x, 5)
    features['normal_y'] = round(normal_y, 5)
    features['normal_z'] = round(normal_z, 5)
    features['volume'] = round(volume, 5)
    features['area'] = round(area, 5)

    points = np.asarray(sampled_pcd.points)

    kdtree = o3d.geometry.KDTreeFlann(sampled_pcd)
    curvatures = []
    for i in range(len(points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(sampled_pcd.points[i], 30)
        neighbors = points[idx, :]
        cov = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)
        curvature = eigvals[0] / np.sum(eigvals)
        curvatures.append(curvature)
    curvatures = np.array(curvatures)

    features['curvature_min'] = round(curvatures.min(), 5)
    features['curvature_max'] = round(curvatures.max(), 5)
    features['curvature_mean'] = round(curvatures.mean(), 5)

    #aspect ratio
    oobb = sampled_pcd.get_minimal_oriented_bounding_box()
    L, l, h = sorted(oobb.extent, reverse=True)
    l, h = max(l, 1e-8), max(h, 1e-8)
    ratio_Lh, ratio_lh, ratio_Ll = [L/h, l/h, L/l]

    features['ratio_Lh'] = round(ratio_Lh, 5)
    features['ratio_lh'] = round(ratio_lh, 5)
    features['ratio_Ll'] = round(ratio_Ll, 5)

    nn_distances = sampled_pcd.compute_nearest_neighbor_distance()
    mean_dist = round(np.mean(nn_distances),5)
    median_dist = round(np.median(nn_distances),5)
    std_dist = round(np.std(nn_distances),5)
    percentile25_dist = round(np.percentile(nn_distances, 25),5)
    percentile50_dist = round(np.percentile(nn_distances, 50),5)
    percentile75_dist = round(np.percentile(nn_distances, 75),5)
    percentile95_dist = round(np.percentile(nn_distances, 95),5)

    features['mean_dist'] = mean_dist
    features['median_dist'] = median_dist
    features['std_dist'] = std_dist
    features['percentile25_dist'] = percentile25_dist
    features['percentile50_dist'] = percentile50_dist
    features['percentile75_dist'] = percentile75_dist
    features['percentile95_dist'] = percentile95_dist

    skew_dist = skew(nn_distances)
    features['skew_dist'] = round(skew_dist, 5)

    sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / area
    features['sphericity'] = round(sphericity, 5)

    with open(os.path.join(features_dir, f"{mol_idx}_features.json"), "w") as f:
        json.dump(features, f)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Downsample ESP point clouds and extract geometric features."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with raw point-cloud CSVs (molecule_id.csv).")
    parser.add_argument("--downsampled-dir", required=True,
                        help="Output directory for downsampled point-cloud CSVs.")
    parser.add_argument("--features-dir", required=True,
                        help="Output directory for per-molecule feature JSONs.")
    parser.add_argument("--voxel-size", type=float, default=1.0,
                        help="Voxel size used for downsampling (default: 1.0).")
    parser.add_argument("--failed-mols-file", default="pointcloud_featurization_failed_mols.json",
                        help="Path to write the list of molecules that failed processing.")
    args = parser.parse_args()

    os.makedirs(args.downsampled_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    point_cloud_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".csv")
    ]
    failed_mols = []
    for file in point_cloud_files:
        print("processing", file)
        try:
            process_pointcloud_csv(
                file, args.downsampled_dir, args.features_dir, voxel_size=args.voxel_size
            )
        except Exception as exc:
            failed_mols.append(os.path.basename(file).replace(".csv", ""))
            print(f"  could not process {file}: {exc}")

    with open(args.failed_mols_file, "w") as f:
        json.dump(failed_mols, f, indent=4)


if __name__ == "__main__":
    main()
