#%%
import os
import argparse
import numpy as np
import pandas as pd
import json
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, PersistenceLandscape, BettiCurve
from gtda.curves import StandardFeatures, Derivative
from gtda.utils import check_collection
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform, pdist


def load_pointcloud(csv_file):
    df = pd.read_csv(csv_file)
    return df[['x', 'y', 'z', 'esp']].to_numpy(dtype=np.float64)


def estimate_esp_lambda(csv_files):
    """
    Dataset-wide ESP weight for the 4D (x, y, z, lambda * esp) distance.

    lambda = median nearest-neighbour distance / median |delta esp| between those same
    neighbours, so a typical ESP step between neighbouring points counts as much as a
    typical spatial step.
    """
    nn_dists, esp_diffs = [], []
    for csv_file in csv_files:
        arr = load_pointcloud(csv_file)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) < 2:
            continue
        dist, idx = cKDTree(arr[:, :3]).query(arr[:, :3], k=2)
        nn_dists.append(dist[:, 1])
        esp_diffs.append(np.abs(arr[idx[:, 1], 3] - arr[:, 3]))

    median_dist = float(np.median(np.concatenate(nn_dists)))
    median_esp_diff = float(np.median(np.concatenate(esp_diffs)))
    if median_dist <= 0 or median_esp_diff <= 0:
        raise ValueError("Cannot derive lambda: median neighbour distance or ESP difference is zero.")
    return median_dist / median_esp_diff


def calculate_PH_features(csv_file, output_dir, esp_lambda=None):
    mol_idx = os.path.basename(csv_file).replace(".csv", "")
    result_dict = dict()
    VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1], n_jobs = -1)
    np_array = load_pointcloud(csv_file)
    if esp_lambda is not None:
        np_array[:, 3] *= esp_lambda
    pairwise_distances = squareform(pdist(np_array))
    diagram = VR.fit_transform([pairwise_distances])

    try:
        check_collection(diagram)
    except ValueError:
        print('could not process mol', mol_idx)
        return mol_idx

    diagramScaler = Scaler()
    diagramFiltering = Filtering(epsilon=0.1)
    diagram = diagramScaler.fit_transform(diagram)
    diagram = diagramFiltering.fit_transform(diagram)
    PE = PersistenceEntropy()
    persistence_entropy = PE.fit_transform(diagram)
    PL = PersistenceLandscape()
    persistence_landscale = PL.fit_transform(diagram)
    sf_argmax = StandardFeatures('argmax')
    persistence_landscape_ARGMAX = sf_argmax.fit_transform(persistence_landscale)
    BC = BettiCurve()
    betti_curves = BC.fit_transform(diagram)
    sf_max = StandardFeatures('max')
    betti_curves_MAX = sf_max.fit_transform(betti_curves)
    D = Derivative()
    betti_curve_derivative = D.fit_transform(betti_curves)
    betti_curve_derivative_MAX = sf_max.fit_transform(betti_curve_derivative)
    result_dict['persistence_entropy'] = [float(val) for val in list(persistence_entropy.flatten())]
    result_dict['persistence_landscape_ARGMAX'] = [float(val) for val in list(persistence_landscape_ARGMAX.flatten())]
    result_dict['betti_curves_MAX'] = [float(val) for val in list(betti_curves_MAX.flatten())]
    result_dict['betti_curve_derivative_MAX'] = [float(val) for val in list(betti_curve_derivative_MAX.flatten())]
    result_dict['betti_curves_dimension_1'] = [float(val) for val in list(betti_curves[:,0,:].flatten())]
    result_dict['betti_curves_dimension_2'] = [float(val) for val in list(betti_curves[:,1,:].flatten())]
    result_dict['betti_curve_derivative_dimension_1'] = [float(val) for val in list(betti_curve_derivative[:,0,:].flatten())]
    result_dict['betti_curve_derivative_dimension_2'] = [float(val) for val in list(betti_curve_derivative[:,1,:].flatten())]

    with open(os.path.join(output_dir, f"{mol_idx}_TDA_features.json"), "w") as f:
        json.dump(result_dict, f)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute Vietoris-Rips TDA features from point-cloud CSVs."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with downsampled point-cloud CSVs.")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for per-molecule TDA feature JSONs.")
    parser.add_argument("--failed-mols-file", default="TDA_featurization_failed_mols.json",
                        help="Path to write the list of molecules that failed processing.")
    parser.add_argument("--scale-esp", action="store_true",
                        help="Weight ESP by a dataset-wide lambda derived from the point clouds "
                             "(median neighbour distance / median neighbour |delta esp|) before "
                             "computing the 4D distance matrix. Default: raw ESP values.")
    parser.add_argument("--lambda-file", default="TDA_esp_lambda.json",
                        help="Path to write the derived lambda (only used with --scale-esp).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    point_cloud_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".csv")
    ]

    esp_lambda = None
    if args.scale_esp:
        esp_lambda = estimate_esp_lambda(point_cloud_files)
        print(f"Derived ESP lambda: {esp_lambda:.6f}")
        with open(args.lambda_file, 'w') as f:
            json.dump({"esp_lambda": esp_lambda}, f)

    failed_mols = []
    for file in point_cloud_files:
        print('processing', file)
        try:
            failed = calculate_PH_features(file, args.output_dir, esp_lambda=esp_lambda)
        except Exception as exc:
            failed = os.path.basename(file).replace(".csv", "")
            print(f'  could not process {file}: {exc}')
        if failed is not None:
            failed_mols.append(failed)

    with open(args.failed_mols_file, 'w') as f:
        json.dump(failed_mols, f)


if __name__ == "__main__":
    main()
