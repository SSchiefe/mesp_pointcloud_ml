#%%
"""
Aggregate per-molecule feature JSON files into a single combined JSON.

The point-cloud (02_process_point_cloud.py) and TDA
(03_calculate_persistent_homology_features.py) scripts write one JSON per molecule.
This tool collects a folder of such files into one
    { molecule_index: { ...features... }, ... }
JSON, mirroring the single-file layout produced by the FPFH scripts.
"""

import os
import json
import argparse


def combine_feature_jsons(input_dir: str, output_file: str, full_key: bool = False):
    combined = {}
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        # Match FPFH/Zernike keys (first underscore-separated token) by default.
        mol_idx = fname[: -len(".json")] if full_key else fname.split("_")[0]
        with open(os.path.join(input_dir, fname)) as f:
            combined[mol_idx] = json.load(f)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-molecule feature JSONs into one JSON file."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing per-molecule feature JSON files.")
    parser.add_argument("--output-file", required=True,
                        help="Path for the combined output JSON.")
    parser.add_argument("--full-key", action="store_true",
                        help="Use the full filename (minus .json) as the molecule key "
                             "instead of the first underscore-separated token.")
    args = parser.parse_args()

    combined = combine_feature_jsons(args.input_dir, args.output_file, full_key=args.full_key)

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(combined, f, indent=4)
    print(f"Wrote {len(combined)} molecules to {args.output_file}")


if __name__ == "__main__":
    main()

# %%
