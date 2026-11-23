# Point Cloud Processing Pipeline

Featurization pipeline that turns molecular electrostatic potential (ESP)
surfaces into machine-learning descriptors. Raw ESP surfaces (PDB vertex files)
are converted to point clouds, downsampled, and encoded with four complementary
descriptor families: geometric shape features, 3D Zernike moments, Fast Point
Feature Histograms (FPFH), and persistent-homology (TDA) features.

## Pipeline overview

```
  *_vtx.pdb                    (raw ESP surface vertices)
      │  01_esp_pdb_to_csv.sh
      ▼
  processed/<id>.csv           (x, y, z, esp)
      │  02_process_point_cloud.py
      ▼
  downsampled/<id>_downsampled.csv   +   pointcloud_features/<id>_features.json
      │                                              │
      │                                              │  combine_feature_jsons.py
      │                                              ▼
      │                                     combined point-cloud feature JSON
      │
      ├─ 03_calculate_3D_Zernike_descriptors.py             → Zernike CSVs (pos/neg ESP)
      ├─ 03_calculate_FastPointFeatureHistogram.py          → FPFH JSON
      ├─ 03_calculate_FastPointFeatureHistogram_with_ESP.py → FPFH+ESP JSON
      └─ 03_calculate_persistent_homology_features.py       → <id>_TDA_features.json
                                                             |
                                                             │ combine_feature_jsons.py
                                                             ▼
                                                    combined TDA feature JSON
```

Both per-molecule JSON outputs — `<id>_features.json` (stage 02) and
`<id>_TDA_features.json` (persistent homology) — are aggregated into a single
combined JSON with `combine_feature_jsons.py`.

Stage 03 scripts are independent and can be run in any order (or in parallel)
once the downsampled clouds exist.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`
- `open3d` (stages 02 and FPFH)
- `giotto-tda` (persistent-homology stage)
- Bash + `awk` (stage 01)

## Data formats

- **Point-cloud CSV**: columns `x, y, z, esp` (one point per row).
- **Combined feature JSON**: `{ molecule_id: { feature_name: value_or_list } }`.
- Molecule IDs are taken from the first underscore-separated token of the
  filename (e.g. `12345_downsampled.csv` → `12345`), consistent across all
  feature files so they can be merged on the molecule ID.

---

## Scripts

### `01_esp_pdb_to_csv.sh`
Converts raw ESP surface vertex PDB files (`<id>_vtx.pdb`) in the current
directory into `x,y,z,esp` CSVs written to a `processed/` subfolder.

```bash
cd /path/to/pdb_files
mkdir -p processed
bash 01_esp_pdb_to_csv.sh
```

### `02_process_point_cloud.py`
Centers and voxel-downsamples each point cloud (ESP carried through the
downsampling), writes the downsampled `x,y,z,esp` CSV, and extracts geometric
shape descriptors (PCA structure features, convex-hull volume/area, curvature,
bounding-box aspect ratios, nearest-neighbor distance statistics, sphericity)
as one JSON per molecule.

```bash
python 02_process_point_cloud.py \
  --input-dir       processed/ \
  --downsampled-dir downsampled_pointcloud/ \
  --features-dir    pointcloud_features/ \
  --voxel-size      1.0
```

### `03_calculate_3D_Zernike_descriptors.py`
Computes rotation-invariant 3D Zernike descriptors from the ESP-weighted point
clouds, separately for positive and negative ESP. Writes two CSVs (one row per
molecule).

```bash
python 03_calculate_3D_Zernike_descriptors.py \
  --input-dir       downsampled_pointcloud/ \
  --output-positive zernike_positive_ESP.csv \
  --output-negative zernike_negative_ESP.csv \
  --n-max           8
```

### `03_calculate_FastPointFeatureHistogram.py`
Computes geometric FPFH descriptors (spatial coordinates only), pooled into
per-dimension histograms to give a fixed-length vector per molecule. Writes one
combined JSON.

```bash
python 03_calculate_FastPointFeatureHistogram.py \
  --input-dir   downsampled_pointcloud/ \
  --output-file fpfh_features.json \
  --voxel-size  1
```

### `03_calculate_FastPointFeatureHistogram_with_ESP.py`
Geometric FPFH combined with an ESP-difference point feature histogram
(distribution of `|esp_i - esp_j|` over local neighborhoods) plus local ESP
gradient statistics. A dataset-wide pass fixes the ESP histogram bin edges so
descriptors are comparable across molecules. Writes one combined JSON and a
list of molecules that failed.

```bash
python 03_calculate_FastPointFeatureHistogram_with_ESP.py \
  --input-dir        downsampled_pointcloud/ \
  --output-file      fpfh_esp_features.json \
  --failed-mols-file fpfh_esp_failed_mols.json \
  --voxel-size       1
```

### `03_calculate_persistent_homology_features.py`
Computes persistent-homology (TDA) features via a Vietoris–Rips filtration on
the precomputed 4D `(x, y, z, esp)` distance matrix (homology dimensions 0 and
1). Extracts persistence entropy, persistence-landscape argmax, Betti curves and
their maxima, and Betti-curve derivatives. Writes one JSON per molecule plus a
list of molecules that failed.

```bash
python 03_calculate_persistent_homology_features.py \
  --input-dir        downsampled_pointcloud/ \
  --output-dir       tda_features/ \
  --failed-mols-file tda_failed_mols.json \
  --scale-esp        # optional, see below
```

By default the raw ESP values are used as the 4th coordinate. ESP is in kcal/mol
(roughly ±100) while coordinates are in Å, so ESP then dominates the distances.
With `--scale-esp`, ESP is multiplied by a dataset-wide lambda = median
nearest-neighbour distance / median nearest-neighbour `|delta esp|`, computed
from all input clouds in a first pass and written to `--lambda-file`
(default `TDA_esp_lambda.json`).

### `combine_feature_jsons.py`
Aggregates a folder of per-molecule JSONs (from stage 02 or the persistent-
homology stage) into a single `{ molecule_id: {...} }` JSON, matching the
single-file layout produced by the FPFH scripts.

```bash
# Geometric point-cloud features
python combine_feature_jsons.py \
  --input-dir   pointcloud_features/ \
  --output-file pointcloud_features.json

# Persistent-homology features
python combine_feature_jsons.py \
  --input-dir   tda_features/ \
  --output-file tda_features.json
```

Use `--full-key` if molecule IDs contain underscores and the full filename
(minus `.json`) should be used as the key instead of the first token.
