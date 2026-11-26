# MESP Point-Cloud Modeling

Code accompanying the manuscript:

> **Molecular Electrostatic Potential Point Clouds as Representations for α1A-Adrenergic Receptor Binding Affinity Prediction**  
> Sebastian Schieferdecker et al.  


This repository contains the computational workflows used to investigate molecular electrostatic potential (MESP) surface point clouds for ligand affinity prediction.

The study compares:

1. descriptors derived from MESP surface point clouds,
2. direct geometric deep-learning models, and
3. downstream prediction using learned geometric embeddings.

Models were evaluated on a dataset of 2,461 α1A-adrenergic receptor ligands using three independent cluster-based train/test splits.

## Contents

- `PointcloudProcessing/` – point-cloud preprocessing and descriptor calculations
- `Models/` – geometric deep-learning models


## Citation

If you use this repository, please cite:

> [DOI]
