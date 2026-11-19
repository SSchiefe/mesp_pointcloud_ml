#%%
"""
Example: 3D Zernike descriptors for ESP point clouds from a pandas DataFrame.

Input DataFrame columns:
  - x, y, z : 3D coordinates
  - esp     : electrostatic potential at each point
  n_max controls descriptor size/cost (higher = richer but slower/noisier).

Output:
  - z_pos: 3D Zernike invariant descriptor vector for positive ESP
  - z_neg: 3D Zernike invariant descriptor vector for negative ESP
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.special import comb
import os
import argparse
import pathlib

try:
    from scipy.special import sph_harm_y  # SciPy >= 1.15; sph_harm was removed in newer releases

    def _sph_harm(m, l, theta, phi):
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm

    def _sph_harm(m, l, theta, phi):
        return sph_harm(m, l, phi, theta)

def radial_poly_3d(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """
    3D Zernike radial function R_{n,l}(r) (Canterakis; Novotni & Klein), n-l even and l<=n.
    Orthogonal on [0, 1] with weight r^2: integral R_{n,l} R_{n',l} r^2 dr = delta / 3.
    """
    if (n - l) % 2 != 0 or l > n or l < 0:
        return np.zeros_like(r, dtype=np.float64)

    k = (n - l) // 2
    out = np.zeros_like(r, dtype=np.float64)
    prefactor = math.sqrt((2 * l + 4 * k + 3) / 3.0) * comb(2 * k, k, exact=True) / 4.0**k

    for nu in range(k + 1):
        q = (
            ((-1.0) ** (k + nu))
            * prefactor
            * comb(k, nu, exact=True)
            * comb(2 * (k + l + nu) + 1, 2 * k, exact=True)
            / comb(k + l + nu, k, exact=True)
        )
        out += q * (r ** (2 * nu + l))
    return out


def zernike_basis_3d(
    n: int, l: int, m: int, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """
    Z_{n,l,m}(r,theta,phi) = R_{n,l}(r) * Y_{l,m}(theta,phi), theta polar, phi azimuthal
    """
    R_nl = radial_poly_3d(n, l, r)
    Y_lm = _sph_harm(m, l, theta, phi)
    return R_nl * Y_lm


def to_unit_ball(coords: np.ndarray) -> np.ndarray:
    """
    Center and scale point cloud so max radius = 1.
    """
    center = coords.mean(axis=0, keepdims=True)
    c = coords - center
    scale = np.max(np.linalg.norm(c, axis=1))
    if scale < 1e-12:
        scale = 1.0
    return c / scale


def cartesian_to_spherical(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns r, theta, phi from Cartesian coordinates.
      theta in [0, pi], phi in [0, 2pi)
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x * x + y * y + z * z)

    # Avoid division by zero for r=0
    z_over_r = z / np.where(r > 1e-12, r, 1.0)
    z_over_r = np.clip(z_over_r, -1.0, 1.0)

    theta = np.arccos(z_over_r)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return r, theta, phi


def zernike_invariants_from_pointcloud(
    coords: np.ndarray,
    weights: np.ndarray,
    n_max: int = 8,
) -> np.ndarray:
    """
    Compute rotationally invariant 3D Zernike descriptor vector F_{n,l}
    from weighted point cloud samples in the unit ball.

    F_{n,l} = sqrt(sum_{m=-l..l} |Omega_{n,l,m}|^2)
    """
    if coords.shape[0] != weights.shape[0]:
        raise ValueError("coords and weights must have the same number of points.")

    coords_u = to_unit_ball(coords)
    r, theta, phi = cartesian_to_spherical(coords_u)

    mask = r <= 1.0 + 1e-12
    r, theta, phi, w = r[mask], theta[mask], phi[mask], weights[mask]

    if w.sum() <= 1e-12:
        # descriptor length = number of valid (n,l) pairs up to n_max
        n_terms = sum(1 for n in range(n_max + 1) for l in range(n + 1) if (n - l) % 2 == 0)
        return np.zeros(n_terms, dtype=np.float64)

    # Normalize to make descriptors less sensitive to point count/intensity scale
    w = w / (w.sum() + 1e-12)

    invariants = []
    for n in range(n_max + 1):
        for l in range(n + 1):
            if (n - l) % 2 != 0:
                continue

            moments = []
            for m in range(-l, l + 1):
                Z_nlm = zernike_basis_3d(n, l, m, r, theta, phi)
                omega_nlm = np.sum(w * np.conjugate(Z_nlm))
                moments.append(omega_nlm)

            moments = np.asarray(moments)
            F_nl = np.sqrt(np.sum(np.abs(moments) ** 2)).real
            invariants.append(F_nl)

    return np.asarray(invariants, dtype=np.float64)


def zernike_pos_neg_from_df(
    df: pd.DataFrame,
    n_max: int = 8,
    xyz_cols: tuple[str, str, str] = ("x", "y", "z"),
    esp_col: str = "esp",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Main function:
      - positive descriptor from esp_pos = max(esp, 0)
      - negative descriptor from esp_neg = max(-esp, 0)
    """
    required = set(xyz_cols + (esp_col,))
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    coords = df.loc[:, xyz_cols].to_numpy(np.float64)
    esp = df[esp_col].to_numpy(np.float64)

    esp_pos = np.clip(esp, 0.0, None)
    esp_neg = np.clip(-esp, 0.0, None)  # absolute magnitude of negative ESP

    z_pos = zernike_invariants_from_pointcloud(coords, esp_pos, n_max=n_max)
    z_neg = zernike_invariants_from_pointcloud(coords, esp_neg, n_max=n_max)

    return z_pos, z_neg


def main():
    parser = argparse.ArgumentParser(
        description="Compute 3D Zernike descriptors (positive/negative ESP) from point-cloud CSVs."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with downsampled point-cloud CSVs.")
    parser.add_argument("--output-positive", required=True,
                        help="Output CSV for positive-ESP Zernike descriptors.")
    parser.add_argument("--output-negative", required=True,
                        help="Output CSV for negative-ESP Zernike descriptors.")
    parser.add_argument("--n-max", type=int, default=8,
                        help="Maximum Zernike order (default: 8).")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    z_pos_mol_dict = {}
    z_neg_mol_dict = {}
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue
        abs_path = input_dir / file_name
        mol_idx = abs_path.stem.split('_')[0]
        print(f'Processing molecule {mol_idx}...')
        df = pd.read_csv(abs_path)
        z_pos, z_neg = zernike_pos_neg_from_df(df, n_max=args.n_max)
        z_pos_mol_dict[mol_idx] = z_pos
        z_neg_mol_dict[mol_idx] = z_neg

    df_z_pos = pd.DataFrame.from_dict(z_pos_mol_dict, orient='index')
    df_z_neg = pd.DataFrame.from_dict(z_neg_mol_dict, orient='index')

    for out_path in (args.output_positive, args.output_negative):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    df_z_pos.to_csv(args.output_positive)
    df_z_neg.to_csv(args.output_negative)


if __name__ == "__main__":
    main()

# %%
