import numpy as np
import torch


@torch.no_grad()
def estimate_radius_candidates_from_training_data(
    train_dataset,
    k_for_estimation: int = 16,
    quantile: float = 0.5,
    max_graphs: int = 200,
    max_points_per_graph: int = 512,
    radius_multipliers: tuple = (0.8, 1.0, 1.2),
):
    """
    Estimate a sensible neighbourhood radius from geometry in the training data.

    Strategy:
      - Sample up to ``max_graphs`` molecules from ``train_dataset``.
      - For each sampled molecule, estimate the distance to the k-th nearest
        neighbour for sampled points.
      - Use the chosen quantile as base radius and generate candidates via
        multipliers for grid search.
    """
    if len(train_dataset) == 0:
        return 1.0, [1.0]

    rng = np.random.default_rng(42)
    n_graphs = min(len(train_dataset), max_graphs)
    sampled_graph_indices = rng.choice(len(train_dataset), size=n_graphs, replace=False)

    kth_neighbor_distances = []

    for graph_idx in sampled_graph_indices:
        data = train_dataset[int(graph_idx)]
        if not hasattr(data, "pos") or data.pos is None:
            continue

        pos = data.pos.detach().cpu().float()
        n_points = int(pos.size(0))
        if n_points < 2:
            continue

        if n_points > max_points_per_graph:
            point_idx = torch.from_numpy(
                rng.choice(n_points, size=max_points_per_graph, replace=False)
            ).long()
            pos_query = pos[point_idx]
        else:
            pos_query = pos

        # Distances from each query point to all points in the same molecule.
        # The first nearest neighbor is the point itself (distance 0).
        dmat = torch.cdist(pos_query, pos)
        k_eff = min(max(k_for_estimation + 1, 2), n_points)
        kth = torch.topk(dmat, k=k_eff, largest=False, dim=1).values[:, -1]
        kth_neighbor_distances.append(kth.numpy())

    if len(kth_neighbor_distances) == 0:
        return 1.0, [1.0]

    all_kth = np.concatenate(kth_neighbor_distances, axis=0)
    base_radius = float(np.quantile(all_kth, quantile))
    if not np.isfinite(base_radius) or base_radius <= 0.0:
        return 1.0, [1.0]

    radius_candidates = sorted({round(base_radius * m, 6) for m in radius_multipliers if (base_radius * m) > 0.0})
    if len(radius_candidates) == 0:
        radius_candidates = [round(base_radius, 6)]

    return base_radius, radius_candidates
