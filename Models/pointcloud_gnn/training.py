import gc
import itertools
import random
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import build_dataloader


def compute_target_stats(dataset):
    """Mean and std of the targets; call on the training set only."""
    y = torch.cat([d.y.view(-1) for d in dataset]).double()
    if not torch.isfinite(y).all():
        raise ValueError(f"{int((~torch.isfinite(y)).sum())} training targets are NaN or inf")
    return y.mean().item(), max(y.std().item(), 1e-8)


def train_one_epoch(model, optimizer, loader, device, y_mean=0.0, y_std=1.0):
    """Train for one epoch on standardised targets; the returned MSE is converted back to original units."""
    model.train()
    total_mse = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred = model(data)
        target = (data.y.view(-1).to(dtype=pred.dtype) - y_mean) / y_std

        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_mse += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

    # MSE scales with the square of the target scale.
    return total_mse / max(total_graphs, 1) * y_std ** 2


@torch.no_grad()
def evaluate(model, loader, device, y_mean=0.0, y_std=1.0):
    """Evaluate regression metrics (MSE, MAE) in original target units."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)
        pred = model(data) * y_std + y_mean
        target = data.y.view(-1).to(dtype=pred.dtype)

        total_mse += F.mse_loss(pred, target, reduction="sum").item()
        total_mae += F.l1_loss(pred, target, reduction="sum").item()
        total_graphs += data.num_graphs

    denom = max(total_graphs, 1)
    return total_mse / denom, total_mae / denom


@torch.no_grad()
def extract_global_embeddings(model, loader, device, return_predictions=False):
    """
    Return (embeddings, targets, mol_ids) for each molecule in ``loader``; with
    ``return_predictions`` a fourth tensor holds the predictions in original target units.
    """
    model.eval()
    y_mean = float(model.y_mean) if hasattr(model, "y_mean") else 0.0
    y_std = float(model.y_std) if hasattr(model, "y_std") else 1.0

    embeddings_chunks = []
    target_chunks = []
    index_chunks = []
    prediction_chunks = []

    for data in loader:
        data = data.to(device)
        pred, emb = model(data, return_global_embeddings=True)

        embeddings_chunks.append(emb.detach().cpu())
        prediction_chunks.append((pred.view(-1) * y_std + y_mean).detach().cpu())
        target_chunks.append(data.y.view(-1).detach().cpu())
        index_chunks.append(data.mol_id.view(-1).detach().cpu())

    if len(embeddings_chunks) == 0:
        embedding_dim = getattr(model, "global_embedding_dim", 0)
        result = (
            torch.empty((0, embedding_dim)),
            torch.empty((0,)),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,)),
        )
    else:
        result = (
            torch.cat(embeddings_chunks, dim=0),
            torch.cat(target_chunks, dim=0),
            torch.cat(index_chunks, dim=0),
            torch.cat(prediction_chunks, dim=0),
        )
    return result if return_predictions else result[:3]


def iter_grid_configs(param_grid: Dict[str, list]):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield {k: v for k, v in zip(keys, combo)}


def run_grid_search(
    model_factory: Callable[[dict], nn.Module],
    train_dataset,
    val_dataset,
    device,
    param_grid: Dict[str, list],
    epochs: int = 25,
    num_workers: int = 0,
    logger=None,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 1e-4,
    lr_step_size: int = 20,
    lr_gamma: float = 0.5,
    max_trials: int | None = None,
    seed: int = 0,
):
    """Grid search over ``param_grid``; ``model_factory`` builds a model from a config dict."""
    best_config = None
    best_val_mse = float("inf")
    best_state = None

    y_mean, y_std = compute_target_stats(train_dataset)
    stats_msg = f"[Grid Search] Target standardisation (train set): mean={y_mean:.4f}, std={y_std:.4f}"
    print(stats_msg)
    if logger is not None:
        logger.info(stats_msg)

    configs = list(iter_grid_configs(param_grid))
    if max_trials is not None and max_trials < len(configs):
        sample_msg = f"[Grid Search] Randomly sampled {max_trials} of {len(configs)} configurations (seed={seed})"
        print(sample_msg)
        if logger is not None:
            logger.info(sample_msg)
        configs = random.Random(seed).sample(configs, max_trials)

    for trial_idx, config in enumerate(configs, start=1):
        torch.cuda.empty_cache()
        trial_start_msg = f"[Grid Search] Trial {trial_idx:03d} configuration: {config}"
        print(f"\n{trial_start_msg}")
        if logger is not None:
            logger.info(trial_start_msg)

        batch_size = int(config.get("batch_size", 16))
        train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = model_factory(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        best_val_mse_this_trial = float("inf")
        epochs_without_improvement = 0
        oom = False
        for epoch in range(1, epochs + 1):
            try:
                #skip trial with CUDA out of memory and set oom flag
                train_mse = train_one_epoch(model, optimizer, train_loader, device, y_mean, y_std)
                _, train_mae = evaluate(model, train_loader, device, y_mean, y_std)
                val_mse, val_mae = evaluate(model, val_loader, device, y_mean, y_std)
            except torch.OutOfMemoryError:
                oom = True
                break
            scheduler.step()

            if val_mse < (best_val_mse_this_trial - early_stopping_min_delta):
                best_val_mse_this_trial = val_mse
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            msg = (
                f"[Grid Search] Trial {trial_idx:03d} | Epoch {epoch:03d} "
                f"| train_mse={train_mse:.6f} | train_mae={train_mae:.6f} "
                f"| val_mse={val_mse:.6f} | val_mae={val_mae:.6f}"
            )
            print(msg)
            if logger is not None:
                logger.info(msg)

            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                stop_msg = (
                    f"[Grid Search] Trial {trial_idx:03d} early stopped at epoch {epoch:03d} "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
                )
                print(stop_msg)
                if logger is not None:
                    logger.info(stop_msg)
                break

        if oom:
            oom_msg = f"[Grid Search] Trial {trial_idx:03d} skipped: CUDA out of memory | config={config}"
            print(oom_msg)
            if logger is not None:
                logger.warning(oom_msg)
            del model, optimizer, scheduler, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            continue

        trial_end_msg = f"Grid trial {trial_idx:03d} | best_val_mse={best_val_mse_this_trial:.6f} | config={config}"
        print(trial_end_msg)
        if logger is not None:
            logger.info(trial_end_msg)

        if best_val_mse_this_trial < best_val_mse:
            best_val_mse = best_val_mse_this_trial
            best_config = config
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Release this trial's GPU memory before starting the next one.
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    return best_config, best_val_mse, best_state


def train_with_config(
    model_factory: Callable[[dict], nn.Module],
    train_dataset,
    val_dataset,
    test_dataset,
    device,
    config: Dict,
    epochs: int = 50,
    num_workers: int = 0,
    logger=None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    lr_step_size: int = 20,
    lr_gamma: float = 0.5,
):
    """Train a single model with ``config`` and report test metrics using the best-val weights."""
    batch_size = int(config.get("batch_size", 16))
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model_factory(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    y_mean, y_std = compute_target_stats(train_dataset)
    stats_msg = f"[Final Train] Target standardisation (train set): mean={y_mean:.4f}, std={y_std:.4f}"
    print(stats_msg)
    if logger is not None:
        logger.info(stats_msg)

    best_val_mse = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(model, optimizer, train_loader, device, y_mean, y_std)
        val_mse, val_mae = evaluate(model, val_loader, device, y_mean, y_std)
        scheduler.step()

        if val_mse < (best_val_mse - early_stopping_min_delta):
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        msg = f"Epoch {epoch:03d} | train_mse={train_mse:.6f} | val_mse={val_mse:.6f} | val_mae={val_mae:.6f}"
        print(msg)
        if logger is not None:
            logger.info(f"[Final Train] {msg}")

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stop_msg = (
                f"[Final Train] Early stopped at epoch {epoch:03d} "
                f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
            )
            print(stop_msg)
            if logger is not None:
                logger.info(stop_msg)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Persisted in state_dict so saved weights can be converted back to original units.
    model.register_buffer("y_mean", torch.tensor(y_mean, device=device))
    model.register_buffer("y_std", torch.tensor(y_std, device=device))

    test_mse, test_mae = evaluate(model, test_loader, device, y_mean, y_std)
    print(f"Test metrics | mse={test_mse:.6f} | mae={test_mae:.6f}")
    if logger is not None:
        logger.info(f"[Final Test] Test metrics | mse={test_mse:.6f} | mae={test_mae:.6f}")

    return model, train_loader, val_loader, test_loader
