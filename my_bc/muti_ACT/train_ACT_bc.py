import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from my_bc.muti_ACT.ACT_model import ACT
from my_bc.muti_ACT.sequence_dataset import LiberoSequenceDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "libero" / "datasets" / "libero_spatial"
DEFAULT_CHECKPOINT = REPO_ROOT / "multi_act_policy.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one task-conditioned ACT policy on several LIBERO tasks."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Use the first N sorted task files; by default use every task.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unbalanced-sampling",
        action="store_true",
        help="Sample transitions uniformly instead of balancing task IDs.",
    )
    return parser.parse_args()


def demo_sort_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def discover_tasks(dataset_dir, num_tasks, val_fraction, seed):
    paths = sorted(dataset_dir.glob("*_demo.hdf5"))
    if num_tasks is not None:
        if num_tasks < 1:
            raise ValueError("--num-tasks must be at least 1")
        paths = paths[:num_tasks]
    if not paths:
        raise RuntimeError(f"no *_demo.hdf5 files found in {dataset_dir}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    train_specs = []
    val_specs = []
    task_infos = []

    for task_id, path in enumerate(paths):
        with h5py.File(path, "r") as file:
            demo_names = sorted(file["data"].keys(), key=demo_sort_key)
        if len(demo_names) < 2:
            raise RuntimeError(
                f"task {path.name} needs at least two demos for train/validation"
            )

        rng = np.random.default_rng(seed + task_id)
        rng.shuffle(demo_names)
        val_count = max(1, int(round(len(demo_names) * val_fraction)))
        val_count = min(val_count, len(demo_names) - 1)
        val_demos = demo_names[:val_count]
        train_demos = demo_names[val_count:]
        task_name = path.name[:-len("_demo.hdf5")]

        common = {
            "path": str(path.resolve()),
            "task_id": task_id,
            "task_name": task_name,
        }
        train_specs.append({**common, "demo_names": train_demos})
        val_specs.append({**common, "demo_names": val_demos})
        task_infos.append(
            {
                **common,
                "train_demos": train_demos,
                "val_demos": val_demos,
            }
        )

    return train_specs, val_specs, task_infos


def make_balanced_sampler(dataset, num_tasks, generator):
    counts = np.bincount(dataset.sample_task_ids, minlength=num_tasks)
    if np.any(counts == 0):
        raise RuntimeError(f"some tasks have no samples: counts={counts.tolist()}")
    sample_weights = 1.0 / counts[dataset.sample_task_ids]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def masked_l1_loss(pred, target, is_pad):
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes differ: {pred.shape} vs {target.shape}"
        )
    valid = (~is_pad).unsqueeze(-1).expand_as(pred)
    if not valid.any():
        raise ValueError("batch contains no valid action targets")
    return torch.abs(pred - target)[valid].mean()


def kl_divergence(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


def move_batch(batch, device):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def add_task_errors(sums, pred, target, is_pad, task_ids, counts=None):
    errors = torch.abs(pred - target)
    valid = (~is_pad).unsqueeze(-1).expand_as(errors)
    for task_id in task_ids.unique().tolist():
        selected = task_ids == task_id
        task_valid = valid[selected]
        sums[task_id] += errors[selected][task_valid].sum().item()
        if counts is not None:
            counts[task_id] += task_valid.sum().item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_specs, val_specs, task_infos = discover_tasks(
        args.dataset_dir,
        args.num_tasks,
        args.val_fraction,
        args.seed,
    )
    num_tasks = len(task_infos)

    print(f"dataset suite: {args.dataset_dir}")
    for task in task_infos:
        print(
            f"task {task['task_id']:02d}: {task['task_name']} | "
            f"train demos={len(task['train_demos'])}, "
            f"val demos={len(task['val_demos'])}"
        )

    # Statistics are shared across tasks and are computed only from train demos.
    train_dataset = LiberoSequenceDataset(
        train_specs,
        chunk_size=args.chunk_size,
    )
    val_dataset = LiberoSequenceDataset(
        val_specs,
        chunk_size=args.chunk_size,
        stats=train_dataset.stats,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)

    sampler = None
    shuffle = True
    if not args.unbalanced_sampling:
        sampler = make_balanced_sampler(train_dataset, num_tasks, generator)
        shuffle = False

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    debug_batch = next(iter(train_loader))
    agent, wrist, state, task_id, actions, is_pad = debug_batch
    print("agent:", tuple(agent.shape))
    print("wrist:", tuple(wrist.shape))
    print("state:", tuple(state.shape))
    print("task_id:", tuple(task_id.shape), task_id.tolist())
    print("actions:", tuple(actions.shape))
    print("is_pad:", tuple(is_pad.shape))
    print("train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))
    print("device:", device)

    model_config = {
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "num_tasks": num_tasks,
        "chunk_size": args.chunk_size,
        "d_model": 256,
        "nhead": 4,
        "latent_dim": 32,
        "style_layers": 2,
        "memory_layers": 2,
        "decoder_layers": 2,
    }
    model = ACT(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    best_prior_l1 = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_l1_sum = 0.0
        train_kl_sum = 0.0
        train_valid_value_count = 0
        train_sample_count = 0

        for batch in train_loader:
            agent, wrist, state, task_id, action, is_pad = move_batch(
                batch, device
            )
            pred, mu, logvar = model(
                agent,
                wrist,
                state,
                task_id,
                action_chunk=action,
                is_pad=is_pad,
                sample_latent=True,
            )

            l1 = masked_l1_loss(pred, action, is_pad)
            kl = kl_divergence(mu, logvar)
            loss = l1 + args.kl_weight * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = agent.shape[0]
            valid_value_count = (~is_pad).sum().item() * action.shape[-1]
            train_l1_sum += l1.item() * valid_value_count
            train_kl_sum += kl.item() * batch_size
            train_valid_value_count += valid_value_count
            train_sample_count += batch_size

        train_l1 = train_l1_sum / train_valid_value_count
        train_kl = train_kl_sum / train_sample_count
        train_loss = train_l1 + args.kl_weight * train_kl

        model.eval()
        posterior_sums = np.zeros(num_tasks, dtype=np.float64)
        prior_sums = np.zeros(num_tasks, dtype=np.float64)
        value_counts = np.zeros(num_tasks, dtype=np.int64)

        with torch.no_grad():
            for batch in val_loader:
                agent, wrist, state, task_id, action, is_pad = move_batch(
                    batch, device
                )
                pred_post, _, _ = model(
                    agent,
                    wrist,
                    state,
                    task_id,
                    action_chunk=action,
                    is_pad=is_pad,
                    sample_latent=False,
                )
                pred_prior, _, _ = model(
                    agent,
                    wrist,
                    state,
                    task_id,
                    action_chunk=None,
                    is_pad=None,
                    sample_latent=False,
                )
                add_task_errors(
                    posterior_sums,
                    pred_post,
                    action,
                    is_pad,
                    task_id,
                )
                add_task_errors(
                    prior_sums,
                    pred_prior,
                    action,
                    is_pad,
                    task_id,
                    counts=value_counts,
                )

        # Macro averages keep every task equally important during selection.
        per_task_posterior = posterior_sums / value_counts
        per_task_prior = prior_sums / value_counts
        posterior_l1 = per_task_posterior.mean()
        prior_l1 = per_task_prior.mean()

        print(
            f"epoch {epoch:03d} | train {train_loss:.5f} | "
            f"L1 {train_l1:.5f} | KL {train_kl:.5f} | "
            f"val_post {posterior_l1:.5f} | val_prior {prior_l1:.5f}",
            flush=True,
        )
        print(
            "  per-task prior: "
            + ", ".join(
                f"{task_id}:{value:.4f}"
                for task_id, value in enumerate(per_task_prior)
            ),
            flush=True,
        )

        if prior_l1 < best_prior_l1:
            best_prior_l1 = float(prior_l1)
            serialized_args = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            checkpoint_stats = {
                key: torch.from_numpy(value.copy())
                for key, value in train_dataset.stats.items()
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": model_config,
                    "stats": checkpoint_stats,
                    "suite_name": args.dataset_dir.name,
                    "tasks": task_infos,
                    "best_prior_l1": best_prior_l1,
                    "per_task_prior_l1": per_task_prior.tolist(),
                    "epoch": epoch,
                    "train_args": serialized_args,
                },
                args.checkpoint,
            )
            print(f"  saved: {args.checkpoint}", flush=True)

    train_dataset.close()
    val_dataset.close()
    print("best prior val L1:", best_prior_l1)


if __name__ == "__main__":
    main()
