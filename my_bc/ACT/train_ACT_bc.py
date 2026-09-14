import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from my_bc.ACT.ACT_model import ACT
from my_bc.ACT.sequence_dataset import LiberoSequenceDataset


BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
CHUNK_SIZE = 10
KL_WEIGHT = 1.0

DATASET_PATH = (
    "../LIBERO/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
CHECKPOINT_PATH = "act_policy.pt"


torch.manual_seed(0)
np.random.seed(0)


def masked_l1_loss(pred, target, is_pad):
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes differ: {pred.shape} vs {target.shape}"
        )

    valid = (~is_pad).unsqueeze(-1).expand_as(pred)
    return torch.abs(pred - target)[valid].mean()


def kl_divergence(mu, logvar):
    kl = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return kl.sum(dim=-1).mean()


if not os.path.isfile(DATASET_PATH):
    raise RuntimeError(f"找不到 dataset: {DATASET_PATH}")

path = DATASET_PATH
print("Dataset:", path)

with h5py.File(path, "r") as file:
    demos = list(file["data"].keys())

demos = sorted(
    demos,
    key=lambda name: int(name.split("_")[-1]),
)
np.random.shuffle(demos)

split = int(len(demos) * 0.8)
train_demos = demos[:split]
val_demos = demos[split:]

print("train_demos:", len(train_demos))
print("val_demos:", len(val_demos))

train_dataset = LiberoSequenceDataset(
    path,
    train_demos,
    chunk_size=CHUNK_SIZE,
)
val_dataset = LiberoSequenceDataset(
    path,
    val_demos,
    chunk_size=CHUNK_SIZE,
    stats=train_dataset.stats,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory,
)

debug_batch = next(iter(train_loader))
debug_agent, debug_wrist, debug_state, debug_actions, debug_is_pad = (
    debug_batch
)
print("agent:", debug_agent.shape)
print("wrist:", debug_wrist.shape)
print("state:", debug_state.shape)
print("actions:", debug_actions.shape)
print("is_pad:", debug_is_pad.shape)
print("device:", device)

model_config = {
    "state_dim": 15,
    "action_dim": 7,
    "chunk_size": CHUNK_SIZE,
    "d_model": 256,
    "nhead": 4,
    "latent_dim": 32,
    "style_layers": 2,
    "memory_layers": 2,
    "decoder_layers": 2,
}

model = ACT(**model_config).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

best_prior_l1 = float("inf")

for epoch in range(EPOCHS):
    model.train()

    train_loss_sum = 0.0
    train_l1_sum = 0.0
    train_kl_sum = 0.0
    train_sample_count = 0
    train_valid_value_count = 0

    for agent_image, wrist_image, state, action, is_pad in train_loader:
        agent_image = agent_image.to(
            device, non_blocking=True
        )
        wrist_image = wrist_image.to(
            device, non_blocking=True
        )
        state = state.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        is_pad = is_pad.to(device, non_blocking=True)

        pred, mu, logvar = model(
            agent_image,
            wrist_image,
            state,
            action_chunk=action,
            is_pad=is_pad,
            sample_latent=True,
        )

        l1 = masked_l1_loss(pred, action, is_pad)
        kl = kl_divergence(mu, logvar)
        loss = l1 + KL_WEIGHT * kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()

        batch_size = agent_image.shape[0]
        valid_value_count = (
            (~is_pad).sum().item() * action.shape[-1]
        )

        train_loss_sum += loss.item() * batch_size
        train_l1_sum += l1.item() * valid_value_count
        train_kl_sum += kl.item() * batch_size
        train_sample_count += batch_size
        train_valid_value_count += valid_value_count

    train_loss_mean = train_loss_sum / train_sample_count
    train_l1_mean = train_l1_sum / train_valid_value_count
    train_kl_mean = train_kl_sum / train_sample_count

    model.eval()
    posterior_l1_sum = 0.0
    prior_l1_sum = 0.0
    val_valid_value_count = 0

    with torch.no_grad():
        for agent_image, wrist_image, state, action, is_pad in val_loader:
            agent_image = agent_image.to(
                device, non_blocking=True
            )
            wrist_image = wrist_image.to(
                device, non_blocking=True
            )
            state = state.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            is_pad = is_pad.to(device, non_blocking=True)

            # Deterministic posterior: use the mean of q(z | state, actions).
            pred_post, _, _ = model(
                agent_image,
                wrist_image,
                state,
                action_chunk=action,
                is_pad=is_pad,
                sample_latent=False,
            )
            posterior_l1 = masked_l1_loss(
                pred_post,
                action,
                is_pad,
            )

            # Inference path: expert actions are unavailable, so ACT uses z=0.
            pred_prior, _, _ = model(
                agent_image,
                wrist_image,
                state,
                action_chunk=None,
                is_pad=None,
                sample_latent=False,
            )
            prior_l1 = masked_l1_loss(
                pred_prior,
                action,
                is_pad,
            )

            valid_value_count = (
                (~is_pad).sum().item() * action.shape[-1]
            )
            posterior_l1_sum += (
                posterior_l1.item() * valid_value_count
            )
            prior_l1_sum += prior_l1.item() * valid_value_count
            val_valid_value_count += valid_value_count

    posterior_l1_mean = (
        posterior_l1_sum / val_valid_value_count
    )
    prior_l1_mean = prior_l1_sum / val_valid_value_count

    print(
        f"epoch {epoch:03d} | "
        f"train {train_loss_mean:.5f} | "
        f"L1 {train_l1_mean:.5f} | "
        f"KL {train_kl_mean:.5f} | "
        f"val_post {posterior_l1_mean:.5f} | "
        f"val_prior {prior_l1_mean:.5f}",
        flush=True,
    )

    if prior_l1_mean < best_prior_l1:
        best_prior_l1 = prior_l1_mean
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": model_config,
                "stats": train_dataset.stats,
                "dataset_path": os.path.abspath(path),
                "train_demos": train_demos,
                "val_demos": val_demos,
                "best_prior_l1": best_prior_l1,
                "epoch": epoch,
            },
            CHECKPOINT_PATH,
        )

print("best prior val L1:", best_prior_l1)
