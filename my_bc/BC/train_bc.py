import glob
import h5py
import numpy as np
import torch

from torch.utils.data import DataLoader

from my_bc.BC.bc_dataset import LiberostateDataset
from my_bc.BC.bc_model import BCPolicy

BATCH_SIZE =256
EPOCHS=100
LR=1e-3

torch.manual_seed(0)
np.random.seed(0)

files = glob.glob(
    "../LIBERO/libero/datasets/libero_spatial/*.hdf5"
)

if len(files) == 0:
    raise RuntimeError("没有找到 libero_spatial hdf5")

path = files[0]

print("Using dataset:")
print(path)

with h5py.File(path, "r") as f:
    demo_names = list(f["data"].keys())

# 排序一下
demo_names = sorted(
    demo_names,
    key=lambda x: int(x.split("_")[-1])
)

np.random.shuffle(demo_names)

split = int(len(demo_names) * 0.8)

# 划分数据集和验证集
train_demos = demo_names[:split]
val_demos = demo_names[split:]

print("num train demos:", len(train_demos))
print("num val demos:", len(val_demos))

train_dataset = LiberostateDataset(
    path,
    train_demos,
)

val_dataset = LiberostateDataset(
    path,
    val_demos,
    state_mean=train_dataset.state_mean,
    state_std=train_dataset.state_std,
)
mean_action = train_dataset.y.mean(axis=0)

baseline_mse = np.mean(
    (val_dataset.y - mean_action) ** 2
)

print("Mean-action baseline MSE:", baseline_mse)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)

model = BCPolicy().to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
)

loss_fn = torch.nn.MSELoss()

best_val = float("inf")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for obs,expert_action in train_loader:

        obs=obs.to(device)
        expert_action=expert_action.to(device)

        pred_action=model(obs)

        loss=loss_fn(pred_action,expert_action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=(loss.item()*obs.size(0))

    train_loss/=len(train_dataset)

    model.eval()

    val_loss = 0.0

    sq_error_sum = np.zeros(7)
    count = 0

    with torch.no_grad():

        for obs, expert_action in val_loader:

            obs = obs.to(device)
            expert_action = expert_action.to(device)

            pred_action = model(obs)
            error = (pred_action - expert_action) ** 2

            sq_error_sum += error.sum(dim=0).cpu().numpy()
            count += obs.shape[0]

            loss = loss_fn(
                pred_action,
                expert_action,
            )

            val_loss += (
                loss.item() * obs.size(0)
            )

    val_loss /= len(val_dataset)
    mse_per_dim = sq_error_sum / count

    names = [
        "dx", "dy", "dz",
        "dRx", "dRy", "dRz",
        "gripper"
    ]

    for name, mse in zip(names, mse_per_dim):
        print(f"{name:8s}: {mse:.6f}")

    # print(
    #     f"epoch {epoch:03d} | "
    #     f"train {train_loss:.6f} | "
    #     f"val {val_loss:.6f}"
    # )

    if val_loss < best_val:

        best_val = val_loss

        torch.save(
            {
                "model": model.state_dict(),

                "state_mean":
                    train_dataset.state_mean,

                "state_std":
                    train_dataset.state_std,

                "val_loss":
                    best_val,
            },
            "bc_policy.pt",
        )


print()
print("best val loss:", best_val)
print("saved: bc_policy.pt")

