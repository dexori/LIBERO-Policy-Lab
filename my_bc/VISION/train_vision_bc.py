import glob
import h5py
import numpy as np
import torch

from torch.utils.data import DataLoader

from my_bc.VISION.vision_dataset import LiberoVisionDataset
from my_bc.VISION.vision_model import VisionBCPolicy

BATCH_SIZE=64
EPORCH=50
LR=3e-4

torch.manual_seed(0)
np.random.seed(0)

files=glob.glob(
     "../LIBERO/libero/datasets/libero_spatial/*.hdf5"
)

print(files)

if len(files)==0:
    raise RuntimeError(
        "找不到 libero_spatial dataset"
    )

path=files[0]

print("Dataset:",path)

with h5py.File(path,"r") as f:
    demos=list(f["data"].keys())

demos=sorted(
    demos,
    key=lambda x:
    int(x.split("_")[-1])
)

np.random.shuffle(demos)

split=int(len(demos)*0.8)

train_demos=demos[:split]
val_demos=demos[split:]

print("train_demos:",len(train_demos))
print("val_demos:",len(val_demos))

train_dataset=LiberoVisionDataset(path,train_demos)
val_dataset=LiberoVisionDataset(path,val_demos,train_dataset.state_mean,train_dataset.state_std)

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model=VisionBCPolicy().to(device)

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

loss_fn=torch.nn.MSELoss()

best_val=float("inf")

for epoch in range(EPORCH):

    model.train()

    train_loss=0

    for agent_image,wrist_image,state,action in train_loader:
        agent_image=agent_image.to(device)
        wrist_image=wrist_image.to(device)
        state=state.to(device)
        action=action.to(device)

        pred=model(agent_image,wrist_image,state)

        loss=loss_fn(pred,action)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss+=loss.item()*agent_image.size(0)

    train_loss/=len(train_dataset)

    model.eval()

    val_loss=0
    with torch.no_grad():
        for agent_image,wrist_image,state,action in val_loader:
            agent_image=agent_image.to(device)
            wrist_image=wrist_image.to(device)
            state=state.to(device)
            action=action.to(device)
    
            pred=model(agent_image,wrist_image,state)
    
            loss=loss_fn(pred,action)
    
            val_loss+=loss.item()*agent_image.size(0)

    val_loss/=len(val_dataset)

    print(
        f"epoch {epoch:03d} | "
        f"train {train_loss:.6f} | "
        f"val {val_loss:.6f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "model":
                    model.state_dict(),

                "state_mean":
                    train_dataset.state_mean,

                "state_std":
                    train_dataset.state_std,

                "val_loss":
                    best_val,
            },
            "vision_bc_policy.pt",
        )

print(
    "best val:",
    best_val
)

print(
    "saved vision_bc_policy.pt"
)





