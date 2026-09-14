import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def make_state(obs):
    return np.concatenate(
        [
            obs["ee_pos"],
            obs["ee_ori"],
            obs["joint_states"],
            obs["gripper_states"],
        ],
        axis=-1,
    )

class LiberostateDataset(Dataset):
    def __init__(self,hdf5_path,demo_names,state_mean=None,state_std=None):
        xs=[]
        ys=[]

        with h5py.File(hdf5_path,"r") as f:
            # 做数据集
            for demo_name in demo_names:
                demo =f["data"][demo_name]
                obs=demo["obs"]

                state=make_state(obs)
                action =demo["actions"][:]

                # 错位
                # 当前
                x=state[:-1]
                # 下一时刻
                y=action[1:]

                xs.append(x)
                ys.append(y)

            # 列表转数组  转float32对齐pytorch    
            self.x=np.concatenate(xs,axis=0).astype(np.float32)
            self.y=np.concatenate(ys,axis=0).astype(np.float32)

            if state_mean is None:
                state_mean=self.x.mean(axis=0)
            if state_std is None:
                state_std=self.x.std(axis=0)

            state_std = np.maximum(state_std,1e-6)

            # 可以不加
            self.state_mean=state_mean.astype(np.float32)
            self.state_std=state_std.astype(np.float32)

            self.x=(self.x-self.state_mean)/self.state_std

            print("X_shape",self.x.shape)
            print("Y_shape",self.y.shape)

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        x=torch.from_numpy(self.x[index])
        y=torch.from_numpy(self.y[index])

        return x,y
            