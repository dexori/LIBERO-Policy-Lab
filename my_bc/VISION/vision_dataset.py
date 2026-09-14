import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def make_state(obs):
    return np.concatenate(
        [
            obs["ee_pos"][:],          # 3
            obs["ee_ori"][:],          # 3
            obs["joint_states"][:],    # 7
            obs["gripper_states"][:],  # 2
        ],
        axis=-1,
    )

class LiberoVisionDataset(Dataset):
    def __init__(self,hdf5_path,demo_names,state_mean=None,state_std=None):
        agent_images=[]
        wrist_images=[]
        images=[]
        states=[]
        actions=[]

        with h5py.File(hdf5_path,"r") as f:

            for demo_name in demo_names:

                demo=f["data"][demo_name]

                obs=demo["obs"]

                agent_image=obs["agentview_rgb"][:]
                wrist_image=obs["eye_in_hand_rgb"][:]
                state=make_state(obs)
                action=demo["actions"][:]

                agent_images.append(agent_image[:-1])
                wrist_images.append(wrist_image[:-1])
                states.append(state[:-1])
                actions.append(action[1:])

        self.agent_images=np.concatenate(
            agent_images,
            axis=0,
        ).astype(np.uint8)

        self.wrist_images=np.concatenate(
                wrist_images,
                axis=0,
            ).astype(np.uint8)

        self.states=np.concatenate(
            states,
            axis=0
        ).astype(np.float32)

        self.actions=np.concatenate(
            actions,
            axis=0
        ).astype(np.float32)

        if state_mean is None:
            state_mean=self.states.mean(
                axis=0
            )
        if state_std is None:
            state_std=self.states.std(
                axis=0
            )

        state_std=np.maximum(state_std,1e-6)

        self.state_mean=state_mean.astype(np.float32)

        self.state_std=state_std.astype(np.float32)

        self.states=(self.states-self.state_mean)/self.state_std

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):

        # (H, W, C)
        agent_images=self.agent_images[index]

        # 转cnn格式输入
        agent_images=torch.from_numpy(agent_images.copy()).float()/255

        # (C, H, W)
        agent_images=agent_images.permute(2,0,1)

        agent_images=(agent_images-0.5)/0.5

        wrist_images=self.wrist_images[index]
        
        # 转cnn格式输入
        wrist_images=torch.from_numpy(wrist_images.copy()).float()/255

        # (C, H, W)
        wrist_images=wrist_images.permute(2,0,1)

        wrist_images=(wrist_images-0.5)/0.5

        state=torch.from_numpy(self.states[index])

        action=torch.from_numpy(self.actions[index])

        return agent_images,wrist_images,state,action
        

