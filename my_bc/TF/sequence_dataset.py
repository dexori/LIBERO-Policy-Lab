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
class LiberoSequenceDataset(Dataset):
    def __init__(
            self,
            hdf5_path,
            demo_names,
            seq_len=10,
            state_mean=None,
            state_std=None
    ):
        self.seq_len=seq_len
        self.samples=[]

        self.demo_data=[]

        all_states=[]

        with h5py.File(hdf5_path,"r") as f:

            for demo_name in demo_names:
                demo=f["data"][demo_name] 
                obs=demo["obs"]

                agent_image=obs["agentview_rgb"][:]
                wrist_image=obs["eye_in_hand_rgb"][:]
                state=make_state(obs)
                action=demo["actions"][:]

                self.demo_data.append(
                    (
                        agent_image,
                        wrist_image,
                        state,
                        action
                    )
                )

                all_states.append(
                    state[:-1]
                )

        all_states=np.concatenate(
            all_states,
            axis=0
        )

        if state_mean is None:
            state_mean = all_states.mean(
                axis=0
            )

        if state_std is None:
            state_std = all_states.std(
                axis=0
            )

        self.state_mean = (
            state_mean.astype(np.float32)
        )

        self.state_std = np.maximum(
            state_std,
            1e-6,
        ).astype(np.float32)

        for demo_id, (agent_image,wrist_image,state,actions) in enumerate(self.demo_data):

            T = len(actions)

            max_start = (
                T - self.seq_len - 1
            )

            for start in range(max_start + 1):
                self.samples.append((demo_id, start))

        print(
            "num sequence samples:",
            len(self.samples)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        demo_id,start=(self.samples[index])

        agent_image,wrist_image,state,actions= self.demo_data[demo_id]

        L=self.seq_len

        agent_image_seq=agent_image[
            start:start+L
        ]

        wrist_image_seq=wrist_image[
            start:start+L
        ]

        state_seq=state[
            start:start+L
        ]

        action_seq=actions[
            start+1:start+L+1
        ]

        state_seq=(state_seq-self.state_mean)/self.state_std

        agent_image_seq = (
            torch.from_numpy(
                agent_image_seq.copy()
            )
            .float()
            / 255.0
        )
        agent_image_seq = (
            agent_image_seq.permute(
                0, 3, 1, 2
            )
        )
        agent_image_seq = (
            agent_image_seq - 0.5
        ) / 0.5

        wrist_image_seq = (
            torch.from_numpy(
                wrist_image_seq.copy()
            )
            .float()
            / 255.0
        )
        wrist_image_seq = (
            wrist_image_seq.permute(
                0, 3, 1, 2
            )
        )
        wrist_image_seq = (
            wrist_image_seq - 0.5
        ) / 0.5

        state_seq=torch.from_numpy(
            state_seq.astype(np.float32)
        )

        action_seq=torch.from_numpy(
            action_seq.astype(np.float32)
        )
        return agent_image_seq,wrist_image_seq,state_seq,action_seq
