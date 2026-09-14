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

def process_image(image):
    image = torch.from_numpy(
        image.copy()
    ).float()
    image = image / 255.0
    image = image.permute(
        2, 0, 1
    )
    image = (
        image - 0.5
    ) / 0.5
    return image


class LiberoSequenceDataset(Dataset):
    def __init__(
            self,
            hdf5_path,
            demo_names,
            seq_len=10,
            chunk_size=10,
            stats=None
    ):
        self.seq_len=seq_len
        self.chunk_size=chunk_size
        self.samples=[]

        self.demo_data=[]

        all_states=[]
        all_actions=[]
        with h5py.File(hdf5_path,"r") as f:

            for demo_name in demo_names:
                demo=f["data"][demo_name] 
                obs=demo["obs"]

                agent_image=obs["agentview_rgb"][:]
                wrist_image=obs["eye_in_hand_rgb"][:]
                state=make_state(obs)
                action=(demo["actions"][:]).astype(np.float32)

                self.demo_data.append(
                    (
                        agent_image,
                        wrist_image,
                        state,
                        action
                    )
                )

                all_states.append(state[:-1])
                all_actions.append(action[1:])

        if stats is None:

            states_cat=np.concatenate(
                all_states,
                axis=0
            )
            actions_cat=np.concatenate(
                all_actions,
                axis=0
            )

            stats={
                "state_mean":states_cat.mean(axis=0),
                "state_std":states_cat.std(axis=0)+ 1e-6,
                "action_mean":actions_cat.mean(axis=0),
                "action_std":actions_cat.std(axis=0)+ 1e-6,
            }
 
        self.stats = {}
        for key, value in stats.items():
            self.stats[key]=np.asarray(value).astype(np.float32)
            

        for demo_id, (agent_image,wrist_image,state,actions) in enumerate(self.demo_data):

            T = len(actions)

            for t in range(T - 1):

                self.samples.append(
                    (
                        demo_id,
                        t,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        demo_id,t=(self.samples[index])

        agent_img,wrist_img,state,actions= self.demo_data[demo_id]

        K=self.chunk_size

        agent_image=agent_img[t]

        wrist_image=wrist_img[t]

        current_state=(state[t].astype(np.float32))

        future_actions = actions[
            t + 1:
            t + 1 + K
        ]

        valid_len = len(
            future_actions
        )

        current_state =(current_state- self.stats["state_mean"])/self.stats["state_std"]

        if valid_len > 0:
            future_actions=(future_actions-self.stats["action_mean"])/self.stats["action_std"]

        action_chunk=np.zeros(
            (K,actions.shape[-1]),dtype=np.float32
        )

        is_pad = np.ones(
            K,
            dtype=np.bool_,
        )

        action_chunk[
            :valid_len
        ] = future_actions

        is_pad[
            :valid_len
        ] = False

        return (

            process_image(
                agent_image
            ),

            process_image(
                wrist_image
            ),

            torch.from_numpy(
                current_state
            ),

            torch.from_numpy(
                action_chunk
            ),

            torch.from_numpy(
                is_pad
            ),
        )
