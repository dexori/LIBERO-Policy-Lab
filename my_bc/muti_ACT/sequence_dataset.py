import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def make_state(obs, index=slice(None)):
    return np.concatenate(
        [
            obs["ee_pos"][index],          # 3
            obs["ee_ori"][index],          # 3
            obs["joint_states"][index],    # 7
            obs["gripper_states"][index],  # 2
        ],
        axis=-1,
    ).astype(np.float32)


def process_image(image):
    image = torch.from_numpy(np.asarray(image).copy()).float()
    image = image.div(255.0).permute(2, 0, 1)
    return image.sub(0.5).div(0.5)


class LiberoSequenceDataset(Dataset):
    """ACT samples pooled from several LIBERO task HDF5 files.

    Images are read lazily. Loading all RGB frames from ten tasks into RAM can
    otherwise consume several gigabytes. ``task_specs`` is a list of dicts with
    ``path``, ``task_id`` and ``demo_names`` fields.
    """

    def __init__(self, task_specs, chunk_size=10, stats=None):
        if not task_specs:
            raise ValueError("task_specs cannot be empty")

        self.chunk_size = chunk_size
        self.samples = []
        self.demo_data = []
        self._files = {}

        all_states = []
        all_actions = []
        state_dim = None
        action_dim = None

        for spec in task_specs:
            path = str(spec["path"])
            task_id = int(spec["task_id"])
            demo_names = list(spec["demo_names"])

            if not demo_names:
                raise ValueError(f"task {task_id} has no demonstrations")

            with h5py.File(path, "r") as file:
                for demo_name in demo_names:
                    demo = file["data"][demo_name]
                    obs = demo["obs"]
                    state = make_state(obs)
                    actions = demo["actions"][:].astype(np.float32)

                    if len(state) != len(actions):
                        raise ValueError(
                            f"{path}:{demo_name} has {len(state)} states but "
                            f"{len(actions)} actions"
                        )
                    if len(actions) < 2:
                        continue

                    current_state_dim = state.shape[-1]
                    current_action_dim = actions.shape[-1]
                    state_dim = current_state_dim if state_dim is None else state_dim
                    action_dim = current_action_dim if action_dim is None else action_dim
                    if current_state_dim != state_dim or current_action_dim != action_dim:
                        raise ValueError(
                            "all tasks must use the same state and action dimensions"
                        )

                    demo_id = len(self.demo_data)
                    self.demo_data.append(
                        {
                            "path": path,
                            "demo_name": demo_name,
                            "task_id": task_id,
                            "length": len(actions),
                        }
                    )

                    # State at t predicts actions beginning at t + 1.
                    all_states.append(state[:-1])
                    all_actions.append(actions[1:])
                    self.samples.extend(
                        (demo_id, t) for t in range(len(actions) - 1)
                    )

        if not self.samples:
            raise ValueError("no valid training samples were found")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sample_task_ids = np.asarray(
            [self.demo_data[demo_id]["task_id"] for demo_id, _ in self.samples],
            dtype=np.int64,
        )

        if stats is None:
            states_cat = np.concatenate(all_states, axis=0)
            actions_cat = np.concatenate(all_actions, axis=0)
            stats = {
                "state_mean": states_cat.mean(axis=0),
                "state_std": states_cat.std(axis=0),
                "action_mean": actions_cat.mean(axis=0),
                "action_std": actions_cat.std(axis=0),
            }

        self.stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in stats.items()
        }
        self.stats["state_std"] = np.maximum(
            self.stats["state_std"], 1e-6
        )
        self.stats["action_std"] = np.maximum(
            self.stats["action_std"], 1e-6
        )

        expected_shapes = {
            "state_mean": (self.state_dim,),
            "state_std": (self.state_dim,),
            "action_mean": (self.action_dim,),
            "action_std": (self.action_dim,),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in self.stats or self.stats[key].shape != expected_shape:
                raise ValueError(
                    f"stats[{key!r}] must have shape {expected_shape}"
                )

    def __len__(self):
        return len(self.samples)

    def _get_file(self, path):
        # Each DataLoader worker owns its own dataset copy and HDF5 handles.
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def close(self):
        for file in self._files.values():
            file.close()
        self._files.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        demo_id, t = self.samples[index]
        demo_info = self.demo_data[demo_id]
        demo = self._get_file(demo_info["path"])["data"][demo_info["demo_name"]]
        obs = demo["obs"]

        agent_image = obs["agentview_rgb"][t]
        wrist_image = obs["eye_in_hand_rgb"][t]
        current_state = make_state(obs, t)

        actions = demo["actions"]
        future_actions = actions[t + 1:t + 1 + self.chunk_size].astype(
            np.float32
        )
        valid_len = len(future_actions)

        current_state = (
            current_state - self.stats["state_mean"]
        ) / self.stats["state_std"]
        future_actions = (
            future_actions - self.stats["action_mean"]
        ) / self.stats["action_std"]

        action_chunk = np.zeros(
            (self.chunk_size, self.action_dim), dtype=np.float32
        )
        action_chunk[:valid_len] = future_actions

        is_pad = np.ones(self.chunk_size, dtype=np.bool_)
        is_pad[:valid_len] = False

        return (
            process_image(agent_image),
            process_image(wrist_image),
            torch.from_numpy(current_state.astype(np.float32)),
            torch.tensor(demo_info["task_id"], dtype=torch.long),
            torch.from_numpy(action_chunk),
            torch.from_numpy(is_pad),
        )
