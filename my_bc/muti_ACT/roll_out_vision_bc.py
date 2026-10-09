import os

os.environ["MUJOCO_GL"] = "glx"

import numpy as np
import torch
import robosuite.utils.transform_utils as T

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from my_bc.muti_ACT.ACT_model import ACT


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "multi_act_policy.pt")
# This is the contiguous task ID used by the policy's task embedding. Change it
# to evaluate another task saved in the same checkpoint.
POLICY_TASK_ID = 0
MAX_STEPS = 220
WAIT_STEPS = 10
NUM_EPISODES = 10
TEMPORAL_ENSEMBLE_COEFF = 0.01
BINARIZE_GRIPPER = True


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if not os.path.isfile(CHECKPOINT_PATH):
    raise RuntimeError(
        f"找不到 checkpoint: {CHECKPOINT_PATH}，请先完成 ACT 训练"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    # Keep normalization statistics on CPU for NumPy preprocessing. Loading
    # the state dict below will still copy model parameters to ``device``.
    map_location="cpu",
)

model_config = checkpoint["model_config"]
chunk_size = model_config["chunk_size"]
action_dim = model_config["action_dim"]

model = ACT(**model_config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

stats = checkpoint["stats"]
state_mean = np.asarray(
    stats["state_mean"], dtype=np.float32
)
state_std = np.asarray(
    stats["state_std"], dtype=np.float32
)
action_mean = np.asarray(
    stats["action_mean"], dtype=np.float32
)
action_std = np.asarray(
    stats["action_std"], dtype=np.float32
)

tasks = checkpoint["tasks"]
if not 0 <= POLICY_TASK_ID < len(tasks):
    raise ValueError(
        f"POLICY_TASK_ID must be in [0, {len(tasks) - 1}]"
    )

task_info = tasks[POLICY_TASK_ID]
dataset_path = task_info["path"]
dataset_task_name = task_info["task_name"]

print("device:", device)
print("checkpoint epoch:", checkpoint["epoch"])
print("checkpoint prior L1:", checkpoint["best_prior_l1"])
print("dataset:", dataset_path)
print("policy task id:", POLICY_TASK_ID)


benchmark_dict = benchmark.get_benchmark_dict()
suite_name = checkpoint.get("suite_name", "libero_spatial")
suite = benchmark_dict[suite_name]()

benchmark_task_id = None
for index in range(suite.n_tasks):
    if suite.get_task(index).name == dataset_task_name:
        benchmark_task_id = index
        break

if benchmark_task_id is None:
    raise RuntimeError(
        f"找不到对应 LIBERO task: {dataset_task_name}"
    )

task = suite.get_task(benchmark_task_id)
print("benchmark task id:", benchmark_task_id)
print("task name:", task.name)
print("language:", task.language)

bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

env = ControlEnv(
    bddl_file_name=bddl_file,
    camera_heights=128,
    camera_widths=128,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)
env.seed(0)


def build_state(obs):
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            T.quat2axisangle(
                obs["robot0_eef_quat"].copy()
            ),
            obs["robot0_joint_pos"],
            obs["robot0_gripper_qpos"],
        ],
        axis=0,
    ).astype(np.float32)


def image_to_tensor(image):
    tensor = torch.from_numpy(
        image.copy()
    ).float()
    tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)


def preprocess_observation(obs):
    agent_image = image_to_tensor(
        obs["agentview_image"]
    )
    wrist_image = image_to_tensor(
        obs["robot0_eye_in_hand_image"]
    )

    state = build_state(obs)
    state = (state - state_mean) / state_std
    state = torch.from_numpy(
        state.astype(np.float32)
    ).unsqueeze(0).to(device)

    return agent_image, wrist_image, state


init_states = suite.get_task_init_states(benchmark_task_id)
num_episodes = min(NUM_EPISODES, len(init_states))
success_count = 0

try:
    for episode in range(num_episodes):
        obs = env.reset()
        obs = env.set_init_state(
            init_states[episode]
        )

        dummy_action = np.array(
            [0, 0, 0, 0, 0, 0, -1],
            dtype=np.float32,
        )
        for _ in range(WAIT_STEPS):
            obs, _, _, _ = env.step(dummy_action)

        # Row q contains the chunk predicted at query time q. Column t
        # contains that chunk's prediction for environment time t.
        all_time_actions = np.zeros(
            (
                MAX_STEPS,
                MAX_STEPS + chunk_size,
                action_dim,
            ),
            dtype=np.float32,
        )
        all_time_valid = np.zeros(
            (
                MAX_STEPS,
                MAX_STEPS + chunk_size,
            ),
            dtype=bool,
        )

        success = False

        for step in range(MAX_STEPS):
            agent_image, wrist_image, state = (
                preprocess_observation(obs)
            )

            # This ACT was trained from the current observation, not a
            # history sequence. Query a fresh action chunk every step.
            with torch.no_grad():
                normalized_chunk, _, _ = model(
                    agent_image,
                    wrist_image,
                    state,
                    torch.full(
                        (state.shape[0],),
                        POLICY_TASK_ID,
                        dtype=torch.long,
                        device=device,
                    ),
                    action_chunk=None,
                    is_pad=None,
                    sample_latent=False,
                )

            normalized_chunk = (
                normalized_chunk[0].cpu().numpy()
            )

            # The dataset standardized action targets during training.
            # Convert predictions back to LIBERO's raw action space before
            # temporal ensembling and clipping.
            action_chunk = (
                normalized_chunk * action_std
                + action_mean
            )

            all_time_actions[
                step,
                step:step + chunk_size,
            ] = action_chunk
            all_time_valid[
                step,
                step:step + chunk_size,
            ] = True

            valid_queries = all_time_valid[:, step]
            actions_for_current_step = all_time_actions[
                valid_queries,
                step,
            ]

            weights = np.exp(
                -TEMPORAL_ENSEMBLE_COEFF
                * np.arange(
                    len(actions_for_current_step),
                    dtype=np.float32,
                )
            )
            weights /= weights.sum()

            action = np.sum(
                actions_for_current_step
                * weights[:, None],
                axis=0,
            )
            action = np.clip(
                action,
                -1.0,
                1.0,
            ).astype(np.float32)

            if BINARIZE_GRIPPER:
                action[-1] = (
                    1.0 if action[-1] >= 0.0 else -1.0
                )

            obs, _, done, _ = env.step(action)
            env.env.render()

            if step % 20 == 0:
                print(
                    f"episode={episode:02d}",
                    f"step={step:03d}",
                    "| action=",
                    np.round(action, 3),
                    "| ee_pos=",
                    np.round(
                        obs["robot0_eef_pos"],
                        3,
                    ),
                )

            if done:
                success_count += 1
                success = True
                print(
                    f"SUCCESS episode={episode} at step {step}"
                )
                break

        if not success:
            print(f"FAILED episode={episode}")
finally:
    env.close()

print(
    f"success: {success_count}/{num_episodes} "
    f"({success_count / num_episodes:.1%})"
)
