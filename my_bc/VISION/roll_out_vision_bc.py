import os
os.environ["MUJOCO_GL"] = "glx"

import glob
import numpy as np
import torch
import imageio.v2 as imageio

import robosuite.utils.transform_utils as T

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from my_bc.VISION.vision_model import VisionBCPolicy


# =========================================================
# Config
# =========================================================

MAX_STEPS = 220
WAIT_STEPS = 10

# gripper expert action 只有 -1 / +1
BINARIZE_GRIPPER = True


# =========================================================
# Device + model
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "vision_bc_policy.pt",
    map_location=device,
)

model = VisionBCPolicy().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

state_mean = checkpoint["state_mean"]
state_std = checkpoint["state_std"]

print("device:", device)
print("checkpoint val loss:", checkpoint["val_loss"])


# =========================================================
# Find SAME dataset used during training
# =========================================================

files = glob.glob(
    "../LIBERO/libero/datasets/libero_spatial/*.hdf5"
)

if len(files) == 0:
    raise RuntimeError("找不到 libero_spatial dataset")

dataset_path = files[0]

dataset_task_name = os.path.basename(
    dataset_path
).replace("_demo.hdf5", "")

print("\nDataset used:")
print(dataset_path)

print("\nDataset task name:")
print(dataset_task_name)


# =========================================================
# Find corresponding LIBERO task
# =========================================================

benchmark_dict = benchmark.get_benchmark_dict()
suite = benchmark_dict["libero_spatial"]()

task_id = None

for i in range(suite.n_tasks):
    task_i = suite.get_task(i)

    if task_i.name == dataset_task_name:
        task_id = i
        break

if task_id is None:
    raise RuntimeError(
        f"找不到对应 LIBERO task: {dataset_task_name}"
    )

task = suite.get_task(task_id)

print("\nMatched task:")
print("task id:", task_id)
print("task name:", task.name)
print("language:", task.language)


# =========================================================
# Create environment
# =========================================================

bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

env_args = {
    "bddl_file_name": bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}

env = ControlEnv(
    **env_args,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,)

env.seed(0)

env.reset()


# =========================================================
# Fixed benchmark initial state
# =========================================================

init_states = suite.get_task_init_states(task_id)

obs = env.set_init_state(
    init_states[0]
)


# =========================================================
# Let objects settle
# =========================================================

dummy_action = np.array(
    [0, 0, 0, 0, 0, 0, -1],
    dtype=np.float32,
)

for _ in range(WAIT_STEPS):
    obs, reward, done, info = env.step(dummy_action)


# =========================================================
# Helper: raw LIBERO obs -> our 15-D state
# =========================================================

def build_state(obs):

    ee_pos = obs["robot0_eef_pos"]

    ee_ori = T.quat2axisangle(
        obs["robot0_eef_quat"].copy()
    )

    joint = obs["robot0_joint_pos"]

    gripper = obs["robot0_gripper_qpos"]

    state = np.concatenate(
        [
            ee_pos,      # 3
            ee_ori,      # 3
            joint,       # 7
            gripper,     # 2
        ],
        axis=0,
    ).astype(np.float32)

    return state

def preprocess_image(obs):

    agent_img=obs["agentview_image"]

    agent_img=torch.from_numpy(agent_img).float()

    agent_img=agent_img/255.0

    agent_img=agent_img.permute(2,0,1)

    agent_img=(agent_img-0.5)/0.5

    agent_img = agent_img.unsqueeze(0).to(device)

    wrist_img=obs["robot0_eye_in_hand_image"]
    
    wrist_img=torch.from_numpy(wrist_img).float()

    wrist_img=wrist_img/255.0

    wrist_img=wrist_img.permute(2,0,1)

    wrist_img=(wrist_img-0.5)/0.5

    wrist_img = wrist_img.unsqueeze(0).to(device)

    return agent_img,wrist_img


# =========================================================
# Closed-loop rollout
# =========================================================
init_states = suite.get_task_init_states(task_id)
successcnt=0;
for epoch in range(10):
    frames = []

    success = False
    obs = env.reset()       # 每个 episode 都要重置环境
    
    obs = env.set_init_state(
        init_states[epoch]
    )
    dummy_action = np.array(
        [0, 0, 0, 0, 0, 0, -1],
        dtype=np.float32,
    )
    for _ in range(WAIT_STEPS):
        obs, reward, done, info = env.step(dummy_action)


    for step in range(MAX_STEPS):

        # ---------------------------------
        # current observation
        # ---------------------------------

        state = build_state(obs)
        agent_image,wrist_image=preprocess_image(obs)
        # same normalization as training
        state_norm = (
            state - state_mean
        ) / state_std

        state_tensor = torch.tensor(
            state_norm,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # ---------------------------------
        # BC policy predicts action
        # ---------------------------------
        zero_wrist = torch.zeros_like(wrist_image)
        with torch.no_grad():
            action = model(
                agent_image,
                wrist_image,
                state_tensor
            )[0].cpu().numpy()

        # ensure legal action range
        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # expert gripper label was binary
        if BINARIZE_GRIPPER:
            action[-1] = (
                1.0 if action[-1] >= 0
                else -1.0
            )

        # ---------------------------------
        # execute action
        # ---------------------------------

        obs, reward, done, info = env.step(
            action
        )
        env.env.render()


        # ---------------------------------
        # Debug print
        # ---------------------------------

        if step % 20 == 0:
            print(
                f"step={step:03d}",
                "| action=",
                np.round(action, 3),
                "| ee_pos=",
                np.round(
                    obs["robot0_eef_pos"],
                    3,
                ),
            )

        # LIBERO uses done for task success
        if done:
            successcnt+=1
            success = True
            print(
                f"\nSUCCESS at step {step}"
            )
            break
    
env.close()


# =========================================================
# Save video
# =========================================================
print(successcnt)

# print("\n==========================")
# print("Task:", task.language)
# print("Success:", success)
# print("Steps:", len(frames))
# print("==========================")