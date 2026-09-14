import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "glx")

import imageio.v2 as imageio
import numpy as np
import torch
import robosuite.utils.transform_utils as T

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from my_bc.ACT.ACT_model import ACT
from my_bc.BC.bc_model import BCPolicy
from my_bc.RNN.RNN_model import BCRNNPolicy
from my_bc.TF.TF_model import DualCameraBCTransformer
from my_bc.VISION.vision_model import VisionBCPolicy


DEFAULT_DATASET_PATH = (
    "../LIBERO/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)

CHECKPOINTS = {
    "state": "bc_policy.pt",
    "vision": "vision_bc_policy.pt",
    "rnn": "rnn_bc_policy.pt",
    "transformer": "rnn_TF_policy.pt",
    "act": "act_policy.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and record all single-task BC policies."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(CHECKPOINTS),
        default=list(CHECKPOINTS),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--record-resolution",
        type=int,
        default=512,
        help=(
            "Per-camera recording resolution. Policy input remains 128x128."
        ),
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/evaluation",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
    )
    return parser.parse_args()


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


def image_to_tensor(image, device):
    tensor = torch.from_numpy(
        image.copy()
    ).float()
    tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)


def make_video_frame(obs, env, resolution):
    if resolution > 0:
        # Render a separate high-resolution frame for the video. The policy
        # still receives the original 128x128 observations used in training.
        agent = env.env.sim.render(
            height=resolution,
            width=resolution,
            camera_name="agentview",
        )[::-1]
        wrist = env.env.sim.render(
            height=resolution,
            width=resolution,
            camera_name="robot0_eye_in_hand",
        )[::-1]
    else:
        agent = obs["agentview_image"][::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1]
    return np.concatenate([agent, wrist], axis=1)


def save_video(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def checkpoint_metric(checkpoint):
    if "best_prior_l1" in checkpoint:
        return {
            "name": "best_prior_l1",
            "value": float(checkpoint["best_prior_l1"]),
        }
    if "val_loss" in checkpoint:
        return {
            "name": "best_val_mse",
            "value": float(checkpoint["val_loss"]),
        }
    return None


def load_policy(method, device):
    checkpoint_path = CHECKPOINTS[method]
    if not os.path.isfile(checkpoint_path):
        return None, {
            "status": "skipped",
            "reason": f"missing checkpoint: {checkpoint_path}",
        }

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if method == "state":
        model = BCPolicy()
    elif method == "vision":
        model = VisionBCPolicy()
    elif method == "rnn":
        model = BCRNNPolicy()
    elif method == "transformer":
        model = DualCameraBCTransformer()
    elif method == "act":
        model = ACT(**checkpoint["model_config"])
    else:
        raise ValueError(f"unknown method: {method}")

    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if method == "act":
        stats = checkpoint["stats"]
        state_mean = stats["state_mean"]
        state_std = stats["state_std"]
        action_mean = stats["action_mean"]
        action_std = stats["action_std"]
    else:
        state_mean = checkpoint["state_mean"]
        state_std = checkpoint["state_std"]
        action_mean = None
        action_std = None

    policy = {
        "model": model,
        "state_mean": np.asarray(
            state_mean, dtype=np.float32
        ),
        "state_std": np.asarray(
            state_std, dtype=np.float32
        ),
        "action_mean": (
            None
            if action_mean is None
            else np.asarray(action_mean, dtype=np.float32)
        ),
        "action_std": (
            None
            if action_std is None
            else np.asarray(action_std, dtype=np.float32)
        ),
        "parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "checkpoint": checkpoint_path,
        "checkpoint_metric": checkpoint_metric(checkpoint),
    }
    return policy, None


def prepare_inputs(obs, policy, device):
    raw_state = build_state(obs)
    normalized_state = (
        raw_state - policy["state_mean"]
    ) / policy["state_std"]
    state = torch.from_numpy(
        normalized_state.astype(np.float32)
    ).unsqueeze(0).to(device)

    agent = image_to_tensor(
        obs["agentview_image"],
        device,
    )
    wrist = image_to_tensor(
        obs["robot0_eye_in_hand_image"],
        device,
    )
    return agent, wrist, state


def create_environment(task):
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = ControlEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    env.seed(0)
    return env


def predict_action(
    method,
    policy,
    inputs,
    method_state,
    step,
    temporal_ensemble_coeff,
):
    agent, wrist, state = inputs
    model = policy["model"]

    with torch.no_grad():
        if method == "state":
            action = model(state)[0].cpu().numpy()

        elif method == "vision":
            action = model(
                agent,
                wrist,
                state,
            )[0].cpu().numpy()

        elif method == "rnn":
            if step % 10 == 0:
                method_state["hidden"] = None
            action_sequence, hidden = model(
                agent.unsqueeze(1),
                wrist.unsqueeze(1),
                state.unsqueeze(1),
                hidden=method_state["hidden"],
            )
            method_state["hidden"] = hidden
            action = action_sequence[0, 0].cpu().numpy()

        elif method == "transformer":
            method_state["agent_history"].append(agent)
            method_state["wrist_history"].append(wrist)
            method_state["state_history"].append(state)

            for key in (
                "agent_history",
                "wrist_history",
                "state_history",
            ):
                if len(method_state[key]) > 10:
                    method_state[key].pop(0)

            action_sequence = model(
                torch.stack(
                    method_state["agent_history"],
                    dim=1,
                ),
                torch.stack(
                    method_state["wrist_history"],
                    dim=1,
                ),
                torch.stack(
                    method_state["state_history"],
                    dim=1,
                ),
            )
            action = action_sequence[0, -1].cpu().numpy()

        elif method == "act":
            normalized_chunk, _, _ = model(
                agent,
                wrist,
                state,
                action_chunk=None,
                is_pad=None,
                sample_latent=False,
            )
            normalized_chunk = (
                normalized_chunk[0].cpu().numpy()
            )
            action_chunk = (
                normalized_chunk * policy["action_std"]
                + policy["action_mean"]
            )

            all_actions = method_state["all_time_actions"]
            all_valid = method_state["all_time_valid"]
            chunk_size = action_chunk.shape[0]
            all_actions[
                step,
                step:step + chunk_size,
            ] = action_chunk
            all_valid[
                step,
                step:step + chunk_size,
            ] = True

            candidates = all_actions[
                all_valid[:, step],
                step,
            ]
            weights = np.exp(
                -temporal_ensemble_coeff
                * np.arange(
                    len(candidates),
                    dtype=np.float32,
                )
            )
            weights /= weights.sum()
            action = np.sum(
                candidates * weights[:, None],
                axis=0,
            )
        else:
            raise ValueError(f"unknown method: {method}")

    action = np.clip(
        action,
        -1.0,
        1.0,
    ).astype(np.float32)
    action[-1] = 1.0 if action[-1] >= 0.0 else -1.0
    return action


def initial_method_state(method, max_steps, policy):
    if method == "rnn":
        return {"hidden": None}
    if method == "transformer":
        return {
            "agent_history": [],
            "wrist_history": [],
            "state_history": [],
        }
    if method == "act":
        chunk_size = policy["model"].chunk_size
        action_dim = policy["model"].action_dim
        return {
            "all_time_actions": np.zeros(
                (
                    max_steps,
                    max_steps + chunk_size,
                    action_dim,
                ),
                dtype=np.float32,
            ),
            "all_time_valid": np.zeros(
                (
                    max_steps,
                    max_steps + chunk_size,
                ),
                dtype=bool,
            ),
        }
    return {}


def evaluate_method(
    method,
    policy,
    task,
    init_states,
    args,
    device,
):
    env = create_environment(task)
    episode_results = []
    start_time = time.time()

    try:
        for episode in range(min(args.episodes, len(init_states))):
            obs = env.reset()
            obs = env.set_init_state(
                init_states[episode]
            )

            dummy_action = np.array(
                [0, 0, 0, 0, 0, 0, -1],
                dtype=np.float32,
            )
            for _ in range(args.wait_steps):
                obs, _, _, _ = env.step(dummy_action)

            method_state = initial_method_state(
                method,
                args.max_steps,
                policy,
            )
            frames = (
                [
                    make_video_frame(
                        obs,
                        env,
                        args.record_resolution,
                    )
                ]
                if args.save_videos
                else None
            )
            success = False
            final_step = args.max_steps

            for step in range(args.max_steps):
                inputs = prepare_inputs(
                    obs,
                    policy,
                    device,
                )
                action = predict_action(
                    method,
                    policy,
                    inputs,
                    method_state,
                    step,
                    args.temporal_ensemble_coeff,
                )
                obs, _, done, _ = env.step(action)

                if frames is not None:
                    frames.append(
                        make_video_frame(
                            obs,
                            env,
                            args.record_resolution,
                        )
                    )

                if done:
                    success = True
                    final_step = step + 1
                    break

            if frames is not None:
                status = "success" if success else "failure"
                video_path = os.path.join(
                    args.output_dir,
                    "videos",
                    method,
                    f"episode_{episode:02d}_{status}.mp4",
                )
                save_video(frames, video_path, args.fps)

            episode_results.append(
                {
                    "episode": episode,
                    "success": success,
                    "steps": final_step,
                }
            )
            print(
                f"{method:11s} episode={episode:02d} "
                f"success={success} steps={final_step}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(
        result["success"]
        for result in episode_results
    )
    successful_steps = [
        result["steps"]
        for result in episode_results
        if result["success"]
    ]

    return {
        "status": "complete",
        "checkpoint": policy["checkpoint"],
        "checkpoint_metric": policy["checkpoint_metric"],
        "parameter_count": policy["parameter_count"],
        "episodes": len(episode_results),
        "successes": successes,
        "success_rate": successes / len(episode_results),
        "mean_steps_on_success": (
            float(np.mean(successful_steps))
            if successful_steps
            else None
        ),
        "elapsed_seconds": time.time() - start_time,
        "episode_results": episode_results,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.isfile(dataset_path):
        raise RuntimeError(
            f"找不到 dataset: {dataset_path}"
        )

    task_name = os.path.basename(
        dataset_path
    ).replace("_demo.hdf5", "")

    suite = benchmark.get_benchmark_dict()[
        "libero_spatial"
    ]()
    task_id = None
    for index in range(suite.n_tasks):
        if suite.get_task(index).name == task_name:
            task_id = index
            break

    if task_id is None:
        raise RuntimeError(
            f"找不到对应 LIBERO task: {task_name}"
        )

    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    summary = {
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "dataset_path": dataset_path,
        "device": str(device),
        "evaluation": {
            "requested_episodes": args.episodes,
            "max_steps": args.max_steps,
            "wait_steps": args.wait_steps,
            "init_state_indices": list(
                range(min(args.episodes, len(init_states)))
            ),
            "temporal_ensemble_coeff": (
                args.temporal_ensemble_coeff
            ),
            "record_resolution_per_camera": (
                args.record_resolution
            ),
        },
        "methods": {},
    }

    # Allow one method to be recorded at a time without discarding results
    # already completed by an earlier invocation.
    summary_path = os.path.join(
        args.output_dir,
        "results.json",
    )
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as file:
            previous_summary = json.load(file)
        if previous_summary.get("task_name") == task.name:
            summary["methods"].update(
                previous_summary.get("methods", {})
            )

    print("task:", task.name)
    print("device:", device)

    for method in args.methods:
        policy, skip_result = load_policy(
            method,
            device,
        )
        if skip_result is not None:
            summary["methods"][method] = skip_result
            print(
                f"{method:11s} skipped: "
                f"{skip_result['reason']}",
                flush=True,
            )
            continue

        summary["methods"][method] = evaluate_method(
            method,
            policy,
            task,
            init_states,
            args,
            device,
        )

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("saved:", summary_path)


if __name__ == "__main__":
    main()
