<div align="center">

# LIBERO Policy Lab

### A reproducible imitation-learning playground built on LIBERO

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Benchmark-LIBERO-6f42c1)](https://libero-project.github.io/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-f59e0b)](#project-status)

Compare state-based BC, visual BC, recurrent policies, Transformers, and
Action Chunking Transformers under one closed-loop evaluation protocol.

[中文](README.zh-CN.md) · [Results](#benchmark-snapshot) · [Experiment report](reports/experiment_report.md) · [Upstream LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

</div>

> [!IMPORTANT]
> This is an independent research project derived from
> [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). It is not an
> official LIBERO release. The original benchmark, environments, datasets,
> and assets remain credited to the LIBERO authors; the policy baselines,
> unified evaluator, experiment artifacts, and analysis in this repository
> are the additions of this project.

## Why this project?

Offline imitation-learning losses often look encouraging while the same
policy fails after being placed back in the environment. LIBERO Policy Lab
keeps training, rollout, video recording, and result export close together so
that policies are compared by what matters most: reproducible closed-loop
task completion.

The current study contributes:

- six compact policy implementations spanning MLP, CNN, LSTM, Transformer,
  single-task ACT, and task-conditioned multi-task ACT;
- a shared evaluator with fixed initial states, identical horizons, optional
  video export, and machine-readable JSON results;
- checkpoints and a complete single-task case study, including failure
  analysis rather than only the best rollout;
- a documented data-alignment and normalization pipeline for LIBERO HDF5
  demonstrations.

## Benchmark snapshot

The published snapshot evaluates the LIBERO-Spatial task **“pick up the black
bowl on the ramekin and place it on the plate”** on the same 10 initial states,
with a 10-step settling period and a maximum horizon of 220 steps.

![Closed-loop model comparison](reports/figures/model_comparison.png)

| Method | Parameters | Success | Mean steps on success | Takeaway |
|:--|--:|--:|--:|:--|
| State BC | 19,463 | **10/10 (100%)** | 96.6 | Strongest small baseline on this narrow task |
| Vision BC | 406,023 | Not evaluated | — | Checkpoint is not included yet |
| RNN BC | 636,295 | **9/10 (90%)** | 102.2 | History helps, but drift remains |
| Transformer BC | 441,351 | **0/10 (0%)** | — | Low validation error did not transfer to rollout |
| ACT | 4,054,983 | **9/10 (90%)** | **92.7** | Fastest successful trajectories |

These numbers are a 10-episode engineering comparison, not a statistically
complete leaderboard. See the [full report](reports/experiment_report.md) for
per-initial-state results, uncertainty, protocol details, and limitations.
Raw results are available in
[`reports/evaluation/results.json`](reports/evaluation/results.json).

## Policy zoo

| Policy | Observation | Temporal design | Training entry point |
|:--|:--|:--|:--|
| State BC | 15-D robot state | Current step | `my_bc/BC/train_bc.py` |
| Vision BC | Two RGB cameras + state | Current step | `my_bc/VISION/train_vision_bc.py` |
| RNN BC | Two RGB cameras + state | LSTM history | `my_bc/RNN/train_RNN_bc.py` |
| Transformer BC | Two RGB cameras + state | Causal context window | `my_bc/TF/train_TF_bc.py` |
| ACT | Two RGB cameras + state | CVAE action chunks | `my_bc/ACT/train_ACT_bc.py` |
| Multi-task ACT | Images + state + task ID | Task-conditioned action chunks | `my_bc/muti_ACT/train_ACT_bc.py` |

## Installation

The project follows the upstream LIBERO environment and currently targets
Python 3.8 with MuJoCo-compatible rendering.

```bash
git clone https://github.com/dexori/LIBERO.git
cd LIBERO

conda create -n libero-policy-lab python=3.8.13 -y
conda activate libero-policy-lab

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

If your CUDA version differs, install the matching PyTorch build instead of
the pinned CUDA 11.3 command above. Refer to the
[official LIBERO documentation](https://lifelong-robot-learning.github.io/LIBERO/)
for simulator and headless-rendering setup.

## Data

Download a benchmark suite with the upstream helper:

```bash
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial \
  --use-huggingface
```

Datasets are stored under `libero/datasets/` and intentionally excluded from
Git. LIBERO datasets use the CC BY 4.0 license; consult the upstream project
before redistributing them.

## Reproduce the evaluation

The repository includes checkpoints for State BC, RNN BC, Transformer BC,
and ACT. From the repository root, run:

```bash
python my_bc/evaluate_single_task.py \
  --episodes 10 \
  --max-steps 220 \
  --record-resolution 512 \
  --save-videos \
  --output-dir reports/evaluation
```

The evaluator writes a JSON summary and, when requested, one video per
episode. It skips a method cleanly when its checkpoint is absent; this is why
Vision BC is marked as not evaluated in the current snapshot.

For the task-conditioned multi-task experiment:

```bash
python my_bc/muti_ACT/train_ACT_bc.py \
  --dataset-dir libero/datasets/libero_spatial \
  --checkpoint multi_act_policy.pt
```

> [!NOTE]
> `muti_ACT` is the historical directory name currently used by the code. It
> will be migrated to `multi_ACT` in a compatibility-preserving cleanup.

## Repository map

```text
.
├── libero/                    # Upstream benchmark, environments, and assets
├── benchmark_scripts/         # Upstream dataset and task utilities
├── my_bc/
│   ├── BC/                    # State behavioral cloning
│   ├── VISION/                # Dual-camera visual BC
│   ├── RNN/                   # Recurrent visual BC
│   ├── TF/                    # Causal Transformer BC
│   ├── ACT/                   # Single-task Action Chunking Transformer
│   ├── muti_ACT/              # Task-conditioned multi-task ACT (experimental)
│   └── evaluate_single_task.py
└── reports/                   # Results, figures, report, and presentation assets
```

## Project status

This repository is a research prototype. The single-task comparison is
reproducible with the included artifacts, while broader multi-task evaluation
is active work. The most useful next milestones are:

- [ ] complete and publish the missing Vision BC checkpoint;
- [ ] evaluate multiple random seeds and report confidence intervals;
- [ ] finish the task-conditioned multi-task ACT benchmark;
- [ ] replace script-level constants with consistent command-line configs;
- [ ] add smoke tests and a lightweight continuous-integration workflow;
- [ ] move large checkpoints to a versioned release or model registry.

## Contributing

Bug reports, reproduction notes, new baselines, and evaluation improvements
are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or
pull request. Please include the task, seed, checkpoint, and exact evaluation
command when reporting policy performance.

## Citation and attribution

If this repository helps your work, cite the original LIBERO paper:

```bibtex
@article{liu2023libero,
  title   = {LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author  = {Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal = {arXiv preprint arXiv:2306.03310},
  year    = {2023}
}
```

When referring specifically to the implementations or experiment artifacts in
this derivative project, link to this repository and the exact commit used.

## License

The code is distributed under the [MIT License](LICENSE). The existing license
and upstream copyright notice are retained as required. Datasets and some
assets have separate upstream terms; see the
[LIBERO project](https://github.com/Lifelong-Robot-Learning/LIBERO) for details.
