# Contributing

Thanks for helping improve LIBERO Policy Lab. Reproduction reports, bug fixes,
documentation, new policy baselines, and stricter evaluation tools are all
useful contributions.

## Before opening an issue

Please search existing issues and include enough context for someone else to
reproduce the problem:

- operating system, Python, CUDA, PyTorch, MuJoCo, and GPU versions;
- LIBERO suite, task name or ID, and dataset source;
- the exact command and relevant configuration;
- the complete error message or the observed-versus-expected behavior;
- checkpoint provenance when the issue occurs during rollout.

Do not upload private datasets, credentials, or machine-specific paths.

## Development setup

Follow the installation instructions in [README.md](README.md), then install
the repository in editable mode:

```bash
pip install -e .
```

Run scripts from the repository root. Keep datasets under `libero/datasets/`;
that directory is ignored by Git.

## Pull requests

Keep each pull request focused. In its description, explain the motivation,
the behavior changed, and how you verified it. For policy or training changes,
also report:

- train/validation split and random seed;
- number of demonstrations and training epochs;
- checkpoint selection rule;
- closed-loop initial states, episode count, and horizon;
- both successes and failures, without selecting only favorable rollouts.

Avoid committing generated caches, local environments, raw datasets, or large
new checkpoints. Prefer a versioned GitHub Release or an external model host
for model artifacts and link it from the pull request.

## Evaluation standard

Offline losses are useful diagnostics but are not a replacement for
closed-loop evaluation. New result claims should use the shared evaluator when
possible and include the generated JSON summary. If a method needs a different
protocol, describe the difference explicitly rather than presenting the
numbers as directly comparable.

## Upstream scope

Changes to the original LIBERO benchmark may be better suited to the
[upstream repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Please keep upstream attribution intact and clearly identify behavior that is
specific to this derivative project.
