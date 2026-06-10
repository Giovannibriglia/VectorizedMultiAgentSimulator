# VectorizedMultiAgentSimulator (Voronoi Fork)

This fork focuses on custom Voronoi experiments built on top of VMAS.

For installation, dependencies, and general VMAS usage, refer to the upstream repository:

- https://github.com/proroklab/VectorizedMultiAgentSimulator

## What is customized in this fork

- Custom scenario: `vmas/scenarios/voronoi.py`
- Training script: `vmas/examples/train.py`
- Evaluation script: `vmas/examples/test.py`

## Train on the Voronoi scenario

From the repository root:

```bash
python vmas/examples/train.py
```

The script builds a TorchRL PPO training pipeline with `SCENARIO_NAME = "voronoi"` and writes outputs to a timestamped folder:

- `voronoi_ppo_YYYYMMDD_HHMMSS/policies/` (policy checkpoints)
- `voronoi_ppo_YYYYMMDD_HHMMSS/videos/` (evaluation videos)
- `voronoi_ppo_YYYYMMDD_HHMMSS/scalars/` (CSV metrics)

Key environment arguments are defined in the script and passed to `VmasEnv`, e.g. `n_agents`, `n_gaussians`, `lidar_range`, `n_rays`, `n_obstacles`, and `if_walls`.

## Test/evaluate a saved policy

From the repository root:

```bash
python vmas/examples/test.py
```

Before running, set the checkpoint filename/path loaded from `saved_policies/` in `vmas/examples/test.py` (current default is `policy_1006.pt`).

The evaluation video is saved to:

- `eval_videos/test.mp4`
