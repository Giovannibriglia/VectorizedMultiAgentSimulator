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

Before running, set the checkpoint filename/path loaded from `saved_policies/` in `vmas/examples/test.py` (current default is `policy_final.pt`).

The evaluation video is saved to:

- `eval_videos/test.mp4`

## Observation and reward (Voronoi scenario)

Each agent observation is built by concatenating:

- agent position `(x, y)`
- agent velocity `(vx, vy)`
- LIDAR range measurements (`n_rays` values)
- density samples from the Gaussian field

By default (`centralized=False`), density samples are taken on a local square grid centered at the agent (size controlled by `cells_range` and `grid_spacing`). If `centralized=True`, density is sampled over the full map grid.

Reward is computed in a centralized way: for each environment, the scenario computes the Voronoi partition induced by all agent positions, then sums a coverage cost over all agents (distance of points in each Voronoi cell weighted by the field density), plus collision penalties.

Since this reward is summed at team level, the same reward value is returned to all agents.
