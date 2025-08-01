#!/usr/bin/env python
"""
VMAS Voronoi PPO trainer (TorchRL ≥ 0.5)
========================================
 * Creates an *experiment folder* named  `{EXP_NAME}_YYYYMMDD_HHMMSS`.
 * Inside it you’ll find three sub‑dirs:
     ├── policies/   – `policy_iter_*.pt`, `policy_final.pt`
     ├── videos/     – `voronoi_iter_*.mp4`, `voronoi_final.mp4`
     └── scalars/    – CSV files with per‑agent reward statistics
 * No figures are generated – everything is stored as CSV/MP4.
"""

from __future__ import annotations

# import csv
from datetime import datetime
from pathlib import Path
from typing import List

import cv2  # required for mp4 encoding
import torch
import imageio

# TorchRL / VMAS imports
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# from torchrl.collectors import SyncDataCollector
# from torchrl.data.replay_buffers import ReplayBuffer
# from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
# from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# from torchrl.objectives import ClipPPOLoss, ValueEstimators
# from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters
# ────────────────────────────────────────────────────────────────────────────

EXP_NAME = "voronoi_ppo"

# devices
IS_FORK = multiprocessing.get_start_method() == "fork"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and not IS_FORK else "cpu")
VMAS_DEVICE = DEVICE

# sampling / training
FRAMES_PER_BATCH = 6_000
N_ITERS = 200
NUM_EPOCHS = 30
MINIBATCH_SIZE = 400
LR = 3e-4
MAX_GRAD_NORM = 1.0
CLIP_EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.9
ENTROPY_EPS = 1e-4
N_CHECKPOINTS = 20  # number of videos / checkpoints you want
LOG_EVERY = max(1, N_ITERS // N_CHECKPOINTS)

# environment
MAX_STEPS = 500
SCENARIO_NAME = "voronoi"
N_AGENTS = 1
N_GAUSSIANS = 3
N_OBSTACLES = 4
LIDAR_RANGE = 0.6
DYNAMIC_PDF = True
N_RAYS = 360

# ────────────────────────────────────────────────────────────────────────────
# Experiment folder layout
# ────────────────────────────────────────────────────────────────────────────

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_DIR = Path.cwd()
POLICY_DIR = ROOT_DIR / "saved_policies"
VIDEO_DIR = ROOT_DIR / "eval_videos"
# SCALAR_DIR = ROOT_DIR / "scalars"
# for d in (POLICY_DIR, VIDEO_DIR, SCALAR_DIR):
#     d.mkdir(parents=True, exist_ok=True)

print("Experiment root:", ROOT_DIR)

# ────────────────────────────────────────────────────────────────────────────
# Build env & networks
# ────────────────────────────────────────────────────────────────────────────

set_composite_lp_aggregate(False)

NUM_VMAS_ENVS = FRAMES_PER_BATCH // MAX_STEPS
print("Num VMAS envs: ", NUM_VMAS_ENVS)
def getPolicy(policy_name: str = "turtle_policy.pt", n_rays=360) -> ProbabilisticActor:
    raw_env = VmasEnv(
        scenario=SCENARIO_NAME,
        num_envs=NUM_VMAS_ENVS,
        continuous_actions=True,
        max_steps=MAX_STEPS,
        device=VMAS_DEVICE,
        n_agents=N_AGENTS,
        n_obstacles=N_OBSTACLES,
        n_gaussians=N_GAUSSIANS,
        lidar_range=LIDAR_RANGE,
        dynamic=DYNAMIC_PDF,
        n_rays=n_rays,
    )

    env = TransformedEnv(
        raw_env,
        RewardSum(in_keys=[raw_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    check_env_specs(env)

    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]
    print("Action dim: ", action_dim)

    policy_backbone = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=2 * action_dim,
        n_agents=N_AGENTS,
        centralized=False,
        share_params=True,
        device=DEVICE,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    policy_module = TensorDictModule(
        torch.nn.Sequential(policy_backbone, NormalParamExtractor()),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    print("Env action keys: ", env.action_key)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=False,
    )

    policy.load_state_dict(
        torch.load(POLICY_DIR / policy_name, weights_only=True)
    )
    return policy
