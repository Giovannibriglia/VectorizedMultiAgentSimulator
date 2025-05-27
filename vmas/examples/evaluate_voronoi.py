# evaluate_voronoi.py
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import List, Type

import matplotlib.pyplot as plt

import torch

from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv  # torchrl ≥ 0.4
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from tqdm import tqdm

from vmas import make_env as make_native_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.utils import save_video

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
CHKPT_PATH = "3agents_1goal/policy.pt"  # ← your weights
SCENARIO = "voronoi"
N_AGENTS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_KWARGS = {
    "n_agents": 3,
    "n_gaussians": 1,
    "n_rays": 50,
    "grid_spacing": 0.2,
    "lidar_range": 0.5,
    "centralized": False,
    "shared_rew": False,
}


# ──────────────────────────────────────────────────────────────────────
# Env factories
# ──────────────────────────────────────────────────────────────────────
def make_torchrl_env(num_envs: int, seed: int, **kwargs) -> TransformedEnv:
    """TorchRL wrapper – provides observation_spec and action_spec."""
    raw = VmasEnv(
        scenario=SCENARIO,
        num_envs=num_envs,
        device=DEVICE,
        continuous_actions=True,
        categorical_actions=False,
        seed=seed,
        **kwargs,
    )
    new_env = TransformedEnv(
        raw,
        RewardSum(in_keys=[raw.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    check_env_specs(new_env)
    return new_env


def make_native_vmas_env(num_envs: int, seed: int, **kwargs):
    """Native VMAS env (used by heuristic policies)."""
    return make_native_env(
        scenario=SCENARIO,
        num_envs=num_envs,
        device=DEVICE,
        continuous_actions=True,
        wrapper=None,
        seed=seed,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────
# Network builder / loader
# ──────────────────────────────────────────────────────────────────────
def build_policy(env: TransformedEnv):
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]

    policy_backbone = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=2 * act_dim,
        n_agents=N_AGENTS,
        centralized=False,
        share_params=False,
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
        return_log_prob=True,
    )

    return policy


def load_trained_policy(env: TransformedEnv, ckpt_path: str) -> ProbabilisticActor:
    policy = build_policy(env)
    state = torch.load(ckpt_path, map_location=DEVICE)
    sd = (
        state["state_dict"]
        if isinstance(state, dict) and "state_dict" in state
        else state
    )
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    return policy


# ──────────────────────────────────────────────────────────────────────
# Roll-out helpers
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def rollout_learned(env: TransformedEnv, policy, steps: int) -> List[float]:
    td = env.reset()
    trace = []
    for _ in tqdm(range(steps), desc="Learned"):
        td = policy(td)
        acts = [td[env.action_key][:, i] for i in range(N_AGENTS)]
        obs, rews, *_ = env.step(acts)
        td = TensorDict(
            {("agents", "observation"): torch.stack(obs, 1)},
            batch_size=[env.num_envs],
            device=env.device,
        )

        g = torch.stack(rews, 1).mean(1).mean(0)  # global reward
        trace.append(g.cpu().item())
    return trace


def rollout_heuristic(env, heuristic_cls, steps: int) -> List[float]:
    if heuristic_cls.__name__ == "VoronoiPolicy":
        pol = heuristic_cls(env=env, continuous_action=True)
    else:
        pol = heuristic_cls(continuous_action=True)
    obs = torch.stack(env.reset(), 0)
    trace = []
    for _ in tqdm(range(steps), desc="Heuristic"):
        acts = [
            pol.compute_action(obs[i], u_range=env.agents[i].u_range)
            for i in range(N_AGENTS)
        ]
        obs, rews, *_ = env.step(acts)
        obs = torch.stack(obs, 0)
        trace.append(torch.stack(rews, 1).mean(1).mean(0).cpu().item())
    return trace


# ──────────────────────────────────────────────────────────────────────
# Main comparison function
# ──────────────────────────────────────────────────────────────────────
def compare(
    heuristic: Type[BaseHeuristicPolicy],
    n_steps: int = 300,
    n_envs: int = 1,
    seed: int = 2,
    render: bool = False,
    save_vid: bool = False,
):
    env_h = make_native_vmas_env(n_envs, seed, **ENV_KWARGS)
    env_p = make_torchrl_env(n_envs, seed, **ENV_KWARGS)
    print("TorchRL-env action dim:", env_p.action_spec.shape[-1])  # expect 2

    policy = load_trained_policy(env_p, CHKPT_PATH)

    hr = rollout_heuristic(env_h, heuristic, n_steps)
    pr = rollout_learned(env_p, policy, n_steps)

    plt.figure(dpi=500)
    plt.plot(hr, label=heuristic.__name__, alpha=0.7)
    plt.plot(pr, label="Learned", alpha=0.9)
    plt.xlabel("step")
    plt.ylabel("mean global reward")
    plt.title(f"{SCENARIO} – {N_AGENTS} agents")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if render:
        frames = env_p.render(mode="gif", n_steps=n_steps)
        if save_vid:
            save_video(f"{SCENARIO}_learned", frames, 1 / env_p.scenario.world.dt)

    env_h.close()
    env_p.close()


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from vmas.scenarios.voronoi import VoronoiPolicy

    compare(VoronoiPolicy)
