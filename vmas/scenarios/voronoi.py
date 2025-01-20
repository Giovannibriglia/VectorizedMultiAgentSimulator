#  Copyright (c) 2023-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Callable, Dict

import numpy as np
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

from pathlib import Path
vmas_dir = Path(__file__).parent
print("vmas dir: ", vmas_dir)

import sys, os
sys.path.append(os.path.dirname(vmas_dir))
from algorithms.VoronoiCoverage import VoronoiCoverage


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 3)
        self.shared_rew = kwargs.pop("shared_rew", True)

        self.comms_range = kwargs.pop("comms_range", 0.0)
        self.lidar_range = kwargs.pop("lidar_range", 0.2)
        self.agent_radius = kwargs.pop("agent_radius", 0.025)
        self.xdim = kwargs.pop("xdim", 1)
        self.ydim = kwargs.pop("ydim", 1)
        self.grid_spacing = kwargs.pop("grid_spacing", 0.05)

        self.n_gaussians = kwargs.pop("n_gaussians", 3)
        self.cov = kwargs.pop("cov", 0.05)
        self.collisions = kwargs.pop("collisions", True)
        self.spawn_same_pos = kwargs.pop("spawn_same_pos", False)
        self.norm = kwargs.pop("norm", True)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert not (self.spawn_same_pos and self.collisions)
        assert (self.xdim / self.grid_spacing) % 1 == 0 and (
            self.ydim / self.grid_spacing
        ) % 1 == 0
        self.covs = (
            [self.cov] * self.n_gaussians if isinstance(self.cov, float) else self.cov
        )
        assert len(self.covs) == self.n_gaussians

        self.plot_grid = False
        self.visualize_semidims = False
        self.n_x_cells = int((2 * self.xdim) / self.grid_spacing)
        self.n_y_cells = int((2 * self.ydim) / self.grid_spacing)
        self.max_pdf = torch.zeros((batch_dim,), device=device, dtype=torch.float32)
        self.alpha_plot: float = 0.5

        self.agent_xspawn_range = 0 if self.spawn_same_pos else self.xdim
        self.agent_yspawn_range = 0 if self.spawn_same_pos else self.ydim
        self.x_semidim = self.xdim - self.agent_radius
        self.y_semidim = self.ydim - self.agent_radius

        self.pdf = [None] * batch_dim

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                render_action=True,
                collide=self.collisions,
                shape=Sphere(radius=self.agent_radius),
                sensors=(
                    [
                        Lidar(
                            world,
                            angle_start=0.05,
                            angle_end=2 * torch.pi + 0.05,
                            n_rays=12,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )

            world.add_agent(agent)

        self.sampled = torch.zeros(
            (batch_dim, self.n_x_cells, self.n_y_cells),
            device=device,
            dtype=torch.bool,
        )

        self.locs = [
            torch.zeros((batch_dim, world.dim_p), device=device, dtype=torch.float32)
            for _ in range(self.n_gaussians)
        ]
        self.cov_matrices = [
            torch.tensor(
                [[cov, 0], [0, cov]], dtype=torch.float32, device=device
            ).expand(batch_dim, world.dim_p, world.dim_p)
            for cov in self.covs
        ]

        return world

    def reset_world_at(self, env_index: int = None):
        for i in range(len(self.locs)):
            x = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.xdim, self.xdim)
            y = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.ydim, self.ydim)
            new_loc = torch.cat([x, y], dim=-1)
            if env_index is None:
                self.locs[i] = new_loc
            else:
                self.locs[i][env_index] = new_loc

        self.gaussians = [
            MultivariateNormal(
                loc=loc,
                covariance_matrix=cov_matrix,
            )
            for loc, cov_matrix in zip(self.locs, self.cov_matrices)
        ]

        x_grid = torch.linspace(-self.xdim, self.xdim, self.n_x_cells)
        y_grid = torch.linspace(-self.ydim, self.ydim, self.n_y_cells)
        xg, yg = torch.meshgrid(x_grid, y_grid)
        xy_grid = torch.vstack((xg.ravel(), yg.ravel())).T
        # xy_grid = xy_grid.unsqueeze(0).expand(self.world.batch_dim, -1, -1)
        self.pdf = [self.sample_single_env(xy_grid, i) for i in range(self.world.batch_dim)]

        if env_index is None:
            self.max_pdf[:] = 0
            self.sampled[:] = False
        else:
            self.max_pdf[env_index] = 0
            self.sampled[env_index] = False
        self.normalize_pdf(env_index=env_index)

        for agent in self.world.agents:
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.agent_xspawn_range, self.agent_xspawn_range),
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.agent_yspawn_range, self.agent_yspawn_range),
                    ],
                    dim=-1,
                ),
                batch_index=env_index,
            )
            agent.sample = self.sample(agent.state.pos, norm=self.norm)

    def sample(
        self,
        pos,
        update_sampled_flag: bool = False,
        norm: bool = True,
    ):
        out_of_bounds = (
            (pos[:, X] < -self.xdim)
            + (pos[:, X] > self.xdim)
            + (pos[:, Y] < -self.ydim)
            + (pos[:, Y] > self.ydim)
        )
        pos[:, X].clamp_(-self.world.x_semidim, self.world.x_semidim)
        pos[:, Y].clamp_(-self.world.y_semidim, self.world.y_semidim)

        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)
        v = torch.stack(
            [gaussian.log_prob(pos).exp() for gaussian in self.gaussians],
            dim=-1,
        ).sum(-1)
        if norm:
            v = v / self.max_pdf

        sampled = self.sampled[
            torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]
        ]

        v[sampled + out_of_bounds] = 0
        if update_sampled_flag:
            self.sampled[
                torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]
            ] = True

        return v

    def sample_single_env(
        self,
        pos,
        env_index,
        norm: bool = True,
    ):
        pos = pos.view(-1, self.world.dim_p)

        out_of_bounds = (
            (pos[:, X] < -self.xdim)
            + (pos[:, X] > self.xdim)
            + (pos[:, Y] < -self.ydim)
            + (pos[:, Y] > self.ydim)
        )
        pos[:, X].clamp_(-self.x_semidim, self.x_semidim)
        pos[:, Y].clamp_(-self.y_semidim, self.y_semidim)

        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)

        pos = pos.unsqueeze(1).expand(pos.shape[0], self.world.batch_dim, 2)

        v = torch.stack(
            [gaussian.log_prob(pos).exp() for gaussian in self.gaussians],
            dim=-1,
        ).sum(-1)[:, env_index]
        if norm:
            v = v / self.max_pdf[env_index]

        sampled = self.sampled[env_index, index[:, 0], index[:, 1]]

        v[sampled + out_of_bounds] = 0

        return v

    def normalize_pdf(self, env_index: int = None):
        xpoints = torch.arange(
            -self.xdim, self.xdim, self.grid_spacing, device=self.world.device
        )
        ypoints = torch.arange(
            -self.ydim, self.ydim, self.grid_spacing, device=self.world.device
        )
        if env_index is not None:
            ygrid, xgrid = torch.meshgrid(ypoints, xpoints, indexing="ij")
            pos = torch.stack((xgrid, ygrid), dim=-1).reshape(-1, 2)
            sample = self.sample_single_env(pos, env_index, norm=False)
            self.max_pdf[env_index] = sample.max()
        else:
            for x in xpoints:
                for y in ypoints:
                    pos = torch.tensor(
                        [x, y], device=self.world.device, dtype=torch.float32
                    ).repeat(self.world.batch_dim, 1)
                    sample = self.sample(pos, norm=False)
                    self.max_pdf = torch.maximum(self.max_pdf, sample)

    def reward(self, agent: Agent) -> Tensor:
        is_first = self.world.agents.index(agent) == 0
        robots = [a.state.pos for a in self.world.agents]
        robots = torch.stack(robots).to(self.world.device)
        voro = VoronoiCoverage(robots, self.pdf, self.grid_spacing, self.xdim, self.ydim, self.world.device, centralized=True)
        voro.partitioning()
        reward = torch.zeros((self.world.batch_dim, self.n_agents))
        if is_first:
            for i,a in enumerate(self.world.agents):
                reward[:, i] = voro.computeCoverageFunction(self.world.agents.index(a))
            self.sampling_rew = reward.sum(-1)

        return self.sampling_rew if self.shared_rew else voro.computeCoverageFunction(self.world.agents.index(agent))

    def observation(self, agent: Agent) -> Tensor:
        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.sensors[0].measure(),
        ]

        for delta in [
            [self.grid_spacing, 0],
            [-self.grid_spacing, 0],
            [0, self.grid_spacing],
            [0, -self.grid_spacing],
            [-self.grid_spacing, -self.grid_spacing],
            [self.grid_spacing, -self.grid_spacing],
            [-self.grid_spacing, self.grid_spacing],
            [self.grid_spacing, self.grid_spacing],
        ]:
            pos = agent.state.pos + torch.tensor(
                delta,
                device=self.world.device,
                dtype=torch.float32,
            )
            sample = self.sample(
                pos,
                update_sampled_flag=False,
            ).unsqueeze(-1)
            observations.append(sample)

        return torch.cat(
            observations,
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {"agent_sample": agent.sample}

    def density_for_plot(self, env_index):
        def f(x):
            sample = self.sample_single_env(
                torch.tensor(x, dtype=torch.float32, device=self.world.device),
                env_index=env_index,
            )

            return sample

        return f

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        from vmas.simulator.rendering import render_function_util

        # Function
        geoms = [
            render_function_util(
                f=self.density_for_plot(env_index=env_index),
                plot_range=(self.xdim, self.ydim),
                cmap_alpha=self.alpha_plot,
            )
        ]

        # Compute Voronoi regions
        vor = self._compute_voronoi_regions(env_index)
        geoms = self._plot_voronoi_regions(vor, geoms)

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        # Perimeter
        for i in range(4):
            geom = Line(
                length=2
                * ((self.ydim if i % 2 == 0 else self.xdim) - self.agent_radius)
                + self.agent_radius * 2
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.x_semidim + self.agent_radius
                        if i == 0
                        else -self.x_semidim - self.agent_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.y_semidim + self.agent_radius
                        if i == 1
                        else -self.y_semidim - self.agent_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        return geoms

    def _compute_voronoi_regions(self, env_index):
        """
        Computes Voronoi regions based on agent positions for a specific environment.

        Args:
            env_index: Index of the environment.

        Returns:
            Voronoi object from scipy.spatial.
        """
        from scipy.spatial import Voronoi

        agent_positions = [
            agent.state.pos[env_index].cpu().numpy() for agent in self.world.agents
        ]
        points = np.array(agent_positions)

        # Compute Voronoi regions
        vor = Voronoi(points)
        return vor

    @staticmethod
    def _clip_line_to_bounds(p1, p2, x_bounds, y_bounds):
        """
        Clips a line segment to fit within the specified rectangular bounds.

        Args:
            p1, p2: Endpoints of the line segment as [x, y].
            x_bounds: Tuple of (x_min, x_max) for x-coordinates.
            y_bounds: Tuple of (y_min, y_max) for y-coordinates.

        Returns:
            Clipped line segment as a list of two points, or None if outside bounds.
        """
        from shapely.geometry import box, LineString

        bbox = box(x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1])
        line = LineString([p1, p2])
        clipped_line = line.intersection(bbox)

        if clipped_line.is_empty:
            return None
        elif clipped_line.geom_type == "LineString":
            return list(clipped_line.coords)
        else:
            return None

    def _plot_voronoi_regions(self, vor, geoms):
        """
        Plots Voronoi regions with finite and clipped infinite edges.

        Args:
            vor: Voronoi object from scipy.spatial.
            geoms: List of geometric shapes for rendering.

        Returns:
            Updated list of geometries including Voronoi regions.
        """
        from vmas.simulator.rendering import PolyLine

        x_min, x_max = -self.xdim, self.xdim
        y_min, y_max = -self.ydim, self.ydim
        ptp_bound = np.array([x_max - x_min, y_max - y_min])

        center = vor.points.mean(axis=0)
        finite_segments = []
        infinite_segments = []

        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]
                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[i] + direction * ptp_bound.max()
                infinite_segments.append([vor.vertices[i], far_point])

        # Render finite segments
        for segment in finite_segments:
            line = PolyLine(segment.tolist(), close=False)
            line.set_color(0.2, 0.8, 0.2)
            geoms.append(line)

        # Render clipped infinite segments
        for segment in infinite_segments:
            clipped_segment = self._clip_line_to_bounds(
                segment[0], segment[1], (x_min, x_max), (y_min, y_max)
            )
            if clipped_segment:
                line = PolyLine(clipped_segment, close=False)
                line.set_color(0.8, 0.2, 0.2)
                geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
