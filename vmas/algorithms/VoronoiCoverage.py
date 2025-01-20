import torch
from scipy.spatial import Voronoi
from matplotlib.path import Path




class VoronoiCoverage:
    def __init__(self, agents, pdf, grid_spacing, xdim=10, ydim=10, device="cpu", centralized=True):
        self.agents = agents                        # [n_agents, n_envs, dim]
        self.pdf = pdf                              # [nxcells, nycells]
        self.centralized = centralized
        self.xmin = -xdim / 2
        self.xmax = xdim / 2
        self.ymin = -ydim / 2
        self.ymax = ydim / 2
        self.grid_spacing = grid_spacing
        self.nxcells = int(2 * xdim / self.grid_spacing)
        self.nycells = int(2 * ydim / self.grid_spacing)
        self.robots_num = self.agents.shape[0]
        self.worlds_num = self.agents.shape[1]
        self.device = device

        self.x_grid = torch.linspace(self.xmin, self.xmax, self.nxcells)
        self.y_grid = torch.linspace(self.ymin, self.ymax, self.nycells)
        xg, yg = torch.meshgrid(self.x_grid, self.y_grid)
        self.xy_grid = torch.vstack((xg.ravel(), yg.ravel())).T
        regions_single_env = [None for i in range(self.robots_num)]
        self.regions = [regions_single_env] * self.worlds_num
        self.vertices = [regions_single_env] * self.worlds_num
        self.voronois = [None] * self.worlds_num

    def partitioning(self):
        # robot_positions = np.array([self._position] + [neighbor.position for neighbor in self._neighbors if np.linalg.norm(neighbor.position - self._position) <= self._range and not np.all(neighbor.position == self._position)])

        points_left = torch.clone(self.agents)
        points_left[:, 0] = 2 * self.xmin - points_left[:, 0]
        points_right = torch.clone(self.agents)
        points_right[:, 0] = 2 * self.xmax - points_right[:, 0]
        points_down = torch.clone(self.agents)
        points_down[:, 1] = 2 * self.ymin - points_down[:, 1]
        points_up = torch.clone(self.agents)
        points_up[:, 1] = 2 * self.ymax - points_up[:, 1]
        points = torch.vstack((self.agents, points_left, points_right, points_down, points_up))

        # Voronoi diagram
        for j in range(self.worlds_num):
            vor = Voronoi(points[:, j, :].cpu().detach().numpy())
            # vor.filtered_points = self.agents[:, j, :].cpu().detach().numpy()
            # regions = [vor.point_region[i] for i in range(self.robots_num)]
            self.regions[j] = [vor.regions[i] for i in vor.point_region[:self.robots_num]]
            self.vertices[j] = [[vor.vertices[v] for v in self.regions[j]]]             #(n_env, n_robots, n_verts)
            # for v in range(self.regions):
            #     self.vertices[j][i].append(vor.vertices[i])
            self.voronois[j] = vor


    def getPointsInRegion(self, region):
        p = Path(region)
        bool_val = p.contains_points(self.xy_grid.cpu().detach().numpy())
        return bool_val

    def computeCoverageFunction(self, agent_id):
        reward = torch.zeros((self.worlds_num))
        for i in range(self.worlds_num):
            vor = self.voronois[i]
            region = vor.point_region[agent_id]
            verts = [vor.vertices[v] for v in vor.regions[region]]
            # regions = [vor.regions[j] for j in vor.point_region[:self.robots_num]]
            # verts = [[vor.vertices[v] for v in vor.regions[region]] for region in regions]
            bool_val = self.getPointsInRegion(verts)
            mask = bool_val.reshape(self.nxcells, self.nycells)
            weights = self.pdf[i][bool_val]
            reward[i] = torch.sum(weights * torch.linalg.norm(self.xy_grid[bool_val] - self.agents[agent_id, i], axis=1)**2) * self.grid_spacing**2

        return reward


    def computeCoverageFunctionSingleEnv(self, agent_id, env_id):
        region = self.regions[env_id][agent_id]
        bool_val = self.getPointsInRegion(region)
        weights = self.pdf[bool_val.reshape(self.nxcells, self.nycells)]
        reward = torch.sum(weights * torch.linalg.norm(self.xy_grid[bool_val] - self.agents[agent_id], axis=1)**2) * self.grid_spacing**2
        return reward


    def computeCentroid(self, agent_id):
        region = self.regions[agent_id]
        bool_val = self.getPointsInRegion(region)
        weights = self.pdf[bool_val.reshape(self.nxcells, self.nycells)]
        dA = self.grid_spacing**2
        A = torch.sum(weights) * dA
        Cx = torch.sum(weights * self.xy_grid[:, 0][bool_val]) * dA
        Cy = torch.sum(weights * self.xy_grid[:, 1][bool_val]) * dA
        return torch.tensor(([Cx, Cy]) / A)






