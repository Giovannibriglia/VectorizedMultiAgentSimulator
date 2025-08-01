import numpy as np
import torch
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

path = Path().home() / "VectorizedMultiAgentSimulator/vmas/examples"
sys.path.append(os.path.abspath(path))
from get_policy import getPolicy


# params
grid_spacing = 0.05
env_size = 10.0
cells_range = 3
robot_range = 5.0
robot_range_norm = robot_range / env_size * 2
dt = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_sim = 10.0


def gauss_pdf(x, y, mean, covariance):
  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)
  return prob

def gmm_pdf(x, y, means, covariances, weights):
  prob = 0.0
  for i in range(len(means)):
    prob += weights[i] * gauss_pdf(x, y, means[i], covariances[i])
  return prob

def quaternion_to_euler(x, y, z, w):
    """
    Convert a quaternion into roll, pitch, yaw (in radians)
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cos_y_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cos_y_cosp)

    return roll, pitch, yaw

# def f(self, state, u_command, ang_vel_command):
#     theta = state[2]
#     dx = u_command * torch.cos(theta)
#     dy = u_command * torch.sin(theta)
#     dtheta = ang_vel_command
#     return torch.stack((dx, dy, dtheta), dim=-1)  # [batch_size,3]

def dynamics(state, acc):
    vel = state[2:]
    return torch.hstack((vel, acc))

def runge_kutta(state, acc):
    f = dynamics
    k1 = f(state, acc)
    k2 = f(state + 0.5*dt * k1, acc)
    k3 = f(state + 0.5*dt * k2, acc)
    k4 = f(state + dt * k3, acc)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

def process_action(self, action):
    u_command = action[0,0]  # Forward velocity
    ang_vel_command = action[0,1]  # Angular velocity

    v_cur_x = self.vel[0]  # Current velocity in x-direction
    
    v_cur_angular = self.vel[1]  # Current angular velocity

    delta_state = self.runge_kutta(u_command, ang_vel_command)

    # Calculate the accelerations required to achieve the change in state
    acceleration_x = (delta_state[0] - v_cur_x * self.dt) / self.dt**2
    # acceleration_y = (delta_state[1] - v_cur_y * self.dt) / self.dt**2
    acceleration_angular = (delta_state[2] - v_cur_angular * self.dt) / self.dt**2
    return torch.tensor([acceleration_x, acceleration_angular])

def getVelocityWorld(pos, vel, a_world):
    th = pos[2]
    vl_now = torch.tensor([vel[0] * torch.cos(th),
                            vel[0] * torch.cos(th)]).to(device)
    v_next = vl_now + a_world * dt

    v_lin_new = torch.linalg.norm(v_next)
    th_new = torch.arctan2(v_next[1], v_next[0])
    w_new = (th_new - th) / dt
    # w_new = self.vel[1] + w_des * self.dt
    return torch.tensor([v_lin_new, w_new]).to(device)


def getVelocityBody(pos, vel, a_body):
    th = pos[2]
    v_new = vel[0] + a_body[0] * dt
    w_new = vel[1] + 0.6 * a_body[1] * dt
    return torch.tensor([v_new, w_new])

def cpu(tensor):
    return tensor.cpu().detach().numpy()

def pd_controller(position, velocity, goal, kp=2.0, kd=1.0):
    # Simple PD controller to compute acceleration command
    pos_error = goal - position
    vel_error = -velocity
    acceleration = kp * pos_error + kd * vel_error
    return acceleration


def main():
    # Load policy
    policy = getPolicy("decentralized_policy.pt", 200).to(device)

    pos = torch.zeros(2, dtype=torch.float)
    th = 0.0
    norm_pos = torch.zeros(2, dtype=torch.float)
    vel = torch.zeros(2, dtype=torch.float)
    lidar_obs = 0.6 * torch.ones(200, dtype=torch.float)

        # GMM parameters
    COMPONENTS_NUM = 1
    means = -0.5*env_size + env_size * np.random.rand(COMPONENTS_NUM, 2)
    means[0, 0] = 3.5
    means[0, 1] = -2.0
    means_norm = means / env_size * 2
    covariances = []
    for i in range(COMPONENTS_NUM):
        # cov = 2*np.random.rand(2, 2)
        # cov = cov @ cov.T  # Ensure positive semi-definite covariance
        cov = 0.5*np.eye(2) + 0.1 * np.random.rand(2, 2)
        covariances.append(cov)
    covariances = np.array(covariances)
    weights = np.random.dirichlet(np.ones(COMPONENTS_NUM))  # Dirichlet distribution makes weights sum to 1
    # print("weights: ", weights)

    xg = np.linspace(-0.5*env_size, 0.5*env_size, 100)
    yg = np.linspace(-0.5*env_size, 0.5*env_size, 100)
    X, Y = np.meshgrid(xg, yg)
    pdf = gmm_pdf(X, Y, means, covariances, weights)
    xy_grid = np.vstack((xg.ravel(), yg.ravel()))
    xy_grid_norm = xy_grid / (0.5 * env_size)

    pdf_global = np.zeros((200, 200))
    pdf_global[50:-50, 50:-50] = pdf.reshape(100, 100) / np.max(pdf)

    pos_hist = [cpu(pos)]

    t = 0.0
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    vmax = 2.0
    
    while t < t_sim:
        # sample pdf
        pdf_local = []
        xs = np.linspace(cpu(norm_pos)[0]-robot_range_norm, cpu(norm_pos)[0]+robot_range_norm, 2*cells_range+1)
        ys = np.linspace(cpu(norm_pos)[1]-robot_range_norm, cpu(norm_pos)[1]+robot_range_norm, 2*cells_range+1)
        for yi in xs:
            for xi in ys:
                pdf_local.append(gmm_pdf(xi, yi, means_norm, covariances, weights))
        # xi, yi = int(pos[0]/100), int(pos[1]/100)
        # pdf_local = pdf_global[50+xi-cells_range: 50+xi+cells_range+1, 50+yi-cells_range:50+yi+cells_range+1]
        pdf_local = torch.tensor(pdf_local).squeeze(1)
        pdf_local = pdf_local.reshape((2*cells_range+1, 2*cells_range+1)) / torch.max(pdf_local)
        # pdf_local = torch.zeros((2*self.cells_range+1, 2*self.cells_range+1))
        # print("pdf local: ", pdf_local)
        pdf_rot = torch.zeros_like(pdf_local)
        for row in range(pdf_local.shape[0]):
            pdf_rot[:, row] = pdf_local[row, :]
        pdf_rot = pdf_rot.reshape(1, 49).squeeze(0)

        # v_lin = torch.tensor([vel[0] * np.cos(th), vel[0] * np.sin(th)])
        obs = torch.cat((norm_pos, vel, lidar_obs, pdf_rot)).to(device)
        obs = obs.to(torch.float)
        action = policy(obs.unsqueeze(0))[-1]
        print("Action: ", action) 

        # v, w = getVelocityWorld(action.squeeze(0))
        # print("v, w: ", v, w)

        # v = torch.clip(v, -1, 1)
        # w = torch.clip(w, -1, 1)
        
        # update state
        # pos += vel * dt
        # th_next = th + w * dt
        # vel += action.squeeze(0).cpu() * dt
        # vel = torch.clip(vel, -vmax, vmax)
        x_next = runge_kutta(torch.hstack((pos, vel)), action.squeeze(0).cpu())
        pos, vel = x_next[:2], x_next[2:]
        # vel = torch.clip(vel, -vmax, vmax)
        print("Vel : ", vel)
        norm_pos = pos / env_size * 2
        pos_hist.append(cpu(pos))

        t += dt


        # plot
        ax.cla()
        ax.pcolormesh(X, Y, pdf.reshape(X.shape), cmap='Greys', alpha=0.75)
        ax.scatter(cpu(pos)[0], cpu(pos)[1], c="tab:blue", label="Robot")
        traj = np.array(pos_hist)
        ax.plot(traj[:, 0], traj[:, 1], c="tab:blue")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')   # keeps squares square
        ax.set_autoscale_on(False)                 # stop anything else changing it
        ax.set_xlim(-0.55*env_size, 0.55*env_size)
        ax.set_ylim(-0.55*env_size, 0.55*env_size)
        fig.canvas.draw()
        # plt.legend()
        plt.pause(0.01)






if __name__ == '__main__':
    main()

    
