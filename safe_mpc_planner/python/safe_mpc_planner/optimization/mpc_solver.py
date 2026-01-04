import logging
import casadi as ca
import numpy as np

from safe_mpc_planner.optimization.robot_model import JackalModel

class MPCSolver:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.horizon = config['horizon']
        self.dt = config['dt']

        self.opti = ca.Opti()
        self.model = JackalModel(self.dt)
        self.kinematics = self.model.get_kinematics()

        # Decision variables
        self.state = self.opti.variable(3, self.horizon + 1) # [x, y, theta]^T
        self.control = self.opti.variable(2, self.horizon) # [v, omega]^T
        self.slack = self.opti.variable(1, self.horizon + 1)

        self.x0 = self.opti.parameter(3)
        self.ref_traj = self.opti.parameter(3, self.horizon + 1) # [x, y, theta]^T

        # Robot limits
        self.max_v = self.opti.parameter()
        self.min_v = self.opti.parameter()
        self.max_omega = self.opti.parameter()

        # State costs
        self.Q_x = self.opti.parameter()
        self.Q_y = self.opti.parameter()
        self.Q_theta = self.opti.parameter()

        # Control costs
        self.R_v = self.opti.parameter()
        self.R_omega = self.opti.parameter()
        self.Rd_v = self.opti.parameter()
        self.Rd_omega = self.opti.parameter()

        # Slack
        self.W_slack = self.opti.parameter()

        # Obstacle
        self.max_obs_num = config['max_obs_num']
        self.robot_radius = self.opti.parameter()
        self.obstacles = self.opti.parameter(7, self.max_obs_num) # [x, y, theta, a, b, vx, vy]^T

        # Warm start
        self.last_sol_x = None
        self.last_sol_u = None
        self.last_sol_slack = None

        self.init_solver()
        self.update_params(config)

    def update_params(self, config):
        # Robot limits
        self.opti.set_value(self.max_v, config['max_v'])
        self.opti.set_value(self.min_v, config['min_v'])
        self.opti.set_value(self.max_omega, config['max_omega'])

        # State costs
        self.opti.set_value(self.Q_x, config['Q_x'])
        self.opti.set_value(self.Q_y, config['Q_y'])
        self.opti.set_value(self.Q_theta, config['Q_theta'])

        # Control costs
        self.opti.set_value(self.R_v, config['R_v'])
        self.opti.set_value(self.R_omega, config['R_omega'])
        self.opti.set_value(self.Rd_v, config['Rd_v'])
        self.opti.set_value(self.Rd_omega, config['Rd_omega'])

        self.opti.set_value(self.W_slack, config['W_slack'])

        # Obstacle
        self.opti.set_value(self.max_obs_num, config['max_obs_num'])
        self.opti.set_value(self.robot_radius, config['robot_radius'])

    def init_solver(self):
        # State
        x = self.state[0, :]
        y = self.state[1, :]
        th = self.state[2, :]

        # Control
        v = self.control[0, :]
        omega = self.control[1, :]

        # --------------------------------- Set costs -------------------------------- #
        cost = 0

        # State error
        err_x = x - self.ref_traj[0, :]
        err_y = y - self.ref_traj[1, :]
        err_th = th - self.ref_traj[2, :]

        cost += ca.sum2(self.Q_x * err_x**2)
        cost += ca.sum2(self.Q_y * err_y**2)
        cost += ca.sum2(self.Q_theta * (1 - ca.cos(err_th)))

        # Control error
        err_v = v[1:] - v[:-1]
        err_omega = omega[1:] - omega[:-1]

        cost += ca.sum2(self.R_v * v**2)
        cost += ca.sum2(self.R_omega * omega**2)
        cost += ca.sum2(self.Rd_v * err_v**2)
        cost += ca.sum2(self.Rd_omega * err_omega**2)

        # Slack
        cost += ca.sum2(self.W_slack * self.slack**2)

        self.opti.minimize(cost)

        # ------------------------------ Set constraints ------------------------------ #
        # System dynamics constraints
        for k in range(self.horizon):
            x_next = self.kinematics(self.state[:, k], self.control[:, k])
            self.opti.subject_to(self.state[:, k+1] == x_next)

        # Initial condition constraints
        self.opti.subject_to(self.state[:, 0] == self.x0)

        # Physical limit constraints
        self.opti.subject_to(self.opti.bounded(self.min_v, v, self.max_v))
        self.opti.subject_to(self.opti.bounded(-self.max_omega, omega, self.max_omega))

        # Slack must be non-negative
        self.opti.subject_to(self.slack >= 0)

        # Obstacle avoidance constraints
        self.add_ellipse_constraints(x, y)

        # Solver settings
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.opti.solver('ipopt', opts)

    def add_ellipse_constraints(self, x, y):
        """
        Add dynamic ellipse obstacle avoidance constraints
        """
        # Check from next time step
        for k in range(1, self.horizon + 1):
            # Prediction time
            t_predict = k * self.dt

            for j in range(self.max_obs_num):
                # Extract parameters
                obs_x = self.obstacles[0, j]
                obs_y = self.obstacles[1, j]
                obs_th = self.obstacles[2, j]
                obs_a = self.obstacles[3, j]
                obs_b = self.obstacles[4, j]
                obs_vx = self.obstacles[5, j]
                obs_vy = self.obstacles[6, j]

                # Linear motion
                curr_obs_x = obs_x + obs_vx * t_predict
                curr_obs_y = obs_y + obs_vy * t_predict

                # Transform frame from world to obstacle
                dx = x[k] - curr_obs_x
                dy = y[k] - curr_obs_y

                c = ca.cos(obs_th)
                s = ca.sin(obs_th)
                x_local = dx * c + dy * s
                y_local = -dx * s + dy * c

                # Safety margin
                safe_a = ca.fmax(obs_a + self.robot_radius, 0.01)
                safe_b = ca.fmax(obs_b + self.robot_radius, 0.01)

                # Ellipse Distance
                dist_sq = (x_local / safe_a)**2 + (y_local / safe_b)**2

                # Soft constraint
                self.opti.subject_to(dist_sq + self.slack[k] >= 1.0)

    def solve(self, x0, ref_traj, obstacles):
        """
        x0: [x, y, theta]
        ref_traj: shape (3, N+1)
        obstacles: List of [x, y, theta, a, b, vx, vy]
        """
        self.opti.set_value(self.x0, x0)

        # Unwrap theta to avoid discontinuities
        ref_traj[2, :] = np.unwrap(ref_traj[2, :])

        # Padding reference if too short
        if ref_traj.shape[1] < self.horizon + 1:
            last_col = ref_traj[:, -1].reshape(3, 1)
            padding = np.tile(last_col, (1, self.horizon + 1 - ref_traj.shape[1]))
            ref_traj = np.hstack((ref_traj, padding))

        self.opti.set_value(self.ref_traj, ref_traj[:, :self.horizon + 1])

        # Set warm start
        if self.last_sol_x is not None:
            guess_x = np.hstack((self.last_sol_x[:, 1:], self.last_sol_x[:, -1:]))
            guess_u = np.hstack((self.last_sol_u[:, 1:], self.last_sol_u[:, -1:]))
            guess_slack = np.hstack((self.last_sol_slack[1:], self.last_sol_slack[-1:]))

            self.opti.set_initial(self.state, guess_x)
            self.opti.set_initial(self.control, guess_u)
            self.opti.set_initial(self.slack, guess_slack)
        else:
            # Cold start
            # assume robot stays at ref trajectory or current pose
            self.opti.set_initial(self.state, ref_traj[:, :self.horizon + 1])
            self.opti.set_initial(self.control, np.zeros((2, self.horizon)))
            self.opti.set_initial(self.slack, np.zeros(self.horizon + 1))

        # Set obstacles
        obs_matrix = np.zeros((7, self.max_obs_num))

        # Move obstacles out of range to infinity to avoid affecting the solver
        obs_matrix[0, :] = 1e6
        obs_matrix[1, :] = 1e6
        obs_matrix[3, :] = 0.1
        obs_matrix[4, :] = 0.1

        if obstacles is not None and len(obstacles) > 0:
            # Only take the first M obstacles
            count = min(len(obstacles), self.max_obs_num)
            # Convert List -> Numpy Array and transpose
            obs_data = np.array(obstacles[:count]).T
            obs_matrix[:, :count] = obs_data

        self.opti.set_value(self.obstacles, obs_matrix)

        try:
            sol = self.opti.solve()

            # Store last solution for warm start
            self.last_sol_x = sol.value(self.state)
            self.last_sol_u = sol.value(self.control)
            self.last_sol_slack = sol.value(self.slack)

            # Extract command
            u_opt = sol.value(self.control)
            v_cmd = u_opt[0, 0]
            omega_cmd = u_opt[1, 0]

            # Extract trajectory
            pred_traj = sol.value(self.state) # (3, N+1)

            return [v_cmd, omega_cmd], pred_traj

        except RuntimeError as e:
            self.logger.warning("MPC solver failed: %s", e)
            return [0.0, 0.0], np.zeros((3, self.horizon+1))
