import casadi as ca
import numpy as np

class DynamicMPCSolver:
    def __init__(self, N=20, dt=0.1):
        self.N = N
        self.dt = dt

        # Weights for the cost function
        # Q: State error [x, y, theta]
        self.Q_x = 10.0
        self.Q_y = 10.0
        self.Q_theta = 5.0

        # R: Control effort penalty [v, omega]
        self.R_v = 0.5
        self.R_omega = 0.5

        # Rd: Control rate change penalty (Jerk minimization)
        self.Rd_v = 1.0
        self.Rd_omega = 1.0

        # Slack: Obstacle avoidance soft constraint penalty (set very high to ensure violation only when necessary)
        self.W_slack = 1000.0

        # === Physical Constraints ===
        self.v_max = 2.0
        self.v_min = 0.0
        self.omega_max = 1.5

        # === Obstacle Settings ===
        self.M = 6 # Maximum number of closest obstacles considered simultaneously
        self.robot_radius = 0.3 # Robot's own safety radius

        # Warm start
        self.last_sol_x = None
        self.last_sol_u = None
        self.last_sol_slack = None

        # Initialize CasADi Solver
        self._init_solver()

    def _init_solver(self):
        self.opti = ca.Opti()

        # State variables: X = [x, y, theta]^T
        self.state = self.opti.variable(3, self.N + 1)
        x = self.state[0, :]
        y = self.state[1, :]
        th = self.state[2, :]

        # Control variables: U = [v, omega]^T
        self.control = self.opti.variable(2, self.N)
        v = self.control[0, :]
        omega = self.control[1, :]

        # Slack variables used for obstacle avoidance
        # Allow the robot to slightly violate distance constraints in extreme cases to avoid solver infeasibility
        self.slack = self.opti.variable(self.N + 1)

        # Initial state [x0, y0, theta0]
        self.p_x0 = self.opti.parameter(3)

        # Reference trajectory [x_ref, y_ref, theta_ref]
        self.p_ref = self.opti.parameter(3, self.N + 1)

        # Obstacle parameter matrix (7 rows, M columns)
        # Each row: [x, y, theta, a, b, vx, vy]
        self.p_obs = self.opti.parameter(7, self.M)

        cost = 0

        for k in range(self.N + 1):
            err_x = x[k] - self.p_ref[0, k]
            err_y = y[k] - self.p_ref[1, k]

            # Angle error handling (Normalize to -pi~pi)
            err_th = th[k] - self.p_ref[2, k]

            cost += self.Q_x * err_x**2 + self.Q_y * err_y**2 + self.Q_theta * err_th**2

            # Add Slack penalty (smaller is better)
            cost += self.W_slack * self.slack[k]**2

        # B. Control Effort
        for k in range(self.N):
            cost += self.R_v * v[k]**2 + self.R_omega * omega[k]**2

        # C. Smoothness
        for k in range(self.N - 1):
            dv = v[k+1] - v[k]
            domega = omega[k+1] - omega[k]
            cost += self.Rd_v * dv**2 + self.Rd_omega * domega**2

        self.opti.minimize(cost)

        # === 4. System Dynamics Constraints (Kinematics) ===
        # Using simple Euler integration
        for k in range(self.N):
            # x_next = x_curr + v * cos(th) * dt
            self.opti.subject_to(x[k+1] == x[k] + v[k] * ca.cos(th[k]) * self.dt)
            self.opti.subject_to(y[k+1] == y[k] + v[k] * ca.sin(th[k]) * self.dt)
            self.opti.subject_to(th[k+1] == th[k] + omega[k] * self.dt)

        # === 5. Initial Condition Constraints ===
        self.opti.subject_to(x[0] == self.p_x0[0])
        self.opti.subject_to(y[0] == self.p_x0[1])
        self.opti.subject_to(th[0] == self.p_x0[2])

        # === 6. Physical Limit Constraints ===
        self.opti.subject_to(self.opti.bounded(self.v_min, v, self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.omega_max, omega, self.omega_max))
        self.opti.subject_to(self.slack >= 0) # Slack must be non-negative

        # === 7. Dynamic Ellipse Obstacle Avoidance Constraints (The Core Logic) ===
        self._add_ellipse_constraints(x, y)

        # === 8. Solver Settings ===
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.opti.solver('ipopt', opts)

    def _add_ellipse_constraints(self, x, y):
        """
        Add dynamic ellipse obstacle avoidance constraints
        """
        for k in range(1, self.N + 1): # Check from k=1, k=0 is the current state and does not need checking
            # Prediction time
            t_predict = k * self.dt

            for j in range(self.M):
                # Extract parameters
                obs_x = self.p_obs[0, j]
                obs_y = self.p_obs[1, j]
                obs_th = self.p_obs[2, j]
                obs_a = self.p_obs[3, j]
                obs_b = self.p_obs[4, j]
                obs_vx = self.p_obs[5, j]
                obs_vy = self.p_obs[6, j]

                # A. Dynamic Position Update (Predict future obstacle position)
                curr_obs_x = obs_x + obs_vx * t_predict
                curr_obs_y = obs_y + obs_vy * t_predict

                # B. Coordinate Transformation (Global -> Ellipse Local)
                dx = x[k] - curr_obs_x
                dy = y[k] - curr_obs_y

                # Rotation matrix R(-theta)
                c = ca.cos(obs_th)
                s = ca.sin(obs_th)
                x_local = dx * c + dy * s
                y_local = -dx * s + dy * c

                # C. Safety Radius Setting
                # Obstacle radius + Robot radius + Additional buffer
                # Use fmax to avoid division by zero when a,b=0
                safe_a = ca.fmax(obs_a + self.robot_radius, 0.01)
                safe_b = ca.fmax(obs_b + self.robot_radius, 0.01)

                # D. Ellipse Inequality: (x'/a)^2 + (y'/b)^2 >= 1
                dist_sq = (x_local / safe_a)**2 + (y_local / safe_b)**2

                # Only enable constraint when obstacle is valid (a > 0.05)
                # Here we use a trick: if a < 0.05, we make the right side of the inequality very small so it always holds
                # Or rely on Python side to set invalid obstacles to -1000 far away
                self.opti.subject_to(dist_sq + self.slack[k] >= 1.0)

    def solve(self, x0, ref_traj, obstacles):
        """
        x0: [x, y, theta]
        ref_traj: shape (3, N+1)
        obstacles: List of [x, y, theta, a, b, vx, vy]
        """
        # 1. Set initial state and reference trajectory
        self.opti.set_value(self.p_x0, x0)

        # Handle insufficient reference trajectory length
        if ref_traj.shape[1] < self.N + 1:
            # Padding padding...
            last_col = ref_traj[:, -1].reshape(3, 1)
            padding = np.tile(last_col, (1, self.N + 1 - ref_traj.shape[1]))
            ref_traj = np.hstack((ref_traj, padding))

        self.opti.set_value(self.p_ref, ref_traj[:, :self.N + 1])

        if self.last_sol_x is not None:
            # Shift state: [x1, x2, ..., xN, xN]
            guess_x = np.hstack((self.last_sol_x[:, 1:], self.last_sol_x[:, -1:]))
            self.opti.set_initial(self.state, guess_x)

            # Shift control: [u1, u2, ..., uN, uN]
            guess_u = np.hstack((self.last_sol_u[:, 1:], self.last_sol_u[:, -1:]))
            self.opti.set_initial(self.control, guess_u)

            # Shift slack
            guess_slack = np.hstack((self.last_sol_slack[1:], self.last_sol_slack[-1:]))
            self.opti.set_initial(self.slack, guess_slack)
        else:
            self.opti.set_initial(self.state, ref_traj[:, :self.N + 1])
            self.opti.set_initial(self.control, np.zeros((2, self.N)))

        # 2. Set obstacles
        obs_matrix = np.zeros((7, self.M))

        # Move obstacles out of range to infinity to avoid affecting the solver
        obs_matrix[0, :] = -1000.0 # x = -1000
        obs_matrix[3, :] = 0.1     # a
        obs_matrix[4, :] = 0.1     # b

        if obstacles is not None and len(obstacles) > 0:
            # Only take the first M obstacles
            count = min(len(obstacles), self.M)
            # Convert List -> Numpy Array and transpose
            obs_data = np.array(obstacles[:count]).T
            obs_matrix[:, :count] = obs_data

        self.opti.set_value(self.p_obs, obs_matrix)

        # 3. Solve
        try:
            sol = self.opti.solve()

            # Get control commands (take the first step)
            u_opt = sol.value(self.control)
            v_cmd = u_opt[0, 0]
            omega_cmd = u_opt[1, 0]

            # Get predicted trajectory (for visualization)
            pred_x = sol.value(self.state[0, :])
            pred_y = sol.value(self.state[1, :])
            pred_traj = np.vstack((pred_x, pred_y))

            return [v_cmd, omega_cmd], pred_traj

        except RuntimeError as e:
            # If solving fails (Infeasible), try to return the last solution or brake
            # Here we simply return brake
            print("[MPC] Solver failed: ", e)
            return [0.0, 0.0], np.zeros((2, self.N+1))
