import casadi as ca
import numpy as np
from amcn.optimization.robot_model import JackalModel

class CasadiMPC:
    def __init__(self, config):
        self.N = config['horizon']
        self.dt = config['dt']

        self.model = JackalModel(self.dt)
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(3, self.N + 1) # State [x, y, theta]
        self.U = self.opti.variable(2, self.N)     # Control [v, omega]

        # Current state
        self.P_start = self.opti.parameter(3)

        # Reference trajectory, allowing changing targets at runtime without rebuilding the problem
        self.P_ref = self.opti.parameter(3, self.N+1)

        # Get kinematic constraints
        F = self.model.get_kinematics()

        cost = 0

        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.P_start)

        for k in range(self.N):
            # Kinematic constraints (Multiple Shooting)
            x_next = F(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

            # State error cost
            error_state = self.X[:, k] - self.P_ref[:, k]
            Q = ca.diag([config['weight_pos_x'], config['weight_pos_y'], config['weight_theta']])
            cost += ca.mtimes([error_state.T, Q, error_state])

            # Control effort cost
            R = ca.diag([config['weight_vel'], config['weight_omega']])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])

        # Minimize
        self.opti.minimize(cost)

        # Constraints
        self.opti.subject_to(self.opti.bounded(config['v_min'], self.U[0, :], config['v_max']))
        self.opti.subject_to(self.opti.bounded(-config['omega_max'], self.U[1, :], config['omega_max']))

        # Set options
        plugin_options = {
            'expand': True,
            'print_time': False
        }
        solver_options = {
            'max_iter': 100,
            'print_level': 0,
            'sb': 'yes', # suppress the startup banner
            'print_user_options': 'no',
        }
        self.opti.solver('ipopt', plugin_options, solver_options)

    def solve(self, current_state, reference_trajectory):
        """
        current_state: [x, y, theta]
        reference_trajectory: shape (3, N+1) -> Local trajectory extracted from the global path
        """
        # Set parameter values
        self.opti.set_value(self.P_start, current_state)
        self.opti.set_value(self.P_ref, reference_trajectory)

        # Warm Start
        if hasattr(self, 'last_X_sol'):
            self.opti.set_initial(self.X, self.last_X_sol)
        else:
            self.opti.set_initial(self.X, np.tile(current_state.reshape(-1,1), (1, self.N+1)))

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U[:, 0]) # the first control input
            self.last_X_sol = sol.value(self.X)
            return u_opt[0], u_opt[1] # v, omega
        except RuntimeError:
            print("Solver Failed! executing backup behavior...")
            return 0.0, 0.0
