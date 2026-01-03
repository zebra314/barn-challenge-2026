import casadi as ca

class JackalModel:
    def __init__(self, dt):
        self.dt = dt
        self.turning_efficiency = 1.0

        # Instantaneous center of rotation distance from robot center
        self.x_icr = 0.0

    def get_kinematics(self):
        """
        Returns the Kinematic Skid-Steer Model
        """

        # State
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)

        # Control
        v_cmd = ca.SX.sym('v_cmd')
        omega_cmd = ca.SX.sym('w_cmd')
        controls = ca.vertcat(v_cmd, omega_cmd)

        # Convert commands to body frame velocities
        v_x_body = v_cmd
        omega_body = omega_cmd * self.turning_efficiency

        # Calculate lateral velocity based on ICR
        v_y_body = self.x_icr * omega_body

        # Calculate world frame velocities
        x_dot = v_x_body * ca.cos(theta) - v_y_body * ca.sin(theta)
        y_dot = v_x_body * ca.sin(theta) + v_y_body * ca.cos(theta)
        theta_dot = omega_body

        rhs = ca.vertcat(x_dot, y_dot, theta_dot)

        # Runge-Kutta 4 Integration
        f_dynamic = ca.Function('f_dynamic', [states, controls], [rhs])

        k1 = f_dynamic(states, controls)
        k2 = f_dynamic(states + self.dt / 2 * k1, controls)
        k3 = f_dynamic(states + self.dt / 2 * k2, controls)
        k4 = f_dynamic(states + self.dt * k3, controls)
        states_next = states + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return ca.Function('F', [states, controls], [states_next])
