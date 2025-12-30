import casadi as ca
import numpy as np

class JackalModel:
    def __init__(self, dt):
        self.dt = dt

    def get_kinematics(self):
        """

        State: [x, y, theta]
        Control: [v, omega]
        """
        #
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)

        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)

        # Differential Drive Kinematics
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = omega
        rhs = ca.vertcat(v * ca.cos(theta),
                         v * ca.sin(theta),
                         omega)

        k1 = rhs
        k2_state = states + 0.5 * self.dt * k1
        states_next = states + rhs * self.dt

        f = ca.Function('F', [states, controls], [states_next])
        return f
