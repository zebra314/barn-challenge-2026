class RobotConfig:
    width = 0.5
    length = 0.7

    dt = 0.1

    max_linear_vel = 2.0
    max_linear_acc = 1.0

    max_angular_vel = 2.0
    max_angular_acc = 3.0

    std_linear_noise = 0.02
    std_angular_noise = 0.05

    max_slip_factor = 0.2 # At max speed, the effective angular velocity is reduced by this factor
