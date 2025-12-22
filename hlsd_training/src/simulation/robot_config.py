class RobotConfig:
    DT = 0.1             # time step (s)
    MAX_V = 2.0          # maximum linear velocity (m/s)
    MAX_W = 2.0          # maximum angular velocity (rad/s)
    ACC_LIM_V = 1.0      # linear acceleration limit (m/s^2)
    ACC_LIM_W = 3.0      # angular acceleration limit (rad/s^2)

    NOISE_V_STD = 0.02   # noise in linear velocity
    NOISE_W_STD = 0.05   # noise in angular velocity

    SLIP_FACTOR_MAX = 0.2 # lost 20% of commanded velocity at max speed
