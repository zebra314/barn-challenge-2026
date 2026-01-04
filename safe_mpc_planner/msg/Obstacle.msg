uint32 id
geometry_msgs/Pose pose        # 位置 (x, y, theta)
geometry_msgs/Twist velocity   # 速度 (vx, vy, omega)
geometry_msgs/Vector3 size     # 長寬高 (或者半徑)
float64 confidence
bool is_static                 # [策略] 標記這個障礙物是否被視為靜態
