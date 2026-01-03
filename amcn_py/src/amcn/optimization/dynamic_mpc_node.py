import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32MultiArray

from amcn.optimization.dynamic_mpc_solver import DynamicMPCSolver

class DynamicMPCNode:
    def __init__(self):
        self.config = rospy.get_param('~mpc')
        print("Loaded config:", self.config)

        self.solver = DynamicMPCSolver(N=self.config['horizon'], dt=self.config['dt'])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.global_path = None
        self.current_odom = None
        self.current_obstacles = []

        rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.path_cb)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)
        rospy.Subscriber('/tracked_obstacles', Float32MultiArray, self.obstacle_cb)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pred_path_pub = rospy.Publisher('/mpc/predicted_path', Path, queue_size=1)

        rospy.Timer(rospy.Duration(self.config['dt']), self.control_loop)

    def odom_cb(self, msg):
        self.current_odom = msg

    def path_cb(self, msg):
        self.global_path = msg

    def obstacle_cb(self, msg):
        """
        Handler for obstacle data
        Format: [x, y, theta, a, b, vx, vy, ...] (7 elements per obstacle)
        """
        if not msg.data:
            self.current_obstacles = []
        else:
            # Reshape to a list of (N, 7)
            try:
                self.current_obstacles = np.array(msg.data).reshape(-1, 7).tolist()
            except ValueError:
                self.current_obstacles = []

    def get_robot_state(self):
        if not self.current_odom: return None
        pose = self.current_odom.pose.pose

        # Quaternion to Yaw
        import tf.transformations
        q = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(q)
        return np.array([pose.position.x, pose.position.y, euler[2]])

    def transform_path_to_odom(self, path_msg):
        if not path_msg: return None

        if path_msg.header.frame_id == "odom":
            return path_msg.poses

        try:
            transform = self.tf_buffer.lookup_transform("odom", path_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            new_path = []
            for pose in path_msg.poses:
                transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
                new_path.append(transformed_pose)
            return new_path
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("TF Error: Cannot transform path to odom frame")
            return None

    def get_local_reference(self, robot_state, transformed_path):
        if not transformed_path:
            return None

        rx, ry = robot_state[0], robot_state[1]
        path_np = np.array([[p.pose.position.x, p.pose.position.y] for p in transformed_path])
        dists = np.linalg.norm(path_np - np.array([rx, ry]), axis=1)
        closest_idx = np.argmin(dists)

        ref_traj = np.zeros((3, self.config['horizon'] + 1))

        target_vel = self.config.get('v_max', 2.0) * 0.6
        dt = self.config.get('dt', 0.1)
        step_dist = target_vel * dt

        curr_idx = closest_idx

        for k in range(self.config['horizon'] + 1):
            pose = transformed_path[curr_idx].pose.position
            ref_traj[0, k] = pose.x
            ref_traj[1, k] = pose.y

            if k < self.config['horizon']:
                temp_idx = curr_idx
                temp_dist = 0.0
                while temp_idx < len(transformed_path) - 1:
                    p1 = transformed_path[temp_idx].pose.position
                    p2 = transformed_path[temp_idx+1].pose.position
                    d = np.hypot(p2.x - p1.x, p2.y - p1.y)
                    temp_dist += d
                    temp_idx += 1
                    if temp_dist >= step_dist * 0.5:
                        break

                next_p = transformed_path[temp_idx].pose.position
                yaw = np.arctan2(next_p.y - pose.y, next_p.x - pose.x)
            else:
                yaw = ref_traj[2, k-1]

            # Yaw unwrapping
            if k == 0:
                base_yaw = robot_state[2]
            else:
                base_yaw = ref_traj[2, k-1]

            while yaw - base_yaw > np.pi:
                yaw -= 2 * np.pi
            while yaw - base_yaw < -np.pi:
                yaw += 2 * np.pi

            ref_traj[2, k] = yaw

            dist_travelled = 0.0
            while curr_idx < len(transformed_path) - 1:
                p1 = transformed_path[curr_idx].pose.position
                p2 = transformed_path[curr_idx+1].pose.position
                d = np.hypot(p2.x - p1.x, p2.y - p1.y)

                dist_travelled += d
                curr_idx += 1

                if dist_travelled >= step_dist:
                    break

        return ref_traj

    def publish_predicted_path(self, pred_traj_np):
        """
        Convert the numpy array output from the solver to a Path message and publish it
        pred_traj_np: shape (2, N+1) -> [[x0, x1...], [y0, y1...]]
        """
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = rospy.Time.now()

        for i in range(pred_traj_np.shape[1]):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = pred_traj_np[0, i]
            pose.pose.position.y = pred_traj_np[1, i]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.pred_path_pub.publish(msg)

    def control_loop(self, event):
        if self.global_path is None or self.current_odom is None:
            return

        state = self.get_robot_state()
        odom_path = self.transform_path_to_odom(self.global_path)

        ref_traj = self.get_local_reference(state, odom_path)
        if ref_traj is None: return

        try:
            control, pred_traj = self.solver.solve(state, ref_traj, self.current_obstacles)

            # 4. Extract control commands
            v = control[0]
            omega = control[1]

            # 5. Publish control commands
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)

            # 6. Publish predicted trajectory for visualization
            self.publish_predicted_path(pred_traj)

        except Exception as e:
            rospy.logerr_throttle(1.0, "MPC Solve Error: {}".format(e))
            self.cmd_pub.publish(Twist())
