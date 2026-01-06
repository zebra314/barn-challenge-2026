import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import tf.transformations
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32MultiArray

from safe_mpc_planner.optimization.mpc_solver import MPCSolver

class MPCNode:
    def __init__(self):
        self.config = rospy.get_param('~mpc')['static']
        self.solver = MPCSolver(self.config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.is_dynamic = None
        self.global_path = None
        self.current_odom = None
        self.current_obstacles = []

        rospy.Subscriber('/scene/is_dynamic', Bool, self.is_dynamic_cb)
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

    def is_dynamic_cb(self, msg):
        """
        msg.data: True (Dynamic), False (Static)
        """
        self.is_dynamic = msg.data

        if self.is_dynamic:
            self.config = rospy.get_param('~mpc')['dynamic']
            rospy.logwarn("[MPCNode] Scene detected as DYNAMIC.")
        else:
            self.config = rospy.get_param('~mpc')['static']
            rospy.loginfo("[MPCNode] Scene detected as STATIC.")

        self.solver.update_params(self.config)

    def get_robot_state(self):
        if not self.current_odom:
            return None

        pose = self.current_odom.pose.pose

        # Quaternion to Yaw
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

        if self.is_dynamic is None:
            if int(rospy.get_time()) % 2 == 0:
                rospy.loginfo_throttle(2.0, "[MPCNode] Waiting for Scene Classifier result...")
            return

        state = self.get_robot_state()
        odom_path = self.transform_path_to_odom(self.global_path)
        ref_traj = self.solver.get_reference_traj(state, odom_path)

        if ref_traj is None:
            return

        try:
            control, pred_traj = self.solver.solve(state, ref_traj, self.current_obstacles)

            # Extract control commands
            v = control[0]
            omega = control[1]

            # Publish control commands
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)

            # Publish predicted trajectory for visualization
            self.publish_predicted_path(pred_traj)

        except Exception as e:
            rospy.logerr_throttle(1.0, "MPC Solve Error: {}".format(e))
            self.cmd_pub.publish(Twist())
