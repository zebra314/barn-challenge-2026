#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from amcn.optimization.mpc_solver import CasadiMPC

class MPCNode:
    def __init__(self):
        rospy.init_node('mpc_controller')

        self.config = {
            'horizon': 20, 'dt': 0.1,
            'v_max': 2.0, 'v_min': 0.0, 'omega_max': 2.0,
            'weight_pos_x': 5.0, 'weight_pos_y': 5.0, 'weight_theta': 2.0,
            'weight_vel': 0.5, 'weight_omega': 0.1
        }

        self.mpc = CasadiMPC(self.config)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.global_path = None
        self.current_odom = None


        rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_cb)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.control_loop)

    def odom_cb(self, msg):
        self.current_odom = msg

    def path_cb(self, msg):
        self.global_path = msg

    def get_robot_state(self):
        if not self.current_odom: return None
        pose = self.current_odom.pose.pose

        # Quaternion to Yaw
        import tf.transformations
        q = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(q)
        return np.array([pose.position.x, pose.position.y, euler[2]])

    def transform_path_to_odom(self, path_msg):
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

        if not transformed_path: return None

        min_dist = float('inf')
        closest_idx = 0
        rx, ry = robot_state[0], robot_state[1]

        path_np = np.array([[p.pose.position.x, p.pose.position.y] for p in transformed_path])
        dists = np.linalg.norm(path_np - np.array([rx, ry]), axis=1)
        closest_idx = np.argmin(dists)

        ref_traj = np.zeros((3, self.config['horizon'] + 1))

        for k in range(self.config['horizon'] + 1):
            idx = min(closest_idx + k, len(transformed_path) - 1)
            pose = transformed_path[idx].pose


            import tf.transformations
            q = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            _, _, yaw = tf.transformations.euler_from_quaternion(q)

            ref_traj[0, k] = pose.position.x
            ref_traj[1, k] = pose.position.y
            ref_traj[2, k] = yaw

        return ref_traj

    def control_loop(self, event):
        if self.global_path is None or self.current_odom is None:
            return

        state = self.get_robot_state()

        odom_path = self.transform_path_to_odom(self.global_path)

        ref_traj = self.get_local_reference(state, odom_path)
        if ref_traj is None: return

        v, omega = self.mpc.solve(state, ref_traj)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    MPCNode()
    rospy.spin()
