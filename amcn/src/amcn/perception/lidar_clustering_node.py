import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tf_trans

from amcn.perception.lidar_cluster import LidarCluster

class LidarClusteringNode:
    def __init__(self):
        # Get parameters
        cluster_threshold = rospy.get_param('~cluster_threshold', 0.25)
        min_points = rospy.get_param('~min_points', 3)
        max_scan_range = rospy.get_param('~max_scan_range', 5.0)
        radius_inflation = rospy.get_param('~radius_inflation', 0.05)

        # Initialize LidarCluster
        self.detector = LidarCluster(
            cluster_threshold=cluster_threshold,
            min_points=min_points,
            max_scan_range=max_scan_range,
            radius_inflation=radius_inflation
        )

        # Set publishers
        self.pub_obstacles = rospy.Publisher('/obstacles_raw', Float32MultiArray, queue_size=1)
        self.pub_markers = rospy.Publisher('/obstacles_raw_markers', MarkerArray, queue_size=1)

        # Set subscriber
        self.sub_scan = rospy.Subscriber('/front/scan', LaserScan, self.scan_callback, queue_size=1)

        rospy.loginfo("Lidar Clustering Node Started.")

    def scan_callback(self, msg):
        obstacles_list = self.detector.process_scan(msg)

        if obstacles_list:
            self.publish_data(obstacles_list)
            self.publish_markers(obstacles_list)

    def publish_data(self, obstacles):
        # Flatten: [[x, y, th, a, b], ...] -> [x, y, th, a, b, x, y, th, a, b...]
        flat_data = np.array(obstacles).flatten().tolist()
        msg = Float32MultiArray()
        msg.data = flat_data
        self.pub_obstacles.publish(msg)

    def publish_markers(self, obstacles):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, obs in enumerate(obstacles):
            # Obstacle: [x, y, theta, semi_axis_a, semi_axis_b]
            ox, oy, theta, oa, ob = obs

            # Create a cylinder marker for each obstacle
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            # Set pose
            marker.pose.position.x = ox
            marker.pose.position.y = oy
            marker.pose.position.z = 0.5

            # Set scale
            marker.scale.x = oa * 2.0
            marker.scale.y = ob * 2.0
            marker.scale.z = 1.0

            # Convert theta to quaternion
            q = tf_trans.quaternion_from_euler(0, 0, theta)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            # Set color
            marker.color.a = 0.6
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.pub_markers.publish(marker_array)
