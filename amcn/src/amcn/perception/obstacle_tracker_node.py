import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf.transformations as tf_trans

from amcn.perception.obstacle_tracker import SimpleTracker

class ObstacleTrackerNode:
    def __init__(self):
        dt_threshold = rospy.get_param('~dt_threshold', 1.0)
        self.tracker = SimpleTracker(dt_threshold=dt_threshold)
        self.last_time = rospy.Time.now()

        # 3. Communication
        # topic for raw obstacles from Lidar Clustering
        self.sub_raw = rospy.Subscriber('/obstacles_raw', Float32MultiArray, self.callback, queue_size=1)

        # topic for MPC
        self.pub_tracked = rospy.Publisher('/tracked_obstacles', Float32MultiArray, queue_size=1)

        # topic for rviz debug
        self.pub_debug = rospy.Publisher('/tracker_markers', MarkerArray, queue_size=1)

        rospy.loginfo("Obstacle Tracker Node Started.")

    def callback(self, msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # 1. phrase raw data
        if len(msg.data) == 0:
            raw_obstacles = []
        else:
            raw_obstacles = np.array(msg.data).reshape(-1, 5)

        # 2. execute tracking update
        # tracked_data format: [[x, y, theta, a, b, vx, vy], ...]
        tracked_data = self.tracker.update(raw_obstacles, dt)

        # 3. publish to MPC
        output_msg = Float32MultiArray()
        output_msg.data = np.array(tracked_data).flatten().tolist()
        self.pub_tracked.publish(output_msg)

        # 4. publish visualization (Debug)
        self.publish_markers(current_time)

    def publish_markers(self, timestamp):
        """
        Visualize the internal state of the Tracker (including ID and velocity arrows)
        """
        marker_array = MarkerArray()

        # Clear old Markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # Iterate over all tracked objects
        # We directly read self.tracker.tracks to get the ID
        for tid, track in self.tracker.tracks.items():
            s = track['state'] # [x, y, theta, a, b]
            v = track['vel']   # [vx, vy, omega]

            # --- A. Green cylinder (represents the smoothed position/shape) ---
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = timestamp
            marker.ns = "tracked_shape"
            marker.id = tid
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = s[0]
            marker.pose.position.y = s[1]
            marker.pose.position.z = 0.5

            q = tf_trans.quaternion_from_euler(0, 0, s[2])
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = s[3] * 2.0 # diameter
            marker.scale.y = s[4] * 2.0
            marker.scale.z = 1.0

            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0 # green
            marker.color.b = 0.0
            marker_array.markers.append(marker)

            # --- B. ID text ---
            text_marker = Marker()
            text_marker.header.frame_id = "odom"
            text_marker.header.stamp = timestamp
            text_marker.ns = "tracked_id"
            text_marker.id = tid
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = s[0]
            text_marker.pose.position.y = s[1]
            text_marker.pose.position.z = 1.2
            text_marker.scale.z = 0.4
            text_marker.text = "ID:{}".format(tid)
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            marker_array.markers.append(text_marker)

            # --- C. Velocity arrow (yellow) ---
            speed = (v[0]**2 + v[1]**2)**0.5
            if speed > 0.1: # Do not display if speed is too low
                arrow = Marker()
                arrow.header.frame_id = "odom"
                arrow.header.stamp = timestamp
                arrow.ns = "tracked_vel"
                arrow.id = tid
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD

                p_start = Point(s[0], s[1], 0.5)
                # Predict the position 1 second later as the arrow length
                p_end = Point(s[0] + v[0], s[1] + v[1], 0.5)

                arrow.points = [p_start, p_end]
                arrow.scale.x = 0.1
                arrow.scale.y = 0.2
                arrow.scale.z = 0.0
                arrow.color.a = 1.0
                arrow.color.r = 1.0
                arrow.color.g = 1.0
                arrow.color.b = 0.0
                marker_array.markers.append(arrow)

        self.pub_debug.publish(marker_array)
