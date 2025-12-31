import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class LidarCluster:
    def __init__(self, cluster_threshold=0.3, min_points=3, max_scan_range=5.0, radius_inflation=0.05):
        self.threshold = cluster_threshold
        self.min_points = min_points
        self.max_scan_range = max_scan_range
        self.radius_inflation = radius_inflation

        # TF Buffer for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def process_scan(self, scan_msg):
        """
        Input: sensor_msgs/LaserScan
        Output: List of [x, y, radius] (in Odom frame)
        """
        # 1. Preprocessing
        points = self._preprocess_scan(scan_msg)
        if len(points) == 0:
            return []

        # 2. Clustering
        clusters = self._perform_clustering(points)

        # 3. Feature extraction and coordinate transformation
        obstacles_odom = self._extract_obstacles(clusters, scan_msg.header.frame_id)

        return obstacles_odom

    def _preprocess_scan(self, msg):
        ranges = np.array(msg.ranges)
        valid_mask = (ranges > msg.range_min) & (ranges < self.max_scan_range) & np.isfinite(ranges)

        if np.sum(valid_mask) < self.min_points:
            return []

        valid_ranges = ranges[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        valid_angles = msg.angle_min + valid_indices * msg.angle_increment

        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        return np.column_stack((x, y))

    def _perform_clustering(self, points):
        clusters = []
        if len(points) == 0:
            return clusters

        current_cluster = [points[0]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            if dist < self.threshold:
                current_cluster.append(points[i])
            else:
                if len(current_cluster) >= self.min_points:
                    clusters.append(np.array(current_cluster))
                current_cluster = [points[i]]

        if len(current_cluster) >= self.min_points:
            clusters.append(np.array(current_cluster))

        return clusters

    def _extract_obstacles(self, clusters, source_frame):
        obstacles = []
        try:
            # Find the latest TF
            transform = self.tf_buffer.lookup_transform("odom", source_frame, rospy.Time(0), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn_throttle(2.0, "LidarCluster: Cannot transform from {} to odom".format(source_frame))
            return []

        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            dists = np.linalg.norm(cluster - centroid, axis=1)
            radius = np.max(dists) + self.radius_inflation

            p_stamped = PointStamped()
            p_stamped.header.frame_id = source_frame
            p_stamped.point.x = centroid[0]
            p_stamped.point.y = centroid[1]
            p_stamped.point.z = 0.0

            try:
                p_out = tf2_geometry_msgs.do_transform_point(p_stamped, transform)
                obstacles.append([p_out.point.x, p_out.point.y, radius])
            except Exception:
                continue

        return obstacles
