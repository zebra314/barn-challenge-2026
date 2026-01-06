import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class LidarCluster:
    def __init__(self, config):
        self.cluster_threshold = config.get('cluster_threshold', 0.3)
        self.min_points = config.get('min_points', 3)
        self.max_scan_range = config.get('max_scan_range', 5.0)
        self.radius_inflation = config.get('radius_inflation', 0.1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def process(self, scan_msg):
        """
        Process a LaserScan message to extract obstacles

        Input: scan_msg (sensor_msgs/LaserScan)
        Output: obstacles ([[x, y, theta, a, b], ...])
        """
        # Convert from scan_msg (sensor_msgs/LaserScan)
        # to points in laser frame ([[x1, y1], ...])
        points = self.filter_scan(scan_msg)

        # Convert from points ([[x1, y1], ...])
        # to clusters([[[x1, y1], [x2, y2], ...], ...])
        clusters = self.cluster_points(points)

        # Convert from clusters ([[[x1, y1], [x2, y2], ...], ...])
        # to obstacles in odom frame ([[x, y, theta, a, b], ...])
        obstacles = self.extract_obstacles(clusters, scan_msg.header.frame_id)

        return obstacles

    def filter_scan(self, msg):
        """
        Filter LaserScan message to extract valid points

        Input: msg (sensor_msgs/LaserScan)
        Output: points (Nx2 array)
        """
        ranges = np.array(msg.ranges)
        valid_mask = (ranges > msg.range_min) & \
                     (ranges < self.max_scan_range) & \
                     np.isfinite(ranges)

        # If not enough valid points, return empty
        if np.sum(valid_mask) < self.min_points:
            return np.empty((0, 2))

        valid_ranges = ranges[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        valid_angles = msg.angle_min + valid_indices * msg.angle_increment

        # Convert polar to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)

        return np.column_stack((x, y))

    def cluster_points(self, points):
        """
        Cluster points based on Euclidean distance

        Input: points (Nx2 array)
        Output: list of clusters (each cluster is an Mx2 array)
        """
        if len(points) == 0:
            return []

        clusters = []
        current_cluster = [points[0]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])

            # If the distance is less than the threshold, add to current cluster
            if dist < self.cluster_threshold:
                current_cluster.append(points[i])
                continue

            # Otherwise, finalize the current cluster if it has enough points
            if len(current_cluster) >= self.min_points:
                clusters.append(np.array(current_cluster))

            # Start a new cluster
            current_cluster = [points[i]]

        # Finalize the last cluster
        if len(current_cluster) >= self.min_points:
            clusters.append(np.array(current_cluster))

        return clusters

    def extract_obstacles(self, clusters, source_frame):
        """
        Convert clusters to obstacles in odom frame

        Input: clusters (list of Nx2 arrays), source_frame (str)
        Output format: [[x, y, theta, a, b], ...]
        """
        obstacles = []
        try:
            # Find the latest TF from laser frame to odom frame
            transform = self.tf_buffer.lookup_transform("odom", source_frame, rospy.Time(0), rospy.Duration(0.5))

            # Extract TF yaw angle for calculating obstacle orientation
            q = transform.transform.rotation
            tf_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn_throttle(2.0, "LidarCluster: Cannot transform from {} to odom".format(source_frame))
            return []

        for cluster in clusters:
            # Calculate the best-fit ellipse in the Laser Frame
            # ellipse_local = [x_l, y_l, theta_l, a, b]
            ellipse_local = self.fit_ellipse_pca(cluster)

            if ellipse_local is None:
                continue

            # Transform ellipse center to Odom Frame
            p_stamped = PointStamped()
            p_stamped.header.frame_id = source_frame
            p_stamped.point.x = ellipse_local[0]
            p_stamped.point.y = ellipse_local[1]
            p_stamped.point.z = 0.0

            try:
                p_out = tf2_geometry_msgs.do_transform_point(p_stamped, transform)

                # Rotate ellipse angle to Odom Frame
                # Ellipse angle in Odom = Ellipse angle in Local + Robot(Sensor) angle in Odom
                theta_odom = ellipse_local[2] + tf_yaw

                # Normalize angle to -pi ~ pi
                theta_odom = np.arctan2(np.sin(theta_odom), np.cos(theta_odom))

                # [x, y, theta, semi_major_axis, semi_minor_axis]
                obstacles.append([
                    p_out.point.x,
                    p_out.point.y,
                    theta_odom,
                    ellipse_local[3],
                    ellipse_local[4]
                ])

            except Exception:
                continue

        return obstacles

    def fit_ellipse_pca(self, points):
        """
        Use Principal Component Analysis (PCA) to fit an ellipse

        Input: points (Nx2 array)
        Output: [cx, cy, theta, a, b]
        """
        if len(points) < self.min_points:
            centroid = np.mean(points, axis=0)
            return [centroid[0], centroid[1], 0.0, 0.1, 0.1]

        # Calculate the centroid
        centroid = np.mean(points, axis=0)

        # Decentralize
        centered_points = points - centroid

        # Calculate the covariance matrix
        # row: points, col: dimensions
        cov = np.cov(centered_points, rowvar=False)

        # Handle singular matrix issue caused by collinearity
        if np.isnan(cov).any() or np.isinf(cov).any():
            return None

        # Eigen Decomposition
        # eigenvalues represent variance along axes
        # eigenvectors represent direction of axes
        try:
            eigen_vals, eigen_vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None

        # Sort eigenvalues and eigenvectors in ascending order
        # to find the major and minor axes
        order = eigen_vals.argsort()
        eigen_vals = eigen_vals[order]
        eigen_vecs = eigen_vecs[:, order]

        # Calculate rotation angle
        # using the eigenvector corresponding to the largest eigenvalue [vx, vy]
        major_vec = eigen_vecs[:, 1]
        theta = np.arctan2(major_vec[1], major_vec[0])

        # Calculate semi-axis lengths
        # Construct rotation matrix R(-theta) to flatten points
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])

        # Rotate the centralized points to align with the major axis
        rotated_points = np.dot(centered_points, R.T)

        # Find the bounding box of the rotated points (min_x, max_x, min_y, max_y)
        min_xy = np.min(rotated_points, axis=0)
        max_xy = np.max(rotated_points, axis=0)

        # Semi-axis lengths = half of the range
        a = (max_xy[0] - min_xy[0]) / 2.0
        b = (max_xy[1] - min_xy[1]) / 2.0

        # Add inflation radius
        a += self.radius_inflation
        b += self.radius_inflation

        # Ensure not too small (to avoid numerical issues)
        a = max(a, 0.05)
        b = max(b, 0.05)

        return [centroid[0], centroid[1], theta, a, b]
