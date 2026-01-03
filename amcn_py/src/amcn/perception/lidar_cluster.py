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
        Output: List of [x, y, theta, semi_axis_a, semi_axis_b] (in Odom frame)
        """
        # 1. Preprocessing
        points = self._preprocess_scan(scan_msg)
        if len(points) == 0:
            return []

        # 2. Clustering
        clusters = self._perform_clustering(points)

        # 3. Feature extraction (Ellipse fitting) and coordinate transformation
        obstacles_odom = self._extract_obstacles(clusters, scan_msg.header.frame_id)

        return obstacles_odom

    def _preprocess_scan(self, msg):
        ranges = np.array(msg.ranges)
        # Filter valid ranges
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
        """
        Convert clusters to obstacles in Odom frame
        Input: clusters (list of Nx2 arrays), source_frame (str)
        Output format: [[x, y, theta, a, b], ...]
        """
        obstacles = []
        try:
            # Find the latest TF from Laser Frame to Odom Frame
            transform = self.tf_buffer.lookup_transform("odom", source_frame, rospy.Time(0), rospy.Duration(0.5))

            # Extract TF Yaw angle (used for rotating the ellipse direction)
            q = transform.transform.rotation
            tf_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn_throttle(2.0, "LidarCluster: Cannot transform from {} to odom".format(source_frame))
            return []

        for cluster in clusters:
            # 1. Calculate the best-fit ellipse in the Laser Frame
            # ellipse_local = [x_l, y_l, theta_l, a, b]
            ellipse_local = self._fit_ellipse_pca(cluster)

            if ellipse_local is None:
                continue

            # 2. Position Transform
            p_stamped = PointStamped()
            p_stamped.header.frame_id = source_frame
            p_stamped.point.x = ellipse_local[0]
            p_stamped.point.y = ellipse_local[1]
            p_stamped.point.z = 0.0

            try:
                p_out = tf2_geometry_msgs.do_transform_point(p_stamped, transform)

                # 3. Rotation Transform
                # Ellipse angle in Odom = Ellipse angle in Local + Robot(Sensor) angle in Odom
                theta_odom = ellipse_local[2] + tf_yaw

                # Normalize angle to -pi ~ pi
                theta_odom = np.arctan2(np.sin(theta_odom), np.cos(theta_odom))

                # Output format: [x, y, theta, semi_major_axis, semi_minor_axis]
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

    def _fit_ellipse_pca(self, points):
        """
        Use PCA (Principal Component Analysis) to fit an ellipse
        Input: points (Nx2 array)
        Output: [cx, cy, theta, a, b]
        """
        if len(points) < 3:
            # Too few points to form an area, degenerate to a small circle
            centroid = np.mean(points, axis=0)
            return [centroid[0], centroid[1], 0.0, 0.1, 0.1]

        # 1. Calculate the centroid
        centroid = np.mean(points, axis=0)

        # 2. Decentralize
        centered_points = points - centroid

        # 3. Calculate the covariance matrix
        # rowvar=False means each row represents a sample (point), each column represents a variable (x, y)
        cov = np.cov(centered_points, rowvar=False)

        # Handle singular matrix issue caused by collinearity
        if np.isnan(cov).any() or np.isinf(cov).any():
            return None

        # 4. Eigen Decomposition
        # vals: eigenvalues (represent variance along axes), vecs: eigenvectors (represent direction of axes)
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None

        # Sort: ensure the largest eigenvalue corresponds to the major axis (Primary Axis)
        # argsort defaults to ascending order, so the last one is the largest
        order = vals.argsort()
        vals = vals[order]
        vecs = vecs[:, order]

        # 5. Calculate rotation angle (theta)
        # Take the eigenvector corresponding to the largest eigenvalue [vx, vy]
        major_vec = vecs[:, 1]
        theta = np.arctan2(major_vec[1], major_vec[0])

        # 6. Calculate semi-axis lengths (a, b)
        # We need to use an "Axis Aligned Bounding Box" (in the rotated coordinate system) to enclose all points

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
