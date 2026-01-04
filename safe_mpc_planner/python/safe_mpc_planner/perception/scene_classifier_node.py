import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool

from safe_mpc_planner.perception.scene_classifier import SceneClassifier

class SceneClassifierNode:
    def __init__(self):
        self.collect_duration = rospy.get_param('~collect_duration', 1.0)

        classifier_config = {
            'variance_threshold': rospy.get_param('~variance_threshold', 0.05),
            'dynamic_ratio_threshold': rospy.get_param('~dynamic_ratio_threshold', 0.02),
            'min_valid_range': rospy.get_param('~min_valid_range', 0.2),
            'max_valid_range': rospy.get_param('~max_valid_range', 8.0)
        }

        self.classifier = SceneClassifier(classifier_config)

        self.scans_buffer = []
        self.is_collecting = True
        self.start_time = None

        self.scan_sub = rospy.Subscriber('/front/scan', LaserScan, self.scan_callback)

        self.result_pub = rospy.Publisher('/scene/is_dynamic', Bool, queue_size=1, latch=True)

        rospy.loginfo("[SceneClassifier] Ready. Waiting for laser scans to collect for {:.2f} seconds...".format(self.collect_duration))

    def scan_callback(self, msg):
        if not self.is_collecting:
            return

        if self.start_time is None:
            self.start_time = rospy.Time.now()
            rospy.loginfo("[SceneClassifier] First scan received. Timer started.")

        self.scans_buffer.append(msg.ranges)

        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed > self.collect_duration:
            self.finish_collection()

    def finish_collection(self):
        self.is_collecting = False
        rospy.loginfo("[SceneClassifier] Collection finished. Frames collected: {}".format(len(self.scans_buffer)))

        if len(self.scans_buffer) < 5:
            rospy.logwarn("[SceneClassifier] Not enough frames to analyze! Defaulting to STATIC.")
            self.publish_result(is_dynamic=False)
            return

        result = self.classifier.process(self.scans_buffer)

        is_dynamic = result['is_dynamic']
        ratio = result['ratio']
        count = result['unstable_count']

        rospy.loginfo("[SceneClassifier] Analysis Result: Unstable Beams={}, Ratio={:.4f}".format(count, ratio))

        self.publish_result(is_dynamic)

        mode_str = "dynamic" if is_dynamic else "static"
        rospy.set_param("/mpc/mode", mode_str)

    def publish_result(self, is_dynamic):
        self.result_pub.publish(is_dynamic)

        mode_str = "dynamic" if is_dynamic else "static"
        if is_dynamic:
            rospy.logwarn(">>> SCENE DETECTED: {} (Ratio > {}) <<<".format(mode_str, self.classifier.dynamic_ratio_threshold))
        else:
            rospy.loginfo(">>> SCENE DETECTED: {} (Stable environment) <<<".format(mode_str))
