#!/usr/bin/env python
import rospy
from safe_mpc_planner.perception.lidar_clustering_node import LidarClusteringNode
from safe_mpc_planner.perception.obstacle_tracker_node import ObstacleTrackerNode
from safe_mpc_planner.perception.scene_classifier_node import SceneClassifierNode

if __name__ == '__main__':
    try:
        rospy.init_node('perception_system', anonymous=True)

        scene_classifier_node = SceneClassifierNode()
        lidar_clustering_node = LidarClusteringNode()
        obstacle_tracker_node = ObstacleTrackerNode()

        rospy.loginfo("Perception System Started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
