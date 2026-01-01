#!/usr/bin/env python
import rospy
from amcn.perception.lidar_clustering_node import LidarClusteringNode
from amcn.perception.obstacle_tracker_node import ObstacleTrackerNode

if __name__ == '__main__':
    try:
        rospy.init_node('perception_system', anonymous=True)

        lidar_clustering_node = LidarClusteringNode()
        obstacle_tracker_node = ObstacleTrackerNode()

        rospy.loginfo("Perception System Started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
