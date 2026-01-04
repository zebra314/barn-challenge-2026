#!/usr/bin/env python
import rospy
from safe_mpc_planner.optimization.mpc_node import MPCNode

if __name__ == '__main__':
    try:
        rospy.init_node('optimization_node', anonymous=True)

        node = MPCNode()

        rospy.loginfo("Optimization System Started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
