#!/usr/bin/env python
import rospy
from amcn.optimization.dynamic_mpc_node import DynamicMPCNode

if __name__ == '__main__':
    try:
        rospy.init_node('optimization_node', anonymous=True)

        node = DynamicMPCNode()

        rospy.loginfo("Optimization System Started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
