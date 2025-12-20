#!/bin/bash
source /opt/ros/melodic/setup.bash
catkin_make -C /jackal_ws
source /jackal_ws/devel/setup.bash
cd /jackal_ws/src/the-barn-challenge
exec ${@:1}
