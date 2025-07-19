#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    parameters = [{
        'poseEstmationA_launch_args': ['mesh_filename:=CONNECTOR-A__MALEv1.obj', 'pose_pub_topic:=/poseEstimation/connectorM/estimated_pose'],
        'poseEstmationB_launch_args': ['mesh_filename:=CONNECTOR-A__FEMALEv1.obj', 'pose_pub_topic:=/poseEstimation/connectorF/estimated_pose']
    }]
    return LaunchDescription([
        Node(
            package='connector_pose_estimation',
            executable='pose_mission',
            name='pose_mission',
            output='screen',
            parameters=parameters
        )
    ])

if __name__ == '__main__':
    generate_launch_description()
