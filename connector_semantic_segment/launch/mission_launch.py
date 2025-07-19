#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    parameters = [{
        'segmentA_launch_args': ['text_prompt:=white plastic rectangular connector. If segmented connector multiple, select only one which is far from camera'],
        'segmentB_launch_args': ['text_prompt:=white plastic rectangular connector. If segmented connector multiple, select only one which is far from camera']
    }]
    return LaunchDescription([
        Node(
            package='connector_semantic_segment',
            executable='connector_mission',
            name='connector_mission',
            output='screen',
            parameters=parameters
        )
    ])

if __name__ == '__main__':
    generate_launch_description()
