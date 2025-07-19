#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    parameters = [{
        'camera1_launch_args': ['camera_name:=camera', 'serial_no:=f1422940', 'output:=log'],
        'camera2_launch_args': ['camera_name:=camera', 'serial_no:=f1422241', 'output:=log']
    }]
    
    return LaunchDescription([
        Node(
            package='realsense_manage', 
            executable='realsense_manager',
            name='realsense_manager',
            output='screen',
            parameters=parameters
        )
    ])

if __name__ == '__main__':
    generate_launch_description()
