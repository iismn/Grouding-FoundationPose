from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'connector_pose_estimation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HandongLee',
    maintainer_email='hdong564@kaist.ac.kr',
    description='Grounded Pose Estimation for zero-shot connector pose estimation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimator = pose_src.ROS_comm:main',
            'pose_mission = pose_src.ROS_mission:main',
        ],
    },
)
