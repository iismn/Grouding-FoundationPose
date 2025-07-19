from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'realsense_manage'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SangminLee',
    maintainer_email='iismn@kaist.ac.kr',
    description='Intel RealSense camera manager for multiple cameras',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'realsense_manager = realsense_manage_src.ROS_DPT:main',
        ],
    },
)
