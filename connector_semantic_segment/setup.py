from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'connector_semantic_segment'

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
    maintainer='SangminLee',
    maintainer_email='iismn@kaist.ac.kr',
    description='Grounded semantic segmentation for zero-shot connector pose estimation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'connector_segmentor = segment_src.ROS_comm:main',
            'connector_mission = segment_src.ROS_mission:main',
        ],
    },
)
