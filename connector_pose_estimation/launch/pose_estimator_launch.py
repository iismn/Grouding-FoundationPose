import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

configurable_parameters = [
    {'name': 'ws_path',                 'default': '/home/iris/Workspace/ROS2',                                                 'description': 'ROS2 워크스페이스 경로'},
    {'name': 'mesh_filename',           'default': 'ConnectorA.obj',                                                            'description': '커넥터 모델(.obj) 파일 이름'},
    {'name': 'est_refine_iter',         'default': '5',                                                                         'description': '초기 추정 시 refinement 반복 횟수'},
    {'name': 'track_refine_iter',       'default': '3',                                                                         'description': '트래킹 시 refinement 반복 횟수'},
    {'name': 'debug',                   'default': '1',                                                                         'description': '디버그 모드(숫자가 높을수록 상세)'},
    {'name': 'disable_logging',         'default': 'true',                                                                      'description': '로그 출력 비활성화'},
    {'name': 'color_topic',             'default': '/camera/color/image_raw/compressed',                                        'description': '컬러 이미지 토픽'},
    {'name': 'depth_topic',             'default': '/camera/aligned_depth_to_color/image_raw/compressedDepth',                  'description': '정렬된 Depth 이미지 토픽'},
    {'name': 'mask_topic',              'default': '/poseEstimation/connector/mask/image/compressed',                           'description': '마스크 이미지 토픽'},
    {'name': 'camera_info_topic',       'default': '/camera/aligned_depth_to_color/camera_info',                                'description': '카메라 정보 토픽'},
    {'name': 'pose_pub_topic',          'default': '/poseEstimation/connector/estimated_pose',                                  'description': '추정한 포즈를 퍼블리시하는 토픽'},
    {'name': 'vis_pub_topic',           'default': '/poseEstimation/connector/estimated_pose/visualization/compressed',         'description': '시각화 이미지를 퍼블리시하는 토픽'},
    {'name': 'lpf_pose',                'default': '0.6',                                                                       'description': '포즈에 대한 저역통과필터 계수'},
    {'name': 'lpf_slerp',               'default': '0.6',                                                                       'description': '회전에 대한 저역통과필터 계수'}
]

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def launch_setup(context, *args, **kwargs):
    param_values = set_configurable_parameters(configurable_parameters)
    
    ws_path = param_values['ws_path'].perform(context)
    mesh_filename = param_values['mesh_filename'].perform(context)
    est_refine_iter = param_values['est_refine_iter'].perform(context)
    track_refine_iter = param_values['track_refine_iter'].perform(context)
    debug = param_values['debug'].perform(context)
    disable_logging = param_values['disable_logging'].perform(context)
    color_topic = param_values['color_topic'].perform(context)
    depth_topic = param_values['depth_topic'].perform(context)
    mask_topic = param_values['mask_topic'].perform(context)
    camera_info_topic = param_values['camera_info_topic'].perform(context)
    pose_pub_topic = param_values['pose_pub_topic'].perform(context)
    vis_pub_topic = param_values['vis_pub_topic'].perform(context)
    lpf_pose = param_values['lpf_pose'].perform(context)
    lpf_slerp = param_values['lpf_slerp'].perform(context)

    package_name = 'connector_pose_estimation'
    package_dir = os.path.join(os.environ.get('ROS_WS', ws_path), 'src', package_name)
    mesh_file = os.path.join(package_dir, 'mesh', mesh_filename)
    debug_dir = os.path.join(package_dir, 'debug')

    parameters = [{
        'mesh_file': mesh_file,
        'est_refine_iter': int(est_refine_iter),
        'track_refine_iter': int(track_refine_iter),
        'debug': int(debug),
        'debug_dir': debug_dir,
        'disable_logging': (disable_logging.lower() == 'true'),
        'color_topic': color_topic,
        'depth_topic': depth_topic,
        'mask_topic': mask_topic,
        'camera_info_topic': camera_info_topic,
        'pose_pub_topic': pose_pub_topic,
        'vis_pub_topic': vis_pub_topic,
        'lpf_pose': float(lpf_pose),
        'lpf_slerp': float(lpf_slerp),
    }]

    node_arguments = []
    if disable_logging.lower() == 'true':
        node_arguments.append('--disable_logging')

    pose_estimator_node = Node(
        package=package_name,
        executable='pose_estimator',
        name='pose_estimator',
        output='screen',
        parameters=parameters,
        arguments=node_arguments
    )

    return [pose_estimator_node]
    
def generate_launch_description():
    return LaunchDescription(declare_configurable_parameters(configurable_parameters) + [
        OpaqueFunction(function=launch_setup)
    ])
