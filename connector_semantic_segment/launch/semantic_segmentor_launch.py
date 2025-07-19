import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

configurable_parameters = [
    {'name': 'ws_path',                         'default': '/home/iris/Workspace/ROS2',                                                     'description': 'ROS2 워크스페이스 경로'},
    {'name': 'input_image_topic',               'default': '/camera/color/image_raw/compressed',                                            'description': '입력 컬러 이미지 토픽'},
    {'name': 'output_image_topic',              'default': '/poseEstimation/connector/mask/image_blend/compressed',                         'description': '세그멘테이션 결과(덧씌운) 이미지를 퍼블리시하는 토픽'},
    {'name': 'mask_compressed_image_topic',     'default': '/poseEstimation/connector/mask/image/compressed',                               'description': '마스크 이미지를 퍼블리시하는 토픽'},
    {'name': 'text_prompt',                     'default': 'white plastic connector consist with grey part.',                               'description': 'GroundingDINO에 넣을 텍스트 프롬프트'},
    {'name': 'grounding_dino_config_rel',       'default': 'segment_src/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py',    'description': 'GroundingDINO 설정파일까지의 상대 경로'},
    {'name': 'grounding_dino_checkpoint_rel',   'default': 'segment_src/gdino_checkpoints/groundingdino_swint_ogc.pth',                     'description': 'GroundingDINO 체크포인트까지의 상대 경로'},
    {'name': 'sam2_checkpoint_rel',             'default': 'segment_src/checkpoints/sam2.1_hiera_small.pt',                                 'description': 'SAM2 체크포인트까지의 상대 경로'},
    {'name': 'sam2_model_cfg',                  'default': 'configs/sam2.1/sam2.1_hiera_s.yaml',                                            'description': 'SAM2 모델 설정파일 경로'},
    {'name': 'box_threshold',                   'default': '0.35',                                                                          'description': 'Box threshold'},
    {'name': 'text_threshold',                  'default': '0.25',                                                                          'description': 'Text threshold'},
    {'name': 'prompt_type_for_video',           'default': 'box',                                                                           'description': '비디오 입력 시 사용할 Prompt 타입 (e.g. box, points ...)'},
    {'name': 'output_compression_format',       'default': 'jpeg',                                                                          'description': '출력 이미지를 압축할 포맷'}
]


def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def launch_setup(context, *args, **kwargs):
    param_values = set_configurable_parameters(configurable_parameters)

    ws_path = param_values['ws_path'].perform(context)
    input_image_topic = param_values['input_image_topic'].perform(context)
    output_image_topic = param_values['output_image_topic'].perform(context)
    mask_compressed_image_topic = param_values['mask_compressed_image_topic'].perform(context)
    text_prompt = param_values['text_prompt'].perform(context)
    grounding_dino_config_rel = param_values['grounding_dino_config_rel'].perform(context)
    grounding_dino_checkpoint_rel = param_values['grounding_dino_checkpoint_rel'].perform(context)
    sam2_checkpoint_rel = param_values['sam2_checkpoint_rel'].perform(context)
    sam2_model_cfg = param_values['sam2_model_cfg'].perform(context)
    box_threshold = float(param_values['box_threshold'].perform(context))
    text_threshold = float(param_values['text_threshold'].perform(context))
    prompt_type_for_video = param_values['prompt_type_for_video'].perform(context)
    output_compression_format = param_values['output_compression_format'].perform(context)

    package_name = 'connector_semantic_segment'
    pkg_share = os.path.join(ws_path, 'src', package_name)
    grounding_dino_config = os.path.join(pkg_share, grounding_dino_config_rel)
    grounding_dino_checkpoint = os.path.join(pkg_share, grounding_dino_checkpoint_rel)
    sam2_checkpoint = os.path.join(pkg_share, sam2_checkpoint_rel)

    # 5) 노드 파라미터 설정
    parameters = [{
        'input_image_topic': input_image_topic,
        'output_image_topic': output_image_topic,
        'mask_compressed_image_topic': mask_compressed_image_topic,
        'text_prompt': text_prompt,
        'grounding_dino_config': grounding_dino_config,
        'grounding_dino_checkpoint': grounding_dino_checkpoint,
        'sam2_checkpoint': sam2_checkpoint,
        'sam2_model_cfg': sam2_model_cfg,
        'box_threshold': box_threshold,
        'text_threshold': text_threshold,
        'prompt_type_for_video': prompt_type_for_video,
        'output_compression_format': output_compression_format
    }]

    # 6) Node 생성
    connector_segmentor_node = Node(
        package='connector_semantic_segment',
        executable='connector_segmentor',
        name='connector_segmentor',
        output='screen',
        parameters=parameters
    )

    return [connector_segmentor_node]
    
def generate_launch_description():
    return LaunchDescription(declare_configurable_parameters(configurable_parameters) + [
        OpaqueFunction(function=launch_setup)
    ])
