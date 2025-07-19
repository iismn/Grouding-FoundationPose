import os
import cv2
import torch
import numpy as np
import supervision as sv
import torchvision.transforms as T
from torchvision.ops import box_convert
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage  # Image 제거
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node_compressed')

        # 파라미터 설정
        self.declare_parameter('input_image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('output_image_topic', '/output_image/compressed')
        self.declare_parameter('mask_compressed_image_topic', '/output_mask/compressed_image')
        self.declare_parameter('text_prompt', 'black plastic connector consist with red part.')
        self.declare_parameter('grounding_dino_config', "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.declare_parameter('grounding_dino_checkpoint', "gdino_checkpoints/groundingdino_swint_ogc.pth")
        self.declare_parameter('sam2_checkpoint', "./checkpoints/sam2.1_hiera_small.pt")
        self.declare_parameter('sam2_model_cfg', "configs/sam2.1/sam2.1_hiera_s.yaml")
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('prompt_type_for_video', "box")
        self.declare_parameter('output_compression_format', 'jpeg')

        # 파라미터 가져오기
        self.input_image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        self.output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        self.mask_compressed_image_topic = self.get_parameter('mask_compressed_image_topic').get_parameter_value().string_value  # 마스크 토픽 가져오기
        self.TEXT_PROMPT = self.get_parameter('text_prompt').get_parameter_value().string_value
        GROUNDING_DINO_CONFIG = self.get_parameter('grounding_dino_config').get_parameter_value().string_value
        GROUNDING_DINO_CHECKPOINT = self.get_parameter('grounding_dino_checkpoint').get_parameter_value().string_value
        sam2_checkpoint = self.get_parameter('sam2_checkpoint').get_parameter_value().string_value
        model_cfg = self.get_parameter('sam2_model_cfg').get_parameter_value().string_value
        BOX_THRESHOLD = self.get_parameter('box_threshold').get_parameter_value().double_value
        TEXT_THRESHOLD = self.get_parameter('text_threshold').get_parameter_value().double_value
        PROMPT_TYPE_FOR_VIDEO = self.get_parameter('prompt_type_for_video').get_parameter_value().string_value
        self.OUTPUT_COMPRESSION_FORMAT = self.get_parameter('output_compression_format').get_parameter_value().string_value.lower()

        # 지원되는 압축 형식 확인
        if self.OUTPUT_COMPRESSION_FORMAT not in ['jpeg', 'png']:
            self.get_logger().warn(f"[POSE ESTIMATOR] 지원되지 않는 포멧 '{self.OUTPUT_COMPRESSION_FORMAT}'. 기본값-'jpeg'.")
            self.OUTPUT_COMPRESSION_FORMAT = 'jpeg'

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().warn('[POSE ESTIMATOR] ROS2 Connector Segmentor 노드 시작')

        # Grounding DINO 모델 로드
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.device
        )
        self.get_logger().info('[POSE ESTIMATOR] Grounding DINO 모델 로드')

        # SAM2 모델 초기화
        self.camera_predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)
        self.get_logger().info('[POSE ESTIMATOR] SAM2 모델 초기화')

        # 비디오 예측 상태 초기화
        self.inference_state = None
        self.initialized = False 
        
        # ROS2 메시지 처리
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage,
            self.input_image_topic,
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(CompressedImage, self.output_image_topic, 10)
        self.mask_compressed_publisher = self.create_publisher(CompressedImage, self.mask_compressed_image_topic, 10)
        self.OBJECTS = []
        self.PROMPT_TYPE_FOR_VIDEO = PROMPT_TYPE_FOR_VIDEO
        self.BOX_THRESHOLD = BOX_THRESHOLD
        self.TEXT_THRESHOLD = TEXT_THRESHOLD

        # 이미지 변환 - Transform Torch Tensor
        self.transform = T.Compose([
            T.Resize(256),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def image_callback(self, msg):
        try:
            # CompressedImage 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'[POSE ESTIMATOR] CvBridge 에러: {e}')
            return
        
        width, height = cv_image.shape[:2][::-1]
        
        # 첫 번째 프레임일 경우 초기화
        if not self.initialized:
            self.get_logger().info('[POSE ESTIMATOR] Initialize 시작')

            # 이미지 소스 및 PIL 이미지로 변환
            image_source = cv_image.copy()
            image_pil = Image.fromarray(cv_image[:, :, ::-1])  # BGR to RGB

            # 이미지 변환 적용
            image_transformed = self.transform(image_pil).to(self.device)

            # Grounding DINO 예측
            self.get_logger().info('[POSE ESTIMATOR] TEXT PROMPT: ' + self.TEXT_PROMPT)
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image_transformed,  # torch.Tensor 전달
                caption=self.TEXT_PROMPT,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
            )

            if len(boxes) == 0:
                self.get_logger().warn('[POSE ESTIMATOR] Grounding DINO 결과 없음')
                return

            # 박스 포맷 변환
            h, w, _ = image_source.shape
            if isinstance(boxes, np.ndarray):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            elif isinstance(boxes, list):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            elif not isinstance(boxes, torch.Tensor):
                self.get_logger().error('[POSE ESTIMATOR] BOX타입 output 에러.')
                return

            # 디바이스로 이동 및 스케일링
            boxes = boxes.to(self.device) * torch.tensor([w, h, w, h], device=self.device)
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            confidences = confidences.cpu().numpy().tolist()
            self.OBJECTS = labels
            self.image_predictor.set_image(image_source)

            # 마스크 예측
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            if masks.ndim == 4:
                masks = masks.squeeze(1)

            # 비디오 예측 상태 초기화
            # self.inference_state = self.camera_predictor.init_state(image_source=image_source)
            self.camera_predictor.load_first_frame(image_source)
            
            # 객체별로 포인트 또는 박스 추가
            if self.PROMPT_TYPE_FOR_VIDEO == "point":
                self.get_logger().info('[POSE ESTIMATOR] 프롬프트 output 타입 : 포인트')
                all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
                for object_id, points in enumerate(all_sample_points, start=1):
                    labels = np.ones((points.shape[0]), dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = self.camera_predictor.add_new_points_or_box(
                        frame_idx=0,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )
            elif self.PROMPT_TYPE_FOR_VIDEO == "box":
                self.get_logger().info('[POSE ESTIMATOR] 프롬프트 output 타입 : 박스')
                for object_id, box in enumerate(input_boxes, start=1):
                    _, out_obj_ids, out_mask_logits = self.camera_predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=object_id,
                        bbox=box
                    )
                
            elif self.PROMPT_TYPE_FOR_VIDEO == "mask":
                self.get_logger().info('[POSE ESTIMATOR] 프롬프트 output 타입 : 마스크')
                for object_id, mask in enumerate(masks, start=1):
                    labels = np.ones((1), dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = self.camera_predictor.add_new_mask(
                        frame_idx=0,
                        obj_id=object_id,
                        mask=mask
                    )
            else:
                self.get_logger().error("[POSE ESTIMATOR] 프롬프트 output 타입 미지원")
                return

            self.initialized = True
            self.get_logger().info('[POSE ESTIMATOR] Initialize 완료')
        else:
            # 이후 프레임 처리
            # self.get_logger().info('이후 프레임 처리')
            # start_time = time.perf_counter()
            image_source = cv_image.copy()
            
            out_obj_ids, out_mask_logits = self.camera_predictor.track(image_source)
            start_time = time.perf_counter()

            video_segments = [(out_obj_ids, out_mask_logits)]
            
            for out_obj_ids, out_mask_logits in video_segments:
                # Segmentation 결과 적용
                segments = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

                # 이미지 복사
                # self.get_logger().info('이미지 복사')
                annotated_img = cv_image.copy()

                # 마스크를 시각화
                # self.get_logger().info('마스크를 시각화')
                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                # print(all_mask.shape)
                for i in range(0, len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                        np.uint8
                    ) * 255

                    all_mask = cv2.bitwise_or(all_mask, out_mask)

                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                image_source = cv2.addWeighted(image_source, 1, all_mask, 0.5, 0)
                annotated_img = image_source.copy()

                try:
                    mask_gray = cv2.cvtColor(all_mask, cv2.COLOR_RGB2GRAY)
                    mask_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(mask_gray, dst_format=self.OUTPUT_COMPRESSION_FORMAT)
                    self.mask_compressed_publisher.publish(mask_compressed_msg) 
                except CvBridgeError as e:
                    self.get_logger().error(f'[POSE ESTIMATOR] CvBridge 에러: {e}')
                    return

                try:
                    out_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img, dst_format=self.OUTPUT_COMPRESSION_FORMAT)
                except CvBridgeError as e:
                    self.get_logger().error(f'[POSE ESTIMATOR] CvBridge 에러: {e}')
                    return

                self.publisher.publish(out_msg)
                break

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    except Exception as e:
        node.get_logger().error(f'Exception in node: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
