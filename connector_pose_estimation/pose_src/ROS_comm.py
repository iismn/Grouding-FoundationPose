#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import numpy as np
import trimesh
import cv2
import imageio
from pose_src.estimater import *  
from pose_src.datareader import *
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import logging
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as R, Slerp
from collections import deque

# LPF XYZ
class LowPassFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.filtered = None

    def filter(self, value):
        if self.filtered is None:
            self.filtered = value
        else:
            self.filtered = self.alpha * value + (1 - self.alpha) * self.filtered
        return self.filtered

# SLERP Quaternion
class QuaternionFilter:
    def __init__(self, alpha=0.1):
        """
        SLERP 기반의 쿼터니언 필터링
        
        :param alpha: 필터링 강도 (0 < alpha < 1). 낮을수록 이전 값에 더 많은 가중치를 부여.
        """
        self.alpha = alpha
        self.filtered = None

    def filter(self, quat):
        """
        입력된 쿼터니언을 필터링
        
        :param quat: 현재 쿼터니언 (순서: [x, y, z, w])
        :return: 필터링된 쿼터니언
        """
        if self.filtered is None:
            self.filtered = quat
            return self.filtered
        
        key_times = [0, 1]
        key_rots = R.from_quat([self.filtered, quat])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([self.alpha])[0]
        self.filtered = interp_rot.as_quat()
        return self.filtered
    
    
def set_logging_format(disable_logging=False):
    if disable_logging:
        logging.disable(logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def set_seed(seed):
    np.random.seed(seed)

class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator_node')
        self.bridge = CvBridge()

        # Declare parameters with default values
        self.declare_parameter('mesh_file', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'demo_data/mustard0/mesh/textured_simple.obj'))
        self.declare_parameter('est_refine_iter', 5)
        self.declare_parameter('track_refine_iter', 2)
        self.declare_parameter('debug', 1)
        self.declare_parameter('debug_dir', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'debug'))
        self.declare_parameter('disable_logging', False)

        self.declare_parameter('color_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw/compressedDepth')
        self.declare_parameter('mask_topic', '/camera/mask/image_raw/compressedMask')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('pose_pub_topic', '/estimated_pose')
        self.declare_parameter('vis_pub_topic', '/pose_vis')

        self.declare_parameter('lpf_pose', 0.5)
        self.declare_parameter('lpf_slerp', 0.5)
        
        # Get parameter values
        mesh_file = self.get_parameter('mesh_file').get_parameter_value().string_value
        est_refine_iter = self.get_parameter('est_refine_iter').get_parameter_value().integer_value
        track_refine_iter = self.get_parameter('track_refine_iter').get_parameter_value().integer_value
        debug = self.get_parameter('debug').get_parameter_value().integer_value
        debug_dir = self.get_parameter('debug_dir').get_parameter_value().string_value
        disable_logging = self.get_parameter('disable_logging').get_parameter_value().bool_value

        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        pose_pub_topic = self.get_parameter('pose_pub_topic').get_parameter_value().string_value
        vis_pub_topic = self.get_parameter('vis_pub_topic').get_parameter_value().string_value
        
        lpf_pose = self.get_parameter('lpf_pose').get_parameter_value().double_value
        lpf_slerp = self.get_parameter('lpf_slerp').get_parameter_value().double_value
        # Initialize logging and seed
        self.get_logger().warn('[POSE ESTIMATOR] ROS2 Connector Pose Node 시작')
        set_logging_format(disable_logging=disable_logging)
        set_seed(0)

        # Load the mesh
        mesh = trimesh.load(mesh_file)

        self.debug = debug
        self.debug_dir = debug_dir
        os.system(f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # Initialize predictors and estimator
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx
        )

        # Initialize publisher with parameterized topic name
        self.pose_pub = self.create_publisher(PoseStamped, pose_pub_topic, 10)
        self.vis_pub = self.create_publisher(CompressedImage, vis_pub_topic, 10)
        # Create synchronized subscribers (color, depth, camera_info)
        self.color_sub = Subscriber(self, CompressedImage, color_topic)
        self.depth_sub = Subscriber(self, CompressedImage, depth_topic)
        self.info_sub = Subscriber(self, CameraInfo, camera_info_topic)

        # Synchronize the color, depth, and camera_info subscribers
        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        # Create separate subscriber for mask
        self.mask_subscriber = self.create_subscription(
            CompressedImage,
            mask_topic,
            self.mask_callback,
            10
        )
        self.mask = None  # Initialize mask
        self.frame_count = 0  # To track number of frames

        # Initialize pose
        self.initialized = False
        self.to_origin = to_origin
        self.bbox = self.bbox
        self.get_logger().info('[POSE ESTIMATOR] MeshFile: ' + mesh_file)
        self.get_logger().info('[POSE ESTIMATOR] Initialize 완료')

        self.position_filter = LowPassFilter(lpf_pose)
        self.orientation_filter = QuaternionFilter(lpf_slerp)

    def mask_callback(self, mask_msg):
        try:
            # Convert ROS CompressedImage message to OpenCV image
            mask = self.bridge.compressed_imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
            mask = (mask * 255).astype(np.uint8)
            self.mask = mask
            self.get_logger().debug('[POSE ESTIMATOR] Mask 업데이트됨')
        except Exception as e:
            self.get_logger().error(f"[POSE ESTIMATOR] Mask 콜백 에러: {e}")

    def compressed_depthmsg_to_numpy(self, msg):
        depth_fmt, compr_type = msg.format.split(';')
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()
        
        if compr_type != "compressedDepth":
                self.get_logger().error('[POSE ESTIMATOR] 지원하지않는 Compression 타입')
                
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]

        depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth_img_raw is None:
            self.get_logger().error('[POSE ESTIMATOR] Depth Header 크기 재설정 필요: Decode 실패')

        return depth_img_raw
    
    
    def callback(self, color_msg, depth_msg, info_msg):
        try:
            # Increment frame count
            self.frame_count += 1

            # Ensure mask has been received
            if self.mask is None:
                self.get_logger().warn('[POSE ESTIMATOR] Mask 데이터 없음')
                return

            # Wait until a certain number of frames have been processed
            if not self.initialized and self.frame_count < 10:
                return
            
            # Convert ROS CompressedImage messages to OpenCV images
            color = self.bridge.compressed_imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            depth = self.compressed_depthmsg_to_numpy(depth_msg)
            depth = depth.astype(np.float32)
            depth = depth/1000
            # print('max depth: ',np.max(depth))
            mask = self.mask  # Use the latest mask

            # Extract camera intrinsic matrix K
            K = np.array(info_msg.k).reshape(3, 3)

            if not self.initialized:
                # First frame after mask is ready: perform registration
                self.get_logger().info('[POSE ESTIMATOR] Foundation Pose 시작')
                pose = self.est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=self.get_parameter('est_refine_iter').get_parameter_value().integer_value)
                self.initialized = True

                if self.debug >= 3:
                    m = self.est.mesh.copy()
                    m.apply_transform(pose)
                    m.export(os.path.join(self.debug_dir, 'model_tf.obj'))
                    xyz_map = depth2xyzmap(depth, K)
                    valid = depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(os.path.join(self.debug_dir, 'scene_complete.ply'), pcd)
            else:
                # Subsequent frames: perform tracking
                pose = self.est.track_one(rgb=color, depth=depth, K=K, iteration=self.get_parameter('track_refine_iter').get_parameter_value().integer_value)

            # Extract position and orientation from pose matrix
            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

            # Apply filters
            filtered_position = self.position_filter.filter(position)
            filtered_quat = self.orientation_filter.filter(quat)

            # Reconstruct pose matrix from filtered position and orientation
            filtered_rot = R.from_quat(filtered_quat).as_matrix()
            filtered_pose = np.eye(4)
            filtered_pose[:3, :3] = filtered_rot
            filtered_pose[:3, 3] = filtered_position

            pose = filtered_pose
            
            if self.debug >= 1:
                center_pose = pose @ np.linalg.inv(self.to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                try:
                    vis_msg = self.bridge.cv2_to_compressed_imgmsg(vis[..., ::-1], dst_format='jpeg')
                    self.vis_pub.publish(vis_msg)
                except Exception as e:
                    self.get_logger().error(f"[POSE ESTIMATOR] 메시지 퍼블리시 에러: {e}")

            if self.debug >= 2:
                os.makedirs(os.path.join(self.debug_dir, 'track_vis'), exist_ok=True)
                track_vis_path = os.path.join(self.debug_dir, 'track_vis', f"{color_msg.header.stamp.sec}_{color_msg.header.stamp.nanosec}.png")
                imageio.imwrite(track_vis_path, vis)

            # Publish the pose
            pose_msg = PoseStamped()
            pose_msg.header = color_msg.header  # Use the same timestamp and frame as the input
            pose_msg.pose = self.transform_to_pose_msg(pose)
            self.pose_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f"[POSE ESTIMATOR] 콜백 함수 에러: {e}")

    def transform_to_pose_msg(self, pose_matrix):
        from tf_transformations import quaternion_from_matrix

        # Extract translation
        translation = pose_matrix[:3, 3]
        # Extract rotation as quaternion
        quaternion = quaternion_from_matrix(pose_matrix)

        # Create Pose message
        pose = PoseStamped().pose
        pose.position.x = translation[0].astype(np.float64)
        pose.position.y = translation[1].astype(np.float64)
        pose.position.z = translation[2].astype(np.float64)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        return pose

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[POSE ESTIMATOR] Pose Estimator 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
