![header](https://capsule-render.vercel.app/api?type=rect&color=timeGradient&text=GROUNDING%20FOUNDATION%20POSE&fontSize=20)

## <div align=left>REPO INFO</div>  
- Hyundai Manufacturing Automation Project
- ROS2 VISION BASED CONNECTOR POSE ESTIMATION & SEMANTIC SEGMENTATION  

## <div align=left>REPO CONFIG</div>  
#### CONNECTOR_POSE_ESTIMATION  
* FoundationPose based 6-DOF Pose Estimation for Connectors
* Real-time pose tracking with RGB-D camera input
* Support for multiple connector types (A, B, C, D)
* BundleSDF integration for neural rendering
#### CONNECTOR_SEMANTIC_SEGMENT  
* SAM2 (Segment Anything Model 2) for video segmentation
* GroundingDINO for open-vocabulary object detection
* Real-time semantic segmentation pipeline
* Web-based demo interface with React frontend
#### REALSENSE_MANAGE  
* Multi-camera RealSense management system
* Depth image processing and filtering
* Camera calibration and synchronization
#### REALSENSE-ROS-4.51.1  
* Official Intel RealSense ROS2 wrapper
* Support for D400/D500 series cameras
* Point cloud generation and visualization

## <div align=left>REPO USE</div> 
<pre>cd ros2_ws/src/  
git clone https://github.com/iismn/Grouding-FoundationPose.git  
cd ../.. && colcon build</pre>

## <div align=left>SYSTEM REQUIREMENTS</div>
#### HARDWARE
- Intel RealSense D435i/D455 RGB-D Camera x2+
- NVIDIA GPU (RTX 3080 or higher recommended)
- Ubuntu 20.04 LTS / 22.04 LTS
- ROS2 Humble/Foxy

#### SOFTWARE DEPENDENCIES
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+
- OpenCV 4.5+
- Open3D 0.16+

## <div align=left>ADD INFO</div>
#### VISION SYSTEM COMPONENTS 
- Intel RealSense D435i RGB-D Camera x2
- Intel RealSense D455 RGB-D Camera x1  
- NVIDIA RTX 3080 GPU x1
- Custom Connector Models (A, B, C, D types)
- Real-time Pose Estimation Pipeline
- Semantic Segmentation with SAM2
- GroundingDINO Object Detection
- Web-based Visualization Interface

#### V 1.0.0
- Initial release with FoundationPose integration
- SAM2 + GroundingDINO semantic segmentation
- Multi-camera RealSense support
- Real-time pose estimation for manufacturing connectors

#### V 1.1.0 (Planned)
- Improved pose estimation accuracy
- Additional connector type support
- Enhanced web interface
- Performance optimization for real-time processing

## <div align=left>LAUNCH INSTRUCTIONS</div>
#### Pose Estimation
<pre>ros2 launch connector_pose_estimation pose_estimator_launch.py</pre>

#### Semantic Segmentation  
<pre>ros2 launch connector_semantic_segment semantic_segmentor_launch.py</pre>

#### Multi-Camera Setup
<pre>ros2 launch realsense_manage rs_multi_camera_mission_launch.py</pre>

#### Mission Mode (Full Pipeline)
<pre>ros2 launch connector_pose_estimation mission_launch.py
ros2 launch connector_semantic_segment mission_launch.py</pre>

## <div align=left>ACKNOWLEDGMENTS</div>
- [FoundationPose](https://github.com/NVlabs/FoundationPose) - NVIDIA Research
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Meta AI
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - IDEA Research
- [RealSense ROS](https://github.com/IntelRealSense/realsense-ros) - Intel Corporation
