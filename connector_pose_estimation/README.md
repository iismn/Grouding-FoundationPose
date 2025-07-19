# Hyundai-NGV : 6-DoF Pose Connector Pose Estimation 

**[KAIST-IRIS Lab / Autnomous Platform Team](https://iris.kaist.ac.kr)**

Sangmin Lee, Ph.D IRiS Lab / iismn@kaist.ac.kr  
Handong Lee, M.S. IRiS Lab / hdong564@kaist.ac.kr

## Basic Highlights

FoundationPose based Connector 6-DoF Pose Estimation


  FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. [FoundationPose](https://arxiv.org/abs/2312.08344)


## Installation

#### 1. Pull Docker Environmetn Image. Docker image contain all of neccesary package [Ubuntu, CUDA, ROS2, PyTorch, ...]

```bash
docker pull iismn/env_hyundai_ngv:FndPose
```
#### 2. Generate docker container by
```bash
sudo docker run -it \
    --name=connectorPose \
    --gpus=all \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --volume="$XAUTH:$XAUTH" \
    --runtime=nvidia \
    -v /home/$USER/Workspace/:/home/$USER/Workspace/ \
    -v /dev:/dev\
    -v /dev/shm:/dev/shm\
    --privileged\
    --net=host \
    --ipc=host \
    --pid=host \
    iismn/env_hyundai_ngv:FndPose
```
#### 3. Excute docker container 
```bash
sudo docker start connectorPose && sudo docker exec -it connectorPose /bin/zsh
```
#### 4. Install Pytorch 3D
Inside container, PyTorch 3D need to be install in local space. Step need several time to finalize install. (~2m)
```bash
git clone https://github.com/facebookresearch/pytorch3d.git && pip install -e .
```

## ROS2 Execution
#### 1. ROS2 Source Environment Setup
First, FoundationPose need to be installed. Docker already provides the environment. But for stability, the installation process needs to be conducted once again.

```bash
# cd /$(ROS_WS)/src/connector_pose_estimation/pose_src/
export PYTHONPATH=$(pwd)
export PYTHONPATH=$ROS_WS/src/connector_pose_estimation/pose_src/
```

#### 2. ROS2 Source Build
Build provided source

```bash
# cd /$(ROS_WS)/
colcon build && source install/setup.zsh
```

#### 3. Execute Launch File 
Build provided source

```bash
ros2 launch connector_pose_estimation pose_estimator_launch.py
```
#### Tip. Edit Parameters
You can edit paramters setting for detecting connector and text prompt in *_launch.py  
```pkg_share``` is hard coded. Need to be specified where your ROS2 package location.

*!! important !!*  
*Mesh (Obj) File need to be specified. If not, pose estimation can not be run.*

> ```pose_estimator_launch.py``` Parameters
```python
# IMPORTANT: Package Directory is /$(ROS_WS)/src/connector_pose_estimation/pose_src/
# IMPORTANT: Mesh file need to be contained under /$(ROS_WS)/src/connector_pose_estimation/mesh/*.obj

package_dir = 'source /home/iris/Workspace/ROS2/src/connector_pose_estimation'

mesh_file = os.path.join(package_dir, 'mesh', 'textured_simple.obj')
debug_dir = os.path.join(package_dir, 'debug')

{'mesh_file': mesh_file},
{'est_refine_iter': 5}, # Fast for 1, Accurate for 5
{'track_refine_iter': 2},
{'debug': 1}, # For visualize with cv2 window live and raw data save to debug folder
{'debug_dir': debug_dir},
{'disable_logging': True}, # Detailed log information print option in terminal
{'color_topic': '/camera/color/image_raw'}, # Topic from Intel Real-Sense L515
{'depth_topic': '/camera/depth/image_raw'}, # Topic from Intel Real-Sense L515
{'mask_topic': '/camera/mask/image_raw'}, # Topic from connectorSegmentor
{'camera_info_topic': '/camera/color/camera_info'}, # Topic from Intel Real-Sense L515
{'pose_pub_topic': '/estimated_pose'}, # Topic of Output Pose
```

## License
The code and data are released under the NVIDIA Source Code License. Copyright © 2024, NVIDIA Corporation. All rights reserved.


## Citation

Check research about grounded segmentation algorithm and pose estimation

```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```