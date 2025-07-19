# Hyundai-NGV : Ground and Track Arbitarily Vehicle Connector

**[KAIST-IRIS Lab / Autnomous Platform Team](https://iris.kaist.ac.kr)**

Sangmin Lee, Ph.D IRiS Lab / iismn@kaist.ac.kr  
Handong Lee, M.S. IRiS Lab / hdong564@kaist.ac.kr

## Basic Highlights

Multiple DPT Sensor Management (Intel® RealSense L515)


## Installation

#### 1. Pull Docker Environmetn Image. Docker image contain all of neccesary package [Ubuntu, CUDA, ROS2, PyTorch, ...]

```bash
docker pull iismn/env_hyundai_ngv:GrndSAM-2
```
#### 2. Generate docker container by
```bash
sudo docker run -it \   
    --name=connectorSegmentor \
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
    iismn/env_hyundai_ngv:GrndSAM-2
```
#### 3. Excute docker container 
```bash
sudo docker start connectorSegmentor && sudo docker exec -it connectorSegmentor /bin/zsh
```


## ROS2 Execution
### **Pre-Requirement -**<span style="color:yellow"> ***Intel® RealSense SDK NEED TO BE INSTALLED*** </span>  
SDK / ROS Version Requirement
>**Intel® RealSense SDK - 2.54.1**  
**Intel® RealSense ROS Branch - 4.54.1**

#### 1. ROS2 Source Build
Build provided source

```bash
# cd /$(ROS_WS)/
colcon build
```

#### 3. Execute Launch File 
Build provided source

```bash
ros2 launch realsense_manage rs_multi_camera_mission_launch.py
```
#### Tip. Edit Parameters
You can edit paramters setting for detecting connector and text prompt in *_launch.py.  
Please set **RealSense serial_no** correspond to *UR16 Manipulator*.

  
> ```rs_multi_camera_mission_launch.py``` Parameters
```python
parameters = [{
      'camera1_launch_args': ['camera_name:=L515_A', 'serial_no:=f1422940', 'output:=log'],
      'camera2_launch_args': ['camera_name:=L515_B', 'serial_no:=f1422241', 'output:=log']
}]
```

