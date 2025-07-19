# Hyundai-NGV : Ground and Track Arbitarily Vehicle Connector

**[KAIST-IRIS Lab / Autnomous Platform Team](https://iris.kaist.ac.kr)**

Sangmin Lee, Ph.D IRiS Lab / iismn@kaist.ac.kr  
Handong Lee, M.S. IRiS Lab / hdong564@kaist.ac.kr

## Basic Highlights

Grounded-SAM 2 based Connector Semantic-Segmentation Module


 Grounded SAM 2 is a foundation model pipeline towards grounding and track anything in Videos with [Grounding DINO](https://arxiv.org/abs/2303.05499), [Grounding DINO 1.5](https://arxiv.org/abs/2405.10300), [Florence-2](https://arxiv.org/abs/2311.06242), [DINO-X](https://arxiv.org/abs/2411.14347) and [SAM 2](https://arxiv.org/abs/2408.00714).


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
#### 1. ROS2 Source Environment Setup
First, SAM2 / Grounding DINO need to be installed. Docker already provides the environment. But for stability, the installation process needs to be conducted once again.

```bash
# cd /$(ROS_WS)/src/connector_semantic_segment/segment_src/
pip install -e .
pip install --no-build-isolation -e grounding_dino
```

#### 2. ROS2 Source Build
Build provided source

```bash
# cd /$(ROS_WS)/
colcon build
```

#### 3. Execute Launch File 
Build provided source

```bash
ros2 launch connector_semantic_segment semantic_segmentor_launch.py
```
#### Tip. Edit Parameters
You can edit paramters setting for detecting connector and text prompt in *_launch.py  
```pkg_share``` is hard coded. Need to be specified where your ROS2 package location.

  
> ```semantic_segmentor_launch.py``` Parameters
```python
pkg_share = '/home/iismn/Workspace_B/Hyundai_NGV/pose_Estimation/ROS2/src/connector_semantic_segment'

grounding_dino_config = os.path.join(pkg_share, 'src', 'grounding_dino', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
grounding_dino_checkpoint = os.path.join(pkg_share, 'src', 'gdino_checkpoints', 'groundingdino_swint_ogc.pth')
sam2_checkpoint = os.path.join(pkg_share, 'src', 'checkpoints', 'sam2.1_hiera_small.pt')


'input_image_topic': '/camera/color/image_raw/compressed',
'output_image_topic': '/output_image/compressed',
'text_prompt': 'black plastic connector consist with red part.',
'grounding_dino_config': grounding_dino_config, 
'grounding_dino_checkpoint': grounding_dino_checkpoint,
'sam2_checkpoint': sam2_checkpoint,
'sam2_model_cfg': 'configs/sam2.1/sam2.1_hiera_s.yaml',
'box_threshold': 0.35,
'text_threshold': 0.25,
'prompt_type_for_video': 'box',
'output_compression_format': 'jpeg',
```




## Citation

Check research about grounded segmentation algorithm and pose estimation

```BibTex
@misc{ravi2024sam2segmentimages,
      title={SAM 2: Segment Anything in Images and Videos}, 
      author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
      year={2024},
      eprint={2408.00714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00714}, 
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@misc{ren2024grounding,
      title={Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection}, 
      author={Tianhe Ren and Qing Jiang and Shilong Liu and Zhaoyang Zeng and Wenlong Liu and Han Gao and Hongjie Huang and Zhengyu Ma and Xiaoke Jiang and Yihao Chen and Yuda Xiong and Hao Zhang and Feng Li and Peijun Tang and Kent Yu and Lei Zhang},
      year={2024},
      eprint={2405.10300},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{ren2024grounded,
      title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks}, 
      author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
      year={2024},
      eprint={2401.14159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@misc{jiang2024trex2,
      title={T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy}, 
      author={Qing Jiang and Feng Li and Zhaoyang Zeng and Tianhe Ren and Shilong Liu and Lei Zhang},
      year={2024},
      eprint={2403.14610},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```