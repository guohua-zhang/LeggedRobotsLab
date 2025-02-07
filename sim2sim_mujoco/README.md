<h1 align="center"> Sim2sim </h1>

- author: guohua zhang
- email: 12433015@mail.sustech.edu.cn
- data: 2024.10.31

<p align="center">
<img src="./images/long_stair_walk.gif" width="90%"/>
</p>

<p align="center">
<img src="./images/terrain_walk.gif" width="90%"/>
</p>

# 1. Environment Setup

```
conda create -n locomotion python=3.8
conda acticate locomotion
# pytorch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# isaacgym
Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .
# this project
cd MLM
cd rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .
cd MLM && pip install -r requirements.txt
cd deployment && pip install -r requirements.txt
```

# 2. Run

1. Train a policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python train.py`

2. Play and export the latest policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python play.py`

3. sim2sim:

  - `cd deployment/scripts`
  - `python sim2sim_go1.py`

![Figure_1](./images/Figure_1.png)

4. terrain_generator

  - `cd deployment/sim_terrain/`
  - `python terrain_generator.py`

<video src="/home/zgh/ws/rl/robotic11_locomotion/MLM/images/terrain_walk.mp4"></video>



# 👏 Acknowledgements

- [legged_gym](https://github.com/leggedrobotics/legged_gym): Our codebase is built upon legged_gym.

- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco): Our terrain_generator is built upon unitree_mujoco



# Tools

## 1. urdf可视化工具

网址：urdf.robotsfan.com

查看层级结构：urdf_to_graphviz

```
sudo apt install liburdfdom-tools
urdf_to_graphviz xxx.urdf
```

## 2. urdf2mjcf转换工具

```
pip install mujoco 
git clone https://github.com/FFTAI/Wiki-MJCF.git
pip install -e . 
urdf2mjcf /path/to/models /path/to/mjcf
```

## 3. mjcf可视化工具

pinocchio，crocoddyl可视化代码 MJCF

```
pip install mujoco
git clone https://github.com/FFTAI/Wiki-MJCF.git
pip install -e . 
python -m mujoco.viewer --mjcf scene.xml
```
