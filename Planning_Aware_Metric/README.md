# Planning Aware Metric

Planning-aware risk identification evaluation takes place in CARLA simulaotr ( 0.9.14 ). We provide the materials (vehicle's control and random seed) to reproduce all testing scenarios.

# Environment setup

1. Go To [link](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EviA5ovlh6hPo_ZXEPQjxAQB2R3vNubk3HM1u4ib1VdPFA?e=WHEWdm)
2. Download the **DATA_FOR_Planning_Aware_Metric** folder 
3. unzip CARLA_0.9.14_instalce_id.zip
4. unzip data_collection.zip under the folder **Planning_Aware_Metric**
5. unzip weights.zip under the folder **Planning_Aware_Metric/models/**
6. `mv Planning_Aware_Metric /PATH/TO/CARLA_0.9.14_instance_id/PythonAPI/`
7. create conda environment follow below infos.

```bash
# create conda environment 
conda create -n carla_14 python=3.7
conda activate carla_14

# instal pytorch 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install pytorch lighting 
pip install pytorch-lightning==0.8.5
pip uninstall setuptools
pip install setuptools==59.5.0

# install opencv
pip install opencv-python

# others
pip install timm
pip install wandb
pip install imgaug
pip install pandas
pip install pygame
pip install attrdict
pip install ujson

# cd PythonAPI/carla/dist
pip install PythonAPI/carla/dist/carla-0.9.14-cp37-cp37m-linux_x86_64.whl

# For QCNet work on 3090, it may fail on RTX40 series 

```
## Run Planning Aware metric

```bash
# simplely run run_inference.sh
bash run_inference.sh

# Choose from the following options:

#  1 - interactive
#  2 - obstacle
#  3 - obstacle region

# Enter scenario type: 

# Input the method id you want to process
# Choose from the following options:

#   1 - Full Observation
#   2 - Ground Truth
#   3 - Random
#   4 - Range
#   5 - KalmanFilter
#   6 - Social-GAN
#   7 - MANTRA
#   8 - QCNet
#   9 - DSA
#  10 - RRL
#  11 - BP
#  12 - BCP
#  13 - AUTO
#  14 - BCP Smoothing
#  15 - RRL Smoothing
#  16 - DSA Smoothing
#  17 - BP Smoothing
 
# Enter ID to run Planning-aware Evaluation Benchmark: 


```
#### Note:
To reproduce the resul.
For testing each method, it 
- require 4-6 hours for interactive scenario
- require 3-5 hours for obstacle scneario
 
### Create a new Basic Scenario
