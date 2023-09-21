# Planning Aware Metric And Data collection

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
 
## Create a new Basic Scenario
### For interactive, Collision, Obstacle scenario
We need two computers to record the scenario. ( One play ego vehicle, the other one play actor )

### Host 
```bash
# using wheel ex: logitech g29
python monitor_control.py --controller wheel --filter model3 --map Town03 --scenario_type {scenario_type}

# without wheel 
python monitor_control.py --filter model3 --map Town03 --scenario_type {scenario_type}
```

### Client 

```bash
# using wheel ex: logitech g29
python manual_control_steeringwheel.py --host ip_address  --filter *yzf

# without wheel 
python manual_control.py --host ip_address  --filter *yzf
```
> For different sceanrio type, change {scenario_type} to interactive, non-interactive, obstacle, non-interactive
(non-interactive scenario only need to run **host**)

> To change different **Town**:
Replace **Town03** to (Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, A0, A1, A6, B3, B7, B8)

> To change actor object type: please reference to https://carla.readthedocs.io/en/latest/bp_library/

>For interactive and collision scenario, we need to first setup the interactor id. 
Press **"c"** to enter ID
( Host terminal: NPC id ? ) 
( Id will be display when you run client instruction be executed in Client terminal. )

---
## How to record
- Press **"r"** to start record the scenario
- Press **"r"** again to stop record the scenario and enter the scenario tags in the terminal to name this scenario ( map_id, road_id, is_traffic_light, actor_type, actor_action, name_my_action, is_interactive, name_violated_rule )


## How to check if the recording file is ok to use ?

```
bash run_test_scenario.sh
```

>Input the scenario_type you want to process
Choose from the following options:
1 - interactive
2 - non-interactive
3 - obstacle
4 - collision

The bash file will generate a video inside the basic scenario id folder 
-->  We can manually check the quality via video.

After creating a basic scenario, we can generate** random seeds** and set differet **weather** by simpely run **bash run_generate_scenario.sh**

It will generate {weather}_{randon_actor} inside the folder `/path/to/data_collection/{scenario_type}/{Baisc scenario ID}/variant_scenario/{weather}_{random_actor}_`

Finally, `cd path/to/data_collection` and `python get_remove_list -s {scenario_type}`. And manually remove the fail folders.

## Labeling start position and end position for collecting the data
We provideo a tool to label the start position and end position 

```
cd path/to/vis_tools/

# instal qt environment
conda create -n label python=3.7
conda activate label

# install package
conda  install pyqt
pip install opencv-python-headless
pip install scipy
pip install six

# run 
python  start.py
```

## Collect the data

```
python get_name_list.py -s interactive > name.txt
bash run_data_collection.sh
```

Data will be save inside **path/to/data_collection/{scenario_type}/{basic_id}/variant_scenario/{weather}_{random_actor}/**
