# RiskBench: A Scenario-based Benchmark for Risk Identification
![RiskBench](images/teaser.png)

## System requirements
- Linux (Tested on Ubuntu 18.04)
- Python3 (Tested using Python 3.7)
- PyTorch (Tested using PyTorch 1.10.0)
- CUDA (Tested using CUDA 11.3)
- CARLA (Tested using CARLA 0.9.13)
- GPU ( Tested using Nvidia rtx3090)

## Installation
1. Download CARLA from https://drive.google.com/file/d/1bQE3H4mh2WBSGK8tZ4UzTfogce-5HcEJ/view?usp=share_link
2. Unzip CARLA.zip
3. install the conda environment following below steps

```bash
# create conda environment 
conda create -n carla python=3.7
conda activate carla

# install pytorch 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install packages
pip install -r requirements.txt

```



## Dataset
We currently provide sample data for testing. The full dataset will be released soon.

Download link : [google drive](https://drive.google.com/drive/folders/1P_cMksFicHEYPuC0Nb2MYt0SbfBi7vz8?usp=share_link) <br />

Also, we provide instructions on how to collect basic scenarios and data augmentation. Please refer to [dataset](dataset/).

## Code execution

### Risk Identification (optional)
We provided risk identification results preserved in json format for each algorithm. Alternatively, one can generate by following the instruction in [models](risk_identification/models). We will release a intergrated API soon!

### Offline Risk Identification Evaluation
We perform offline risk identification evaluation (with metric F-1 score and PIC) by taking input as preserved risk identification prediction:
```
python risk_identification.py --path {PREDICTION_PATH} --model {MODEL} --metric {METRIC} --scenario {ATTRIBUTE}
```

Arguments: 

| Parameter     | Description                                          |  Example   |
| :------------ | :--------------------------------------------------- | :--------: |
| --path        | path of the stored prediction .json file             | prediction |
| --model       | name of the risk identification method               | two-stage  |
| --metric      | risk identification metric                           |     F1     |
| --scenario    | scenario filter, default is ""                       |    Rain    |
| --save_result | save result to {PREDICTION_PATH}/result/{MODEL}.json |    None    |


### Planning-aware Risk Identification
![planning aware ](images/planning_aware.gif)
Planning-aware risk identification evaluation takes place in CARLA simulaotr. We provide the materials (vehicle's control and random seed) to reproduce all testing scenarios.

step 1: move the recorded scenarios to the CARLA folder
```
cd planning_aware_eval/
mv -r interactive/ path_to_carla/PythonAPI/
mv -r obstacle/ path_to_carla/PythonAPI/
```

setp 2: execute planning-aware risk identification evaluation
```
# For interactive scenario
bash test_interactive.sh

# For obstacle scenario
bash test_obstacle.sh
```
User can select the algorithm to be evaluated in the process.

The evaluation for each algorithm take about 4 hours.

The testing results will be save to ./results/ 

To obtain the Influenced ratio and Collision rate, run the command:
```
python data_analysis.py
```
