# RiskBench: A Scenario-based Benchmark for Risk Identification
![RiskBench](images/teaser.png)

## System requirements
- Linux (Tested on Ubuntu 18.04)
- Python3 (Tested using Python 3.7)
- PyTorch (Tested using PyTorch 1.10.0)
- CUDA (Tested using CUDA 11.3)
- CARLA (Tested using CARLA 0.9.13)
- GPU ( Tested using Nvidia rtx3090)

## clone 
```bash
git clone --depth 1 https://github.com/HCIS-Lab/RiskBench.git
```
## Installation
### CARLA
Please refer to [CARLA's page](https://carla.readthedocs.io/en/latest/start_quickstart/).

### Baselies
There are multiple baselines in RiskBench. We provide a single installation guide that is compatible with every baselines except for QCNet.
```bash
# create conda environment 
conda create -n carla python=3.7
conda activate carla

# install pytorch 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install packages
pip install -r requirements.txt

```
Note: For QCNet's installation, please refer to QCNet's main page.

## Dataset
The RiskBench dataset can be downloaded [here](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EviA5ovlh6hPo_ZXEPQjxAQB2R3vNubk3HM1u4ib1VdPFA?e=WHEWdm).

Download **RiskBench_Dataset** for the whole dataset. In addition, we provide **DATA_FOR_Planning_Aware_Metric** and **DATASET_for_LBC_Training** for planning aware metric evaluation and LBC training data respectively.

Also, we provide instructions on how to collect basic scenarios and data augmentation. Please refer to [link](Planning_Aware_Metric/).

<!-- Dataset statistics: 

|       | Amount                     | Example       |
| :-----| :--------------------------- | :-----------: |
| Train |                      |    |
| Val   |                 |      |
| Test  |                                         |         | -->

## Risk identification
We provide each baseline's training and inference details which can be found [here](risk_identification/Baselines).

## Citation
```
@misc{RiskBench,
    title={A Scenario-based Benchmark for Risk Identification},
    author={Chi-Hsi Kung and Chieh-Chih Yang and Pang-Yuan Pao and Shu-Wei Lu and Pin-Lun Chen and Hsin-Cheng Lu and Yi-Ting Chen},
    year={2023},
    organization={National Yang Ming Chiao Tung University}
}
```
