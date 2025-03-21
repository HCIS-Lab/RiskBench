# [ICRA 2024] RiskBench: A Scenario-based Benchmark for Risk Identification
![RiskBench](images/teaser.png)
<!-- > **RiskBench: A Scenario-based Benchmark for Risk Identification**  
> [Chi-Hsi Kung](https://hankkung.github.io/website/), Chieh-Chih Yang, Pang-Yuan Pao, Shu-Wei Lu, Pin-Lun Chen, Hsin-Cheng Lu, [Yi-Ting Chen](https://sites.google.com/site/yitingchen0524/)
> - [Paper](https://hcis-lab.github.io/RiskBench/)
> - [Website](https://hcis-lab.github.io/RiskBench/) -->
---

## Setup

```bash
git clone --depth 1 https://github.com/HCIS-Lab/RiskBench.git
```

### System Requirements
- Linux ( Tested on Ubuntu 18.04, 20.04 )
- Python3 ( Tested on Python 3.7 )
- PyTorch ( Tested on PyTorch 1.10.0 )
- CUDA ( Tested on CUDA 11.3 )
- CARLA ( Tested on CARLA 0.9.14 )
- GPU ( Tested on Nvidia RTX3090, RTX4090 )
- CPU ( Tested on AMD 7950X3D, Intel 12900kf )


### Dataset
* The complete **RiskBench dataset** is available for download
  * [split folder into multiple large zip files. ](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EviA5ovlh6hPo_ZXEPQjxAQB2R3vNubk3HM1u4ib1VdPFA?e=WHEWdm)
  * [split folder into multiple fixed tar files, each 2 Gb. ](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/Ekv-ljiKGVJHmXaiSUCYTjoBIoGHvTIcMexbed5tp2oAhQ?e=RbxfhA).


* We provide **DATA_FOR_Planning_Aware_Metric** and **DATASET_for_LBC_Training** for planning aware metric evaluation and LBC training data respectively.

* We provide instructions on how to collect basic scenarios and data augmentation. Please refer to [link](Planning_Aware_Metric/).

<!-- Dataset statistics: 

|       | Amount | Example |
| :---- | :----- | :-----: |
| Train |        |         |
| Val   |        |         |
| Test  |        |         |

--> 

---

## Risk Identification Benchmark

### Baseline
We provide each baseline's training and inference details which can be found [here](risk_identification/Baselines).


### Offline Risk Identification Evaluation
We perform offline risk identification evaluation and fine-grained scenario-based analysis by taking input as preserved risk identification prediction. You can generate by following the instruction in this [page](risk_identification/Risk_identification_tool). 


### Planning-aware Evaluation
We provide data collection pipeline and planning aware evaluation platform which can be found [here](Planning_Aware_Metric)



---

## Citation
```
@inproceedings{kung2024riskbench,
  title={RiskBench: A Scenario-based Benchmark for Risk Identification},
  author={Kung, Chi-Hsi and Yang, Chieh-Chi and Pao, Pang-Yuan and Lu, Shu-Wei and Chen, Pin-Lun and Lu, Hsin-Cheng and Chen, Yi-Ting},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024},
  organization={IEEE}
}
```
