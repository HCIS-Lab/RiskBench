# RiskBench: A Scenario-based Risk Assessment Benchmark
This is an anonymous repo for submission.
PyTorch code for RiskBench.
![RiskBench](images/teaser2.png)

## System requirements
- Linux (Tested on Ubuntu 18.04)
- Python3 (Tested using Python xxx)
- PyTorch (Tested using PyTorch xxx)
- CUDA (Tested using CUDA xxx)
- CARLA (Tested using CARLA 0.9.13)

## Installation

## Dataset
We currently provide sample scenarios for testing. The full dataset will be released soon.

## Code execution

### Risk Assessment Prediction (optional)
We provide risk assessment results in json format though, user can generate risk assessment prediction from raw data by:

### Offline Risk Assessment
We perform offline risk assessment evaluation (with metric F-1 score and PIC) by taking input as preserved risk assessment prediction:
```
python risk_assessment_identification.py --path {PREDICTION_PATH} --model {MODEL} --mode {MODE} --scenario {ATTRIBUTE}
```

Arguments: 

| Parameter     | Description                                          |  Example   |
| :------------ | :--------------------------------------------------- | :--------: |
| --path        | path of the stored prediction .json file             | prediction |
| --model       | name of the risk assessment method                   | two-stage  |
| --mode        | Risk assessment metric                               |     F1     |
| --scenario    | scenario filter, default is ""                       |    Rain    |
| --save_result | save result to {PREDICTION_PATH}/result/{MODEL}.json |    None    |

### Planning-aware Risk Assessment
Planning-aware risk assessment evaluation takes place in CARLA simulaotr.
XXX
```
python planning_aware_eval.py --path xxx --model xxx 
```
