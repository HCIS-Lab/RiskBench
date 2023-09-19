# LBC
Train LBC model for plannning aware metric.

Most code fork from https://github.com/bradyz/2020_CARLA_challenge


# Environment setup

```bash
# create conda environment 
conda create -n LBC python=3.7
conda activate LBC

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

```
## Training

1. we need to login in wandb.

2.  download LBC Dataset from [link](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EviA5ovlh6hPo_ZXEPQjxAQB2R3vNubk3HM1u4ib1VdPFA?e=WHEWdm). And unzip the dataset.

3.  Ready to train.

```bash
# train interactive ( + non-interactive ) dataset
python map_model.py --max_epochs 8 --train_type interactive --dataset_dir /path/to/dataset

# train obstacle ( + non-interactive ) datasset
python map_model.py --max_epochs 7 --train_type obstacle --dataset_dir /path/to/dataset
```

## Reference 

[Learning by cheating]( https://arxiv.org/abs/1912.12294 )