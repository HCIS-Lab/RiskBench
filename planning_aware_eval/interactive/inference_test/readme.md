pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3.1

## **Testing ROI**
Step1: put n rgb img in test_data/rgb/front, n bbox json file in test_data/bbox/front

Step2: 


Excute **"python inference.py --model baseline3/model_15 --mode 3"  for RiskyRegion &Supervised

Excute **"python inference.py --model baseline2/model_19 --mode 2"  for SA