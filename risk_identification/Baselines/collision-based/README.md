# Collision Based
The two baselines' models are both trained using exponential cross entropy loss for collision prediction with a batch size of 8 and optimized by AdamW with learning rate 1e-4. Each data contains 40 frames and we choose ResNet50 as our image backbone. Roi_Align is used for retreiving objects' features, kernel size is set to 8x8. For positive data, the time of collision is set at 0.8 times the total duration.

## DSA
DSA is trained for 10 epochs.

## RRL
Except for collision prediction, RRL is also trained with object collision label using binary cross entropy loss for 5 epochs. Experiment results show that RRL would be overfitted on collision scenarios for 10 epochs which induces inferior performance on risk identification on other scenario types. Hence we train RRL for only 5 epochs.

## Training
To train a RRL model, run a command like this:
```bash
python trainer.py --root ../dataset/ --epoch 10 --lr 1e-4 --batch 8 --supervised
```
Arguments: 

| Parameter     | Description                                          | Example       |
| :-------------| :--------------------------------------------------- | :-----------: |
| --root        | Dataset's root path                                  | ../dataset/   |
| --epoch       | Max training epochs                                  | 10            |
| --lr          | Learning rate                                        |     1e-4      |
| --batch       | Batch size                                           |    8          |
| --supervised  | Train a RRL model, if not train a DSA model          |    None       |
| --intention   | Train a model with agents' intention                 |    None       |
| --state       | Train a model with agents' state information         |    None       |
| --vis         | Visualize risk identification result, demo video is saved in ./vis/ folder. If not, run risk identification           |    None       |
| --model_path  | Model's checkpoint path                              | 9_12_0_56     |

## Inference
To inference a RRL model, run a command like this:
```bash
python demo.py --root ../dataset/ --supervised --model_path 9_12_0_56
```
Risk identification results will be saved as json file in this [folder](../../Risk_identification_tool/model/) with model's name.