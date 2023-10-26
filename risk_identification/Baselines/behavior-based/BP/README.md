# Behavior Prediction (BP)
> **Notice**: The training detail of BP is similar to BCP, the only difference is how the risk score is obtained. The former is based on the attention weight of message passing, and the latter is based on causal inference.

## **Abstruct**
The task utilizes graph attention networks to model interactions between traffic participants and the ego-vehicle to predict ego behavior, and simplify the ego behaviors to be ‘Go’or‘Stop’.
If the predicted behavior is ”Influenced”, we select the object with the highest attention scores as the risk.

## **Problem Formulation**
* **Input:** A sequence of RGB frames $I = \{I_1, I_2\ ...\ I_T\}$  & object tracklet $O = \{O_1,O_2\ ...\ O_N\}$, $N$ is number of objects in the given tracklet list

* **Output:**  All the object ID with risk score $s^{go}_k$

## **Experiment**
	
- **Dataset**
	* Training & Validation Dataset: interactive, obstacle, non-interactive
    * Testing Dataset: interactive, collision, obstacle, non-interactive
<br />
<br />

- **Training Hyperparameter:**
	
	> Batch Size:	4
	> 
	> Learning Rate:	1e-7
	> 
	> Weight Decay:  0.01
	> 
	> Time Steps:	5
	> 
	> Loss Function: nn.BCELoss, **loss weight: {'Go': 1.0, 'Stop': 1.8}**
	> 
	> Ego Feature Backbone:	PDResNet50 pretrained on ImageNet
	> 
	> Object Feature Backbone:	ResNet50 pretrained on ImageNet
	> 
	> Image Size: 640\*256
	> 
	> Image Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

- **Pipeline**
    1. For the training phase, an object-level manipulable driving model is learned to predict driver’s behavior $s^{go}$, $s^{stop}$
	
	2. We select the attention weight ($attn_k$) of graph attention networks as a risk score , which means the interactions between traffic participant $O_k$ and the ego-vehicle

- **Metric:**
	1. Behavior prediction: recall, precision, f1-Score, AP
	2. Risk object identification (ROI): recall, precision, f1-Score. 
		
		* BP Condition: **$s^{go}$ < 0.4** and **($attn_k$) > 0.35**
		* When an object satasify the above conditions, $O_k$ is a risk object



## **Startup**

1. Environment set up
  	
	* Install dependencies in your favorite environment. 	
		```bash
		conda create -n BP_env python=3.7
		conda activate BP_env
		cd ${BP_ROOT}
		pip install -r requirements.txt
		```
	* Download and unzip [metadata.zip](https://nycu1-my.sharepoint.com/personal/ychen_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fychen%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FRiskBench%2FDATA%5FFOR%5FPlanning%5FAware%5FMetric) to `./datasets/`
1. Tracklet generating
	```bash
	python utils/gen_tracking.py	
	```
2.  Training
	```bash
	python train.py --data_type all --gpu DEVICE_ID
	```
3. Testing
	```bash
	python test.py --data_type ${DATA_TYPE} --ckpt_path ${PATH}
	```
4. Traing log
	```bash
	tensorboard dev upload --logdir logs/${LOG_NAME} --name ${LOG_NAME}
	```


		
