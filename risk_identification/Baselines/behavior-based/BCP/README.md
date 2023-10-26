# Behavior Change Prediction (BCP)


## **Abstruct**
The task is formulated as the cause-effect problem which is aims to predict the risk score of object. We defines **objects** (cause) that **influence driver behavior** (effect) as risky , and simplify the driver behaviors to be ‘Go’or‘Stop’.

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
	
	
    1. For the first stage and training phase, an object-level manipulable driving model is learned to predict driver’s behavior $s^{go}$, $s^{stop}$.
	
	2. For the second stage and testing phase, given a‘Stop’prediction ($s^{go}$ < 0.5) , i.e., driver behavior is influenced by objects) , we intervene input sequence by removing selected object tracklet $O_k$ and inpainting the removed area in each frame to simulate a scenario without the presence of the tracklet. The trained driving model is used to predict the corresponding driver behavior $s^{go}_k$, $s^{stop}_k$.
	
	3. Finally, we select objects that satisfy the following conditions as risk objects $O_k$: **driver behaviors (after remove object) $s^{go}_k$ - driver behaviors $s^{go}$ > 0.2** , indicating this object causes the most substantial driver behavioral change as the risk object. 

- **Metric:**
	1. Behavior prediction: recall, precision, f1-Score, AP
	2. Risk object identification (ROI): recall, precision, f1-Score. 
		
		* BCP Condition: **$s^{go}$ < 0.5** and **$s^{go}_k$ - $s^{go}$ > 0.2**
		* When an object satasify the above conditions, $O_k$ is a risk object



## **Startup**

1. Environment set up
  	
	* Install dependencies in your favorite environment. 	
		```bash
		conda create -n BCP_env python=3.7
		conda activate BCP_env
		cd ${BCP_ROOT}
		pip install -r requirements.txt
		```
	* Download and unzip [metadata.zip](https://nycu1-my.sharepoint.com/personal/ychen_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fychen%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FRiskBench%2FDATA%5FFOR%5FPlanning%5FAware%5FMetric) to `./datasets/`
2. Tracklet generating
	```bash
	python utils/gen_tracking.py	
	```
3.  Training
	```bash
	python train.py --data_type all --gpu DEVICE_ID
	```
4. Testing
	```bash
	python test.py --data_type ${DATA_TYPE} --ckpt_path ${PATH}
	```
5. Traing log
	```bash
	tensorboard dev upload --logdir logs/${LOG_NAME} --name ${LOG_NAME}
	```


		
