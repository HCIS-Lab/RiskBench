## **Training single-stage model**
	Input: RGB image frames and object tracking

	Output: 'Go' or 'Stop', e.g. driver behavior

    The training weights store in the "./snapshots/all/"

Execute **"python train.py"**


## **Testing ROI**
	Input: RGB image frames and object tracking

	Output: All the Object ID with the corresponding risk score in the whole dataset.
	e.g. {'17342': 0.52, '17490': 0.33, '17719': 0.18, ...}

    The testing result stored in the "./roi/"

Execute **"python eval_intervention_test.py"**
