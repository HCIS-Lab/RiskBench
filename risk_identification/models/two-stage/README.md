## **Training two-stage model**
	Input: RGB image frames and object tracking

	Output: 'Go' or 'Stop', e.g. driver behavior

    The training weights store in the "./snapshots/all/"

Step1: Move to two-stage\train\HDD\gcn

Step2: Execute **"python train.py --lr 0.0000001 --time_steps 5 --batch_size 4 --num_workers 2 --cause 'all' "**

## **Testing ROI**
	Input: RGB image frames and object tracking

	Output: All the Object ID with the corresponding risk score in the whole dataset.
	e.g. {'17342': 0.52, '17490': 0.33, '17719': 0.18, ...}

    The testing result stored in the "./RA/"

Step1: Move to two-stage\train\HDD\gcn

Step2: Execute **"python eval_intervention_benchmark.py --cause all --time_steps 5 --vis --show_process"**
