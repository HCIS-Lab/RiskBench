## **Environment set up**
	pip install -r requirements.txt

## **Generate tracklet**
	python utils/gen_tracking.py	

## **Training**
	python train.py --data_type all --gpu DEVICE_ID
	
## **Testing**
	python test.py --data_type {DATA_TYPE} --ckpt_path {PATH}

## **See traing log**
	tensorboard dev upload --logdir logs/{LOG_NAME} --name {LOG_NAME}
