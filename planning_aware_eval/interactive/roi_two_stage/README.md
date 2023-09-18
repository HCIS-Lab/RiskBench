## **Environment**
**conda Environment A**\
Step1: conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge                
Step2: Move to models/roi_align/ and excute "python setup.py install"


## **Testing ROI**
	Input: RGB image frames and object tracking

	Output: True or False, it represents whether the corresponding object ID is risky
	e.g. {'17342': False, '17490': False, '17719': False, '17820': True, '21183': False}

Step1: Move to roi_jacky_v5

Step2: Activate **conda environment A**

Step3: Put **5 testing rgb image**, **instance image**, **obstacle_info.json (if required)** and **actor_list.csv** in the testing folder with specific format (default folder is **inference/test_data/**)

Step4: Excute **"python demo.py --cause {data_type} "** or **"python demo.py --cause {data_type} --vis"**

	Notice:
	1. ***Important*** inference/temp_weight/{data_type} (hidden state & roi history) needs to be cleaned up before running inference in the new scenario. You can use "--clean_state" to delete the saved hidden state and roi_history.json.
   
	2. You can also see the visualization result in inference/vis/

	3. {data_type}: interactuve, collision, obstacle, non-interactive



	python demo.py --cause interactuve