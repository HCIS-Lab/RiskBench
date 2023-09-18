# Setup
1. git clone [https://github.com/RiskBench_two-stage.git](https://github.com/WaywayPao/RiskBench_two-stage.git)
2. Download RiskBench Dataset & metadata
3. Install dependencies in your favorite environment. 

	e.g. start by creating a new conda environment:
		
		conda create -n analysis_tool python=3.7
		conda activate analysis_tool
		cd ${TOOL_ROOT}

		conda  install pyqt
		pip3 install -r requirements.txt


# Scenario-based Analysis

1.		python start.py --data_root ${DATASET_ROOT} --metadata_root ${METADATA_ROOT} --vis_result_path ${VIS_PATH}
2. Choose Model and Scenario Type
3. Check interest attributes
4. Click **Filter Scenario**
5. Choose one scenario and click **Generate Video**
6. Click **Generate JSON** to save the reult in JSON file

