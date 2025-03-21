# Setup
1. Download `RiskBench_Dataset` [here](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EviA5ovlh6hPo_ZXEPQjxAQB2R3vNubk3HM1u4ib1VdPFA?e=WHEWdm).
2. Download `metadata.zip` [here](https://nycu1-my.sharepoint.com/personal/ychen_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fychen%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FRiskBench%2FDATA%5FFOR%5FPlanning%5FAware%5FMetric).
3. Unzip `model.zip`, `metadata.zip`, `RiskBench_Dataset`
4. Install dependencies in your favorite environment. 
	
	e.g. start by creating a new conda environment:
	```bash
	conda create -n analysis_tool python=3.7
	conda activate analysis_tool
	cd ${TOOL_ROOT}

	conda  install pyqt
	pip3 install -r requirements.txt
	```

# Quantitative Results For Risk Object Identification (ROI)
1. Execute 
	```bash
	python ROI_tool.py --method ${MODEL} --data_type ${DATA_TYPE} --metadata_root ${METADATA_ROOT} --save_result --result_path ${ROI_PATH}
	```

2. The results will be saved to `${ROI_PATH}/${MODEL}/${DATA_TYPE}.josn`




# Fine-grained Scenario-based Analysis

1. Execute 
	```bash
	python ROI_vis_tool.py --data_root ${DATASET_ROOT} --metadata_root ${METADATA_ROOT} --vis_result_path ${VIS_PATH}
	```
2. Choose Model and Scenario Type
3. Check interest attributes
4. Click **Filter Scenario**
5. Choose one scenario
   * Click **Generate Video** to save the qualitative result in gif file
   * Click **Generate JSON** to save the quantitative result in JSON file
   * The results will be saved to `${VIS_PATH}/gif/${MODEL}/${DATA_TYPE}` or `{VIS_PATH}/json/${MODEL}/${DATA_TYPE}`


	![Fine-grained Scenario-based Analysis](utils/localization_anticipation.gif)

