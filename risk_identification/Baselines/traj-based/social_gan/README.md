# Social-GAN

## Installation
You can also install the dependencies using pip
```shell
pip install -r requirements.txt
```

## Data Retrieving
You can retrieve raw data to your desire position. data_type: non-interactive, interactive, obstacle, collision
```shell
python retrieve_data.py --input_path ${input_path} --data_type ${data_type} --output_path ${output_path}
```

## Data Preprocessing
You can run the following script to convert the format of the original data to the social-GAN input format.
```shell
python data_proc.py --input_path ${input_path} --dynamic_description_path ${dynamic_description_path} --output_path ${output_path}
```

## Training
You can train the model by running the following script.
```shell
python scripts/train.py ... (Some training hyperparameters and settings)
```
Or you can easily run the following shell script to train the model with our default settings.
```shell
bash run_traj.sh
```
You can download the pretrained weight for each scenario type by the following [google drive link](https://drive.google.com/drive/u/4/folders/1MjpmgLGMpvrm3WEV9WCuBcwyJh7wGJLy).


## Evaluation
For the performance evaluation, you can use the following script to get the ADE/FDE for each scenario type.
```shell
python scripts/evaluate_model.py --model_path ${model_path} \
    --num_samples ${num_samples} \
    --pred_len ${pred_len} \
    --sc_type ${scenario_type} \
    --dset_type ${test_or_val}
```

## Inference
You can feed single input file to the model and save it's prediction result by running the following script.
```shell
python scripts/evaluate_model_carla.py \
    --model_path ${pretrained_model_path} \
    --infer_data ${infer_data} \
    --result_dir ${output_dir} \
    --dset_type ${test_or_val} \
    --sc_type ${scenario_type} \
    --pred_len ${pred_len}
```

If you want to save all the prediction results under a directory, you can directly run the following shell script.
```shell
bash save_result.sh ${sc_type} ${test_or_val} ${pred_len} ${input_dir} ${output_dir} ${pretrained_model_path}
```

## Risky object identification (faster)
You can get a risky object file as a json file in each scenario.
```shell
python extract_data_no_plot.py
    --future_length ${future_length} \
    --data_path ${data_path} 
```
## Risky object identification with bev figures (slower)
You can get a risky object file as a json file and bird-eye-view prediction figures of each frame in each scenario.
```shell
python extract_data.py
    --future_length ${future_length} \
    --data_path ${data_path} 
```
## Preprocess for metric calculating
You can integrate these json files into a overall format.
```shell
python final_output.py
    --future_length ${future_length} \
    --data_path ${data_path} \
    --val_or_test ${val_or_test} \
    --method ${mantra}
```