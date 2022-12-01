# Mantra

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

## Training
You can train the model by running the following script.
```shell
python3.8 train_ae_risk.py
python3.8 train_controllerMem_risk.py --model ${model_ae_path}
```
You can download the pretrained weight for each scenario type by the following [google drive link](https://drive.google.com/drive/folders/1DGENQhLkSGrihMi1JsaZItqkNazUD52_).

## Evaluation
For the performance evaluation, you can use the following script to get the ADE/FDE for each scenario type.
```shell
python test.py \
    --evaluate_or_inference ${evaluate} \
    --model ${model_path} \
    --memories_path ${memories_path} \
    --saved_memory ${memory_exist} \
    --dataset_file ${dataset_file} \
    --future_len ${future_len} \
    --data_type ${data_type} \
    --dset_type ${val_or_test}
```

## Inference
You can feed files to the model and save it's prediction result at default directory by running the following python file.
```shell
python test.py \
    --evaluate_or_inference ${inference} \
    --model ${model_path} \
    --memories_path ${memories_path} \
    --saved_memory ${memory_exist} \
    --dataset_file ${dataset_file} \
    --dset_type ${val_or_test} \
    --data_type ${data_type} \
    --future_len ${future_len}
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