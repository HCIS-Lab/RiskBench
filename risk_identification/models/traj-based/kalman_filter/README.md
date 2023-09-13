# Kalman_filter

## Data Retrieving
You can retrieve raw data to your desire position. data_type: non-interactive, interactive, obstacle, collision
```shell
python retrieve_data.py --input_path ${input_path} --data_type ${data_type} --output_path ${output_path}
```

## Inference with only results of risky objects (faster)
You can get all types' results of risky objects by running this python file.
```shell
python3.8 kf_no_plt.py
    --future_length ${future_length} \
    --data_path ${data_path} \
    --val_or_test ${val_or_test} 
```

## Inference with bev figures and results of risky objects (slower)
You can get all types' results of risky objects by running this python file.
```shell
python3.8 kf_per_frame.py
    --future_length ${future_length} \
    --data_path ${data_path} \
    --val_or_test ${val_or_test} 
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