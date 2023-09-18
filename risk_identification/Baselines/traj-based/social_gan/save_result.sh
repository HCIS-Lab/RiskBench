#!/bin/bash

count=0

for f in ${4}/*.txt; do
	python3.8 scripts/evaluate_model_carla.py \
		--model_path ${6} \
		--infer_data $(basename "$f") \
		--result_dir ${5} \
		--dset_type ${2} \
		--sc_type ${1} \
		--pred_len ${3}
	count=$((count+1))
done
echo $count
