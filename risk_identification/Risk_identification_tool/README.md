
1. 把 inference 完的結果 (raw scroe) 按照格式放到 `"./model/RRL/{DATA_TYPE}.json"`
   
2. 執行 `"python roi_metric.py --method RRL --transpose --threshold 0.00 --data_type interactive --save_result"`

3. 結果會存到 `"result/RRL/{DATA_TYPE}_result.json"`


