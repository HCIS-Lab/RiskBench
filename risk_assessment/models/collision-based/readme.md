# Collision-based
## Args
* --mode
    * training
        * Training
    * testing
        * testing & validating
    * TTC
        * TTC testing & validating
    * ROI
        * ROI testing & validating
* --model
    * model path
* --batch
    * batch number

## GT_loader.py
Get ground truth frame, object id.
All ground truth files are in "GT_loader/" folder.
## carla_dataset.py
customize dataloader.
## SA

### dsa_rnn.py
run model
### baseline2_model.py 
model file
## Risky Region

### dsa_rnn_supervised.py
run model
### baseline3_model.py
model file