import torch

state = torch.load('pretrained_models/model_controller/model_controller')
for name, para in state.named_parameters():
    print(name)
    print(para)
