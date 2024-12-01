import numpy as np
import shap
import torch
import torch.nn as nn
from models.unet_tranformer import PPGUnet
from models.fine_tune import FineTuneSBP, FineTuneDBP, FineTuneMAP
from models.custom_dataloader import UCIBPDatasetNPY
from torch.utils.data import random_split

class scaling(nn.Module):

    def __init__(self, const):
        super().__init__()
        self.const = const


    def forward(self, x):
        return x*self.const

map_location = "cpu"

device = torch.device(map_location)

# Loading model and setting requires grad to False 

model = PPGUnet(in_channels=1).to(device)
model.load_state_dict(torch.load("checkpoints/ucibp-npy-ReLU/PPGUnet_epoch299.pth", map_location=map_location))

# Loading the finetune model

fine_tune_param = "dbp"

if fine_tune_param == "sbp":  
    finetune = FineTuneSBP(1, 64, 1).to(device)
    param_idx = 0


elif fine_tune_param == "dbp":
    finetune = FineTuneDBP(1, 64, 1).to(device)
    param_idx = 1

finetune.load_state_dict(torch.load(f"checkpoints/ucibp-finetune/{fine_tune_param}/PPGUnet_finetune_{fine_tune_param}_epoch19.pth", map_location=map_location))

scaling_layer = scaling(const=1/200)

aggregated_model = nn.Sequential(
    model, 
    finetune, 
    scaling_layer
)


UCIBP_dataset = UCIBPDatasetNPY("dataset/uci-bp-processed")
train_dataset, valid_dataset, test_dataset = random_split(UCIBP_dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))

shap_dataset = torch.from_numpy(train_dataset[:100][0])
shap_dataset = shap_dataset.unsqueeze(1).to(device)

explainer = shap.DeepExplainer(aggregated_model, shap_dataset)


for i in range(100):
    x = torch.from_numpy(train_dataset[100 + i:101 + i][0])
    x = x.unsqueeze(1).to(device)




    values = explainer.shap_values(x)
    print(type(values))

    with open(f"checkpoints/explainers/{fine_tune_param}/shap_{i}.npy", "wb") as f:
        np.save(f, values)
