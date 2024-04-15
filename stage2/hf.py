import os
import argparse
import torch
from models.transformer_net import ImageTransfomer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained model
nst_model = ImageTransfomer()
training_state = torch.load('../checkpoints/stage2/wave_crop.pth')

state_dict = training_state["state_dict"]
nst_model.load_state_dict(state_dict, strict=True)

nst_model.save_pretrained("wave_crop")

nst_model.push_to_hub("SavageSanta25/johnson-wavecrop")
nst_model = ImageTransfomer.from_pretrained("SavageSanta25/johnson-wavecrop")
print(nst_model)