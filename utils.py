import os

from unet3d import UNet3D
import torch

def str_to_bool(s):
    return s.lower() in ('true', '1')

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
def load_model(model_path):
    # Load the trained model
    model = UNet3D(in_channels=4, num_classes=4)  # Ensure in_channels is set to 4 to match the checkpoint

    # Load the entire checkpoint
    checkpoint = torch.load(model_path)
    # Ensure the keys in the checkpoint match the model state_dict
    print("Checkpoint keys:", checkpoint.keys())

    # Extract only the model state_dict
    model.load_state_dict(checkpoint['Model'])
    model.eval()
    return model