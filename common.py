import io
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def get_model():
    checkpoint_path = ' classifier.pt'
    model = models.densenet121(pretrained=True)
    model.classifier=nn.Linear(1024,102)
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=False)
    model.eval()
    return model

def get_tensor()
