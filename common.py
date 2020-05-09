import io
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torchvision.models as models

#from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

# def get_model():

#     checkpoint_path = '/Users/bjoh1/python/capst/classifier.pt'
#     model = models.densenet121(pretrained=True)
#     model.classifier=nn.Linear(1024,102)
#     model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=False)
#     model.eval()
#     return model

def get_body():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 6, bias=True)
    fc_parameters = model.fc.parameters()
    for param in fc_parameters:
        param.requires_grad = True
    model.load_state_dict(torch.load('saved_models/model_transfer.pt'))   

    #img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    return model


def load_input_image(img_path):
    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])    
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image    

def get_tensor(image_bytes):
    transform=transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485 ,0.456, 0.406],
    [0.228, 0.224, 0.225])])

    image=Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)
