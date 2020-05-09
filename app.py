#import torch
import sys
import cv2

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
#파일 업로드

import glob
#파일이름 보호
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
import torch
import numpy as np
import os
from glob import glob
import cv2
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models


app = Flask(__name__)
# Disable scientific notation for clarity
model_transfer = models.resnet50(pretrained=True)
for param in model_transfer.parameters():
    param.requires_grad = False
model_transfer.fc = nn.Linear(2048, 133, bias=True)
fc_parameters = model_transfer.fc.parameters()
for param in fc_parameters:
    param.requires_grad = True

model_transfer.load_state_dict(torch.load('saved_models/model_transfer.pt'))

data_dir = '/Users/bjoh1/python/capst/body/'
train_dir = os.path.join(data_dir, 'train/')
valid_dir = os.path.join(data_dir, 'valid/')
test_dir = os.path.join(data_dir, 'test/')
batch_size = 20
num_workers = 0



standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   'val': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])
                  }

train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])


train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)
loaders_scratch = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}
loaders_transfer = loaders_scratch.copy()

class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]

def load_input_image(img_path):    
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image

def body(model, class_names, img_path):
    # load the image and return the predicted breed
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]



@app.route("/",methods=['GET'])
def index():

    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']

        basepath = os.path.dirname['file']
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        preds = body(model, class_names, file_path)
       
        
        return preds

    return None
#메인 모듈로 실행될 때 플라스크 서버 구동
if __name__ == "__main__":
    #app.run(host='0.0.0.0' , debug=True)
    app.run(debug=True)
