import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data
from torchvision.models import resnet50
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Resnet50(nn.Module):
    
  def __init__(self,use_pretrained = True,feature_extract = True):
    super(Resnet50,self).__init__()
    self.feature_extract = feature_extract
    self.model_ft = models.resnet50(pretrained=use_pretrained)
    self.num_ftrs = self.model_ft.fc.in_features
    self.model_ft = torch.nn.Sequential(*list(self.model_ft.children())[:-1])
    set_parameter_requires_grad(self.model_ft, feature_extract)
    
  def forward(self,image):
    resnet_feat = torch.squeeze(self.model_ft(image))
    if self.feature_extract:
        resnet_feat = resnet_feat.detach()
    return resnet_feat

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Corrnet_Model(nn.Module):
    def __init__(self,image_emb_size,text_emb_size,middle_emb_size,loss_function):
        super(Corrnet_Model,self).__init__()
        self.image_emb_size = image_emb_size
        self.text_emb_size = text_emb_size
        self.dense_image_0 = nn.Linear(image_emb_size,middle_emb_size)
        self.dense_text_0 = nn.Linear(text_emb_size,middle_emb_size)

        self.dense_image_1 = nn.Linear(middle_emb_size,middle_emb_size)
        self.dense_text_1 = nn.Linear(middle_emb_size,middle_emb_size)

        self.dense_image_2 = nn.Linear(middle_emb_size,image_emb_size)
        self.dense_text_2 = nn.Linear(middle_emb_size,text_emb_size) 

        self.dense_common = nn.Linear(middle_emb_size,middle_emb_size)

        self.loss_function=loss_function
        
        if torch.cuda.is_available:
            self.loss_function =self.loss_function.cuda()
        
    def forward(self,image_input,text_input):
        image_input = F.normalize(image_input, p=2, dim=1)
        text_input = F.normalize(text_input, p=2, dim=1)
        batch_size = image_input.size(0)
        image_zeros_arr = torch.zeros(batch_size,self.image_emb_size)
        text_zeros_arr = torch.zeros(batch_size,self.text_emb_size)

        if torch.cuda.is_available:
            image_zeros_arr,text_zeros_arr = image_zeros_arr.cuda(),text_zeros_arr.cuda()

        recx1,recy1,h1 = self.model(image_input,text_zeros_arr)
        recx2,recy2,h2 = self.model(image_zeros_arr,text_input)
        recx3,recy3,h3 = self.model(image_input,text_input)
        loss=self.loss_function(h1,h2,recx1,recx2,recx3,recy1,recy2,recy3)
        
        return h3,h1,h2,recx3,recy3,loss 

    
    def model(self,image_input,text_input):
        x_image = self.dense_image_0(image_input)
        x_text = self.dense_text_0(text_input)

        h = x_image + x_text

        rec_x_image = self.dense_image_2(h)
        rec_y_image = self.dense_text_2(h)

        return rec_x_image,rec_y_image,h
