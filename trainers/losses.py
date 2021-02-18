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


class clip_loss(nn.Module):

    def __init__(self):
        super(clip_loss,self).__init__()
                                     
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self,common_image_emb,common_text_emb,recx1,recx2,recx3,recy1,recy2,recy3):    
        img_feats=F.normalize(common_image_emb,p=2,dim=1)
        text_feats=F.normalize(common_text_emb,p=2,dim=1)


        logits_t2i = torch.matmul(img_feats, text_feats.T)
        logits_i2t = torch.matmul(text_feats, img_feats.T)

        labels = torch.arange(img_feats.size(0)).cuda()

        loss_t2i = self.criterion(logits_t2i, labels)
        loss_i2t = self.criterion(logits_i2t, labels)
        loss=(loss_t2i+loss_i2t)/2

        return loss 

class clip_corrnet_loss(nn.Module):

    def __init__(self,lamda):
        super(clip_corrnet_loss,self).__init__()

        self.clip=clip_loss()
        self.corrnet=CorrnetCost(lamda)

    def forward(self,y1,y2,recx1,recx2,recx3,recy1,recy2,recy3):

        clip_loss=self.clip(y1,y2)
        corrnet_loss=corrnet(y1,y2,recx1,recx2,recx3,recy1,recy2,recy3)  

        total_loss=clip_loss+corrnet_loss

        return total_loss

class CorrnetCost(nn.Module):
    def __init__(self,lamda):
        super(CorrnetCost,self).__init__()
        self.lamda = -lamda
        self.mse_loss = nn.MSELoss()

    
    def forward(self,y1,y2,recx1,recx2,recx3,recy1,recy2,recy3):
        y1_mean = torch.unsqueeze(torch.mean(y1,dim=1),dim=1)
        y1_centered = y1 - y1_mean
        y2_mean = torch.unsqueeze(torch.mean(y2,dim=1),dim=1)
        y2_centered = y2 - y2_mean
        corr_nr = torch.sum(y1_centered*y2_centered,dim=1)
        corr_dr1 = torch.sqrt(torch.sum(y1_centered*y1_centered,dim=1)+ 1e-8)
        corr_dr2 = torch.sqrt(torch.sum(y2_centered*y2_centered,dim=1)+ 1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr / corr_dr
        corr = torch.sum(corr)*self.lamda

        loss1 = self.mse_loss(recx1,image_input)
        loss2 = self.mse_loss(recx2,image_input)
        loss3 = self.mse_loss(recx3,image_input)
        loss4 = self.mse_loss(recy1,text_input)
        loss5 = self.mse_loss(recy2,text_input)
        loss6 = self.mse_loss(recy3,text_input)

        loss = corr + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        
        return loss



