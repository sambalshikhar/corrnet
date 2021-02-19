from data_prep.preprocessing import *
from dataloaders.loaders import *
from models.model import *
from utils.trainer_utils import *

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




def train_epoch(corrnet_model,trainLoader,resnet_model,cosine_sim,optimizer,epoch,print_freq,my_lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    loss_train = AverageMeter('clip_loss', ':6.2f')
    loss_test = AverageMeter('clip_loss', ':6.2f')

    img2img = AverageMeter('img2img', ':6.2f')
    txt2txt = AverageMeter('txt2txt', ':6.2f')
    img2txt = AverageMeter('img2txt', ':6.2f')

    test_img2img = AverageMeter('test_img2img', ':6.2f')
    test_txt2txt = AverageMeter('test_txt2txt', ':6.2f')
    test_img2txt = AverageMeter('test_img2txt', ':6.2f')

    val_r1_acc_txt2img = AverageMeter('val_r@1_txt2img', ':6.2f')
    val_r5_acc_txt2img = AverageMeter('val_r@5_txt2img', ':6.2f')
    val_r10_acc_txt2img = AverageMeter('val_r@10_txt2img', ':6.2f')

    val_r1_acc_img2txt = AverageMeter('val_r@1_img2txt', ':6.2f')
    val_r5_acc_img2txt = AverageMeter('val_r@5_img2txt', ':6.2f')
    val_r10_acc_img2txt = AverageMeter('val_r@10_img2txt', ':6.2f')

    corrnet_model.train() 
    end = time.time()

    if my_lr_scheduler!=None:
        my_lr_scheduler.step()
    for batch,(image,text_emb,instance_labels) in enumerate(trainLoader):
        # zero the parameter gradients
        optimizer.zero_grad()
        progress_of_batch = ProgressMeter(
        len(trainLoader),
        [batch_time,data_time,loss_train,img2img,txt2txt,img2txt],
        prefix="Epoch: [{}] , Batch: [{}]".format(epoch,batch))
        data_time.update(time.time() - end)
        instance_labels = torch.squeeze(instance_labels,dim=0)

        if torch.cuda.is_available:
            image,text_emb,instance_labels = image.cuda(),text_emb.cuda(),instance_labels.cuda()

        image_emb = resnet_model(image)
        common_emb,common_image_emb,common_text_emb,rec_image_emb,rec_text_emb,loss = corrnet_model(image_emb,text_emb)

        image2image = cosine_sim(image_emb,rec_image_emb)
        image2image = torch.mean(image2image)
        text2text = cosine_sim(text_emb,rec_text_emb)
        text2text = torch.mean(text2text)
        image2text = cosine_sim(common_image_emb,common_text_emb)
        image2text = torch.mean(image2text)

        img2img.update(image2image.item(),image_emb.size(0))
        txt2txt.update(text2text.item(),image_emb.size(0))
        img2txt.update(image2text.item(),image_emb.size(0))

        loss.backward()
        optimizer.step()
       
        loss_train.update(loss.item(),image_emb.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % print_freq == 0:
            progress_of_batch.display(batch)

    return corrnet_model,resnet_model        


def test_epoch(corrnet_model,trainLoader,resnet_model,cosine_sim,optimizer,epoch,print_freq):

    corrnet_model.eval()
    resnet_model.eval()
    with torch.no_grad():
        for batch,(image,text_emb,instance_labels) in enumerate(testLoader):
        
            progress_of_test_batch = ProgressMeter(
            len(testLoader),    
            [loss_test,test_img2img,test_txt2txt,test_img2txt,val_r1_acc_txt2img,val_r5_acc_txt2img,val_r10_acc_txt2img,val_r1_acc_img2txt,val_r5_acc_img2txt,val_r10_acc_img2txt],
            prefix="Epoch: [{}] , Batch: [{}]".format(epoch,batch))
            data_time.update(time.time() - end)
            instance_labels = torch.squeeze(instance_labels,dim=0)
            if torch.cuda.is_available:
                image,text_emb,instance_labels = image.cuda(),text_emb.cuda(),instance_labels.cuda()

            image_emb = resnet_model(image)
            common_emb,common_image_emb,common_text_emb,rec_image_emb,rec_text_emb,loss = corrnet_model(image_emb,text_emb)
            

            image2image = cosine_sim(image_emb,rec_image_emb)
            image2image = torch.mean(image2image)
            text2text = cosine_sim(text_emb,rec_text_emb)
            text2text = torch.mean(text2text)
            image2text = cosine_sim(common_image_emb,common_text_emb)
            image2text = torch.mean(image2text)

            test_img2img.update(image2image.item(),image_emb.size(0))
            test_txt2txt.update(text2text.item(),image_emb.size(0))
            test_img2txt.update(image2text.item(),image_emb.size(0))

            
            img_feats=F.normalize(common_image_emb,p=2,dim=1)
            text_feats=F.normalize(common_text_emb,p=2,dim=1)
                    
            img_index=retreival_acc(img_feats,text_feats,instance_labels)
            txt_index=retreival_acc(text_feats,img_feats,instance_labels)

            r1 = len(np.where(np.array(txt_index)== 0)[0])/len(txt_index)
            val_r1_acc_txt2img.update(r1)
            r5 = len(np.where(np.array(txt_index)<= 5)[0])/len(txt_index)
            val_r5_acc_txt2img.update(r5)
            r10 = len(np.where(np.array(txt_index)<= 10)[0])/len(txt_index)
            val_r10_acc_txt2img.update(r10)

            r1 = len(np.where(np.array(img_index)== 0)[0])/len(img_index)
            val_r1_acc_img2txt.update(r1)
            r5 = len(np.where(np.array(img_index)<= 5)[0])/len(img_index)
            val_r5_acc_img2txt.update(r5)
            r10 = len(np.where(np.array(img_index)<= 10)[0])/len(img_index)
            val_r10_acc_img2txt.update(r10)

            loss_test.update(loss.item(),image_emb.size(0))

            if batch % print_freq == 0: 
                progress_of_test_batch.display(batch)

    return val_r5_acc_img2txt,val_r5_acc_txt2img