from data_prep.preprocessing import *
from dataloaders.loaders import *
from models.model import *
from utils.embed_utils import *
from utils.trainer_utils import *
from trainers import *
from trainers.losses import *

from config import config
import wandb

import numpy as np
import pandas as pd
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



def load_existing_data(source):
    copy_files_h5 = glob(source+"*.h5")
    copy_file = copy_files_h5 
    _=[copyfile(x, '/content/corrnet/{}'.format(x.split('/')[-1])) for x in copy_file] 


if __name__ == '__main__':

    df_train=pd.read_csv(config['train_csv_path'])
    df_test=pd.read_csv(config['test_csv_path'])

    if not config['hdf5_exists']:
        hdf5_ob=create_hdf5()
        train_invalid=hdf5_ob.write_images_into_HDF5(df_train['media_path'],config['train_hdf5_name'])
        test_invalid=hdf5_ob.write_images_into_HDF5(df_test['media_path'],config['test_hdf5_name'])
        
        if train_invalid!=df_train.shape[0] and test_invalid!=df_test.shape[0]:
            df_train=df_train[~df_train['media_path'].isin(train_invalid)]
            df_test=df_test[~df_test['media_path'].isin(test_invalid)]

    else:
        source=config['source_dir']
        load_existing_data(source)
        print("Hdf5 loaded")    



    
    train_transformations = transforms.Compose([torchvision.transforms.ToPILImage(mode="RGB"),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomResizedCrop(224,interpolation=3),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])

    transformations = transforms.Compose([torchvision.transforms.ToPILImage(mode="RGB"),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])

    get_fasstext_vector = Fasttext(fasttext_model)                                     

    text_emb_size = config['text_emb_size']
    training_image_array = h5py.File(config['train_hdf5_name'], 'r')
    train_ds = CustomDataset(df_train,training_image_array,train_transformations,get_fasstext_vector,text_emb_size)
    trainLoader = DataLoader(train_ds, shuffle=True, batch_size=config['batch_size'], pin_memory = torch.cuda.is_available())

    testing_image_array = h5py.File(config['test_hdf5_name'],'r')
    test_ds = CustomDataset(df_test,testing_image_array,transformations,get_fasstext_vector,text_emb_size)
    testLoader = DataLoader(test_ds, shuffle=False, batch_size=config['batch_size'], pin_memory = torch.cuda.is_available())   

    image_emb_size = config['image_emb_size']
    text_emb_size = config['text_emb_size']
    middle_emb_size = config['middle_emb_size']
    lamda = config['lamda']

    if config['loss_function']=='clip':
        loss_function=clip_loss()
    elif config['loss_function']=='corr_loss':
        loss_function=CorrnetCost(lamda)
    else:
        loss_function=clip_corrnet_loss(lamda)        

    corrnet_model = Corrnet_Model(image_emb_size,text_emb_size,middle_emb_size,loss_function)
    if config['load_pretrained']:
        corrnet_model.load_state_dict(torch.load(os.path.join(config['source_dir'],config['model_name'])))

    resnet_model = Resnet50()
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    if config['optimizer']=='sgd':
        optimizer = optim.SGD(corrnet_model.parameters(), lr=config['lr'])
    else:        
        optimizer = optim.Adam(corrnet_model.parameters(), lr=config['lr'])

    decayRate = config['decayRate']

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    if torch.cuda.is_available:
        corrnet_model,resnet_model,cosine_sim= corrnet_model.cuda(),resnet_model.cuda(),cosine_sim.cuda()

    n_epochs = config['n_epochs']
    print_freq = config['print_freq']  

    best_test_img2img,best_test_txt2txt,best_test_img2txt = 0,0,0
    val_r5_acc_img2txt_best,val_r5_acc_txt2img_best=0,0  

    #wandb stuff

    wandb.init(project=config['project_name'],entity=config['username'])
    wandb.run.name = config['experiment_name']
    wandb.run.save()
    wandb.watch(corrnet_model)

    #training loop
    for epoch in range(0, n_epochs + 1):
        corrnet_model,resnet_model=train_epoch(corrnet_model,trainLoader,resnet_model,cosine_sim,optimizer,epoch,print_freq,wandb,)
        val_r5_acc_img2txt,val_r5_acc_txt2img=test_epoch(corrnet_model,testLoader,resnet_model,cosine_sim,optimizer,epoch,print_freq,wandb)

        if val_r5_acc_img2txt.avg>=val_r5_acc_img2txt_best or val_r5_acc_txt2img.avg>=val_r5_acc_txt2img_best:
            val_r5_acc_img2txt_best,val_r5_acc_txt2img_best=val_r5_acc_img2txt.avg,val_r5_acc_txt2img.avg
            if not os.path.isdir(config['source_dir']):
                os.mkdir(config['source_dir'])
            torch.save(corrnet_model.state_dict(),os.path.join(config['source_dir'],config['model_name']))
            print("YIPPEEE MODEL SAVED")
            print(f"{best_test_img2img} , {best_test_txt2txt} , {best_test_img2txt}")






        







