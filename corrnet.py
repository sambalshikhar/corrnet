from data_prep.preprocessing import *
from dataloaders.loaders import *
from models.model import *
from utils.embed_utils import *
from utils.trainer_utils import *
from trainers import *
from trainers.losses import *
from evaluation import *

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



class corrnet():
    def __init__(self):

        self.df_train=pd.read_csv(config['train_csv_path'])
        self.df_test=pd.read_csv(config['test_csv_path'])

        source=config['source_dir']

        self.text_emb_size = config['text_emb_size']
        self.training_image_array = h5py.File(config['train_hdf5_name'], 'r')
        self.testing_image_array = h5py.File(config['test_hdf5_name'],'r')

        self.image_emb_size = config['image_emb_size']
        self.text_emb_size = config['text_emb_size']
        self.middle_emb_size = config['middle_emb_size']
        self.lamda = config['lamda']

        if config['loss_function']=='clip':
            self.loss_function=clip_loss()
        elif config['loss_function']=='corr_loss':
            self.loss_function=CorrnetCost(lamda)
        else:
            self.loss_function=clip_corrnet_loss(lamda)

        self.get_fasstext_vector = Fasttext(fasttext_model)        


        self.corrnet_model = Corrnet_Model(self.image_emb_size,self.text_emb_size,self.middle_emb_size,self.loss_function)
        if config['load_pretrained']:
            self.corrnet_model.load_state_dict(torch.load(os.path.join(config['source_dir'],config['model_name'])))

        self.resnet_model = Resnet50()
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)  

        if torch.cuda.is_available:
            self.corrnet_model,self.resnet_model,self.cosine_sim= self.corrnet_model.cuda(),self.resnet_model.cuda(),self.cosine_sim.cuda()  

    def train(self):

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

        train_ds = CustomDataset(self.df_train,self.training_image_array,train_transformations,self.get_fasstext_vector,self.text_emb_size)
        trainLoader = DataLoader(train_ds, shuffle=True, batch_size=config['batch_size'], pin_memory = torch.cuda.is_available())
        
        test_ds = CustomDataset(self.df_test,self.testing_image_array,transformations,self.get_fasstext_vector,self.text_emb_size)
        testLoader = DataLoader(test_ds, shuffle=False, batch_size=config['batch_size'], pin_memory = torch.cuda.is_available())                                       


        if config['optimizer']=='sgd':
            optimizer = optim.SGD(self.corrnet_model.parameters(), lr=config['lr'])
        else:        
            optimizer = optim.Adam(self.corrnet_model.parameters(), lr=config['lr'])

        decayRate = config['decayRate']

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        n_epochs = config['n_epochs']
        print_freq = config['print_freq']  

        best_test_img2img,best_test_txt2txt,best_test_img2txt = 0,0,0
        val_r5_acc_img2txt_best,val_r5_acc_txt2img_best=0,0  

        #wandb stuff

        wandb.init(project=config['project_name'],entity=config['username'])
        wandb.run.name = config['experiment_name']
        wandb.run.save()
        wandb.watch(self.corrnet_model)

        #training loop
        for epoch in range(0, n_epochs + 1):
            self.corrnet_model,self.resnet_model=train_epoch(self.corrnet_model,trainLoader,self.resnet_model,self.cosine_sim,optimizer,epoch,print_freq,wandb)
            val_r5_acc_img2txt,val_r5_acc_txt2img=test_epoch(self.corrnet_model,testLoader,self.resnet_model,self.cosine_sim,optimizer,epoch,print_freq,wandb)

            if val_r5_acc_img2txt.avg>=val_r5_acc_img2txt_best or val_r5_acc_txt2img.avg>=val_r5_acc_txt2img_best:
                val_r5_acc_img2txt_best,val_r5_acc_txt2img_best=val_r5_acc_img2txt.avg,val_r5_acc_txt2img.avg
                if not os.path.isdir(config['source_dir']):
                    os.mkdir(config['source_dir'])
                torch.save(self.corrnet_model.state_dict(),os.path.join(config['source_dir'],config['model_name']))
                print("YIPPEEE MODEL SAVED")
                print(f"{best_test_img2img} , {best_test_txt2txt} , {best_test_img2txt}")

    def evaluate(self,retrieval_type,random_seed):

        transformations = transforms.Compose([torchvision.transforms.ToPILImage(mode="RGB"),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        test_ds = CustomDataset(self.df_test,self.testing_image_array,transformations,self.get_fasstext_vector,self.text_emb_size)
        testLoader = DataLoader(test_ds, shuffle=False, batch_size=config['batch_size'], pin_memory = torch.cuda.is_available())

        if not os.path.isdir("./annoy_indices"):
            common_emb_pool,only_image_emb_pool,only_text_emb_pool=create_emb_pool(self.corrnet_model,self.resnet_model,testLoader)
            create_annoy_index(common_emb_pool,only_image_emb_pool,only_text_emb_pool)

        retreival_df=self.df_test.sample(config['evaluate_n_results'],random_state=random_seed)
        test_ds_eval = CustomDataset(retreival_df,self.testing_image_array,transformations,self.get_fasstext_vector,self.text_emb_size)
        retreival_loader = DataLoader(test_ds_eval, shuffle=False, batch_size=1, pin_memory = torch.cuda.is_available())
        
        annoy_index= AnnoyIndex(1024, 'angular')  

        if retrieval_type=='txt2img':   
            annoy_index.load("./annoy_indices/image.ann")
        elif retrieval_type=='img2txt': 
            annoy_index.load("./annoy_indices/text.ann")
        else:
            annoy_index.load("./annoy_indices/common.ann")

        retreival_indexes=get_retrievals(retreival_loader,retreival_df,annoy_index,self.corrnet_model,self.resnet_model,self.df_test,self.testing_image_array,retrieval_type)    



if __name__ == '__main__':
    
    config['load_pretrained']=True
    corrnet_object=corrnet()
    corrnet_object.evaluate("txt2img",123)























