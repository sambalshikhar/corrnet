import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.utils.data
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,image_array,transforms,get_fasstext_vector,text_emb_size):
        super(CustomDataset, self).__init__()
        self.data = dataset
        self.image_array = image_array
        self.transforms = transforms
        self.fasttext = get_fasstext_vector
        self.text_emb_size = text_emb_size 


    def __getitem__(self, index):
        obj = self.data.iloc[index]
        img_emb = self.transforms(np.array(self.image_array[str(obj['_id'])]))
        try:
            txt_emb =  self.fasttext(obj['combined_name_and_breadcrumbs'])
        except:
            txt_emb = torch.zeros(self.text_emb_size)
        instance_labels = torch.tensor(obj["instance_labels"].tolist())    
        return img_emb,txt_emb,instance_labels
            
    def __len__(self):
        return len(self.data)