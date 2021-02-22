import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50
import torch.nn.functional as F
import random
from annoy import AnnoyIndex
import os
import matplotlib.pyplot as plt
import argparse
from tqdm.notebook import tqdm
import numpy as np


def create_emb_pool(corrnet_model,resnet_model,testLoader):
    common_emb_pool = []
    only_image_emb_pool = []
    only_text_emb_pool = []
    with torch.no_grad():
        for (image_emb, text_emb,_) in tqdm(testLoader):
            if torch.cuda.is_available():
                image_emb = image_emb.cuda()
                text_emb  = text_emb.cuda()

            resnet_model.eval()    
            resnet_out = resnet_model(image_emb)
            corrnet_model.eval()
            common_emb, only_image_emb, only_text_emb, _, _, _ = corrnet_model(resnet_out, text_emb)

            only_image_emb, only_text_emb=F.normalize(only_image_emb, p=2, dim=1),F.normalize(only_text_emb, p=2, dim=1)

            common_emb_pool.append(common_emb.detach().cpu().numpy())
            only_image_emb_pool.append(only_image_emb.cpu().numpy())
            only_text_emb_pool.append(only_text_emb.cpu().numpy())

    common_emb_pool = np.vstack(common_emb_pool)
    only_image_emb_pool = np.vstack(only_image_emb_pool)
    only_text_emb_pool = np.vstack(only_text_emb_pool)

    common_emb_pool.shape, only_image_emb_pool.shape, only_text_emb_pool.shape

    print("Pool created")

    return common_emb_pool,only_image_emb_pool,only_text_emb_pool

def get_retrievals(loader,df,annoy_index,corrnet_model,resnet_model,df_test,testing_image_array,retrieval_type):
        text = df.combined_name_and_breadcrumbs.values
        id = df._id.values

        full_id=df_test._id.values
        full_text=df_test.combined_name_and_breadcrumbs.values

        for i,(image_emb, text_emb,_) in enumerate(loader):        
            print("Query")
            p = np.array(testing_image_array[str(id[i])])
            plt.title(text[i])
            plt.imshow(p)
            plt.show()
            
            corrnet_model.eval()
            resnet_model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    image_emb = image_emb.cuda()
                    text_emb  = text_emb.cuda()
                resnet_out = torch.unsqueeze(resnet_model(image_emb),0)
                common_emb, only_image_emb, only_text_emb, _, _, _ = corrnet_model(resnet_out, text_emb)
                common_emb,only_image_emb,only_text_emb=F.normalize(common_emb, p=2, dim=0),F.normalize(only_image_emb, p=2, dim=0),F.normalize(only_text_emb, p=2, dim=0)

            common_emb=common_emb.detach().cpu().numpy()
            only_image_emb=only_image_emb.cpu().numpy()
            only_text_emb=only_text_emb.cpu().numpy()
            
            if retrieval_type=='txt2img':

                query_vector=only_text_emb
                retrieved_index = annoy_index.get_nns_by_vector(query_vector[0], 11)[1:]

            elif retrieval_type=='img2txt': 
                query_vector=only_image_emb   
                retrieved_index = annoy_index.get_nns_by_vector(query_vector[0], 11)[1:]

            else:
                query_vector=common_emb
                retrieved_index = annoy_index.get_nns_by_vector(query_vector[0], 10)[1:]

            print('retrieved_index:', retrieved_index)

            for r_ix in retrieved_index:
                p = np.array(testing_image_array[str(full_id[r_ix])])
                print("Name:",full_text[r_ix])
                plt.imshow(p)
                plt.show()
                print()  
            print("******************")


def create_annoy_index(common_emb_pool,only_image_emb_pool,only_text_emb_pool):

    annoy_common = AnnoyIndex(1024, 'angular') 
    annoy_image = AnnoyIndex(1024, 'angular') 
    annoy_text = AnnoyIndex(1024, 'angular') 

    for ix in tqdm(range(len(common_emb_pool))):

        IT = common_emb_pool[ix]
        I_ = only_image_emb_pool[ix]
        _T = only_text_emb_pool[ix]

        annoy_common.add_item(ix, IT)
        annoy_image.add_item(ix, I_)
        annoy_text.add_item(ix, _T)
        
    annoy_common.build(10) # 10 trees
    annoy_image.build(10) # 10 trees
    annoy_text.build(10) # 10 trees

    

    if not os.path.isdir("./annoy_indices"):
        os.mkdir("./annoy_indices")
        annoy_common.save('./annoy_indices/common.ann')
        annoy_image.save('./annoy_indices/image.ann')
        annoy_text.save('./annoy_indices/text.ann')

    print("Annoy Indices created")    
