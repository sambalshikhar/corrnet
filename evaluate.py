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

from scipy.spatial.distance import cdist

from data_prep.preprocessing import *
from dataloaders.loaders import *
from models.model import *
from utils.embed_utils import *
from utils.trainer_utils import *
from trainers import *
from trainers.losses import *

from config import config


def load_existing_data(source):
    copy_files_h5 = glob(source+"*.h5")
    copy_file = copy_files_h5 
    _=[copyfile(x, '/content/corrnet/{}'.format(x.split('/')[-1])) for x in copy_file] 


def create_emb_pool(corrnet_model,resnet_model,testLoader):
    common_emb_pool = []
    only_image_emb_pool = []
    only_text_emb_pool = []
    with torch.no_grad():
        for (image_emb, text_emb,_) in tqdm.notebook.tqdm(testLoader):
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

def get_retrievals(loader,n_matches,df):
        text = df.combined_name_and_breadcrumbs.values
        id = df._id.values
        for i,(image_emb, text_emb,_) in enumerate(testLoader):        
            print("Query:"
            p = np.array(testing_image_array[str(id[r_ix])])
            plt.title(text[i])
            plt.imshow(p)
            print("******************")
            
            corrnet_model.eval()
            resnet_model.eval()
            with torch.no_grad():
                resnet_out = resnet_model(image_emb)
                common_emb, only_image_emb, only_text_emb, _, _, _ = corrnet_model(resnet_out, text_emb)
                common_emb,only_image_emb,only_text_emb=F.normalize(common_emb, p=2, dim=1),F.normalize(only_image_emb, p=2, dim=1),F.normalize(only_text_emb, p=2, dim=1)
            
            retrieved_index = annoy.get_nns_by_vector(query_vector[0], 10)
            print('retrieved_index:', retrieved_index)

def create_annoy_index(common_emb_pool,only_image_emb_pool,only_text_emb_pool):

    annoy_common = AnnoyIndex(1024, 'angular') 
    annoy_image = AnnoyIndex(1024, 'angular') 
    annoy_text = AnnoyIndex(1024, 'angular') 

    # index flair vectors

    for ix in tqdm.notebook.tqdm(range(len(common_emb_pool))):

        IT = common_emb_pool[ix]
        I_ = only_image_emb_pool[ix]
        _T = only_text_emb_pool[ix]

        annoy_common.add_item(ix, IT)
        annoy_image.add_item(ix, I_)
        annoy_text.add_item(ix, _T)
        
    annoy_common.build(10) # 10 trees
    annoy_image.build(10) # 10 trees
    annoy_text.build(10) # 10 trees

    

    if not os.isdir("./annoy_indices")
        os.mkdir("./annoy_indices")
        annoy_common.save('./annoy_indices/common.ann')
        annoy_image.save('./annoy_indices/image.ann')
        annoy_text.save('./annoy_indices/text.ann')


if __name__ == '__main__':

    df_test=pd.read_csv(config['test_csv_path'])

    transformations = transforms.Compose([torchvision.transforms.ToPILImage(mode="RGB"),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
    source=config['source_dir']
    load_existing_data(source)
    print("Hdf5 loaded") 

    get_fasstext_vector = Fasttext(fasttext_model)

    text_emb_size = config['text_emb_size']
    testing_image_array = h5py.File(train_config['test_hdf5_name'],'r')
    test_ds = CustomDataset(df_test,testing_image_array,transformations,get_fasstext_vector,text_emb_size)
    testLoader = DataLoader(test_ds, shuffle=False, batch_size=train_config['batch_size'], pin_memory = torch.cuda.is_available())

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
        corrnet_model.load_state_dict(torch.load(os.path.join(train_config['source_dir'],config['model_name'])))

    resnet_model = Resnet50()

    if torch.cuda.is_available():
        corrnet_model,resnet_model = corrnet_model.cuda(),resnet_model.cuda()

    common_emb_pool,only_image_emb_pool,only_text_emb_pool=create_emb_pool(corrnet_model,resnet_model,testLoader)

    create_annoy_index(common_emb_pool,only_image_emb_pool,only_text_emb_pool)

    retreival_df=df_test.sample(config['evaluate_n_results'])

    test_df_eval = CustomDataset(retreival_df,testing_image_array,transformations,get_fasstext_vector,text_emb_size)
    retreival_loader = DataLoader(test_ds_eval, shuffle=False, batch_size=1, pin_memory = torch.cuda.is_available())

    retreival_indexes=get_retrievals(retreival_loader,test_df_eval)















    











