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


def create_emb_pool(corrnet_model,resnet_model,testLoader,):
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

def get_text2img_retrievals(corrnet_model,resnet_model,only_image_emb_pool,n_random_indices,text):

    for i in n_random_indices:
        #query = 'brush geschirrbürste saisonale kollektion frühling brabantia brush geschirrbürste'
        
        print("Query:",text[i])
        print()
        # retrieve
        query=text[i]
        query_text_emb = torch.tensor(get_fasstext_vector(query)).unsqueeze(0).detach().cuda()
        query_image_emb = torch.zeros((1, 2048)).cuda()

        #print(query_text_emb.shape, query_image_emb.shape)
        corrnet_model.eval()
        with torch.no_grad():
            _, _,query_vector,_,_,_ = corrnet_model(query_image_emb,query_text_emb.cuda())
            query_vector=F.normalize(query_vector, p=2, dim=1)
            query_vector=query_vector.detach().cpu().numpy()


        distance_matrix = cdist(query_vector,only_image_emb_pool)
        sorted_distance_matrix = np.argsort(distance_matrix)
        print(sorted_distance_matrix[0])
        retrieved_index =sorted_distance_matrix[0][:10]
        print(retrieved_index)



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


    corrnet_model.eval()
    resnet_model.eval()

    common_emb_pool,only_image_emb_pool,only_text_emb_pool=create_emb_pool(corrnet_model,resnet_model,testLoader)

    text = df_test.combined_name_and_breadcrumbs.values
    product_id = df_test._id.values

    id_x=[i for i in range(len(product_id))]

    n_random_indices=random.sample(idx,config['evaluate_n_results'])







    











