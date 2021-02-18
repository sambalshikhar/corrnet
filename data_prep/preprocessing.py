import h5py
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
from glob import glob
from shutil import copyfile
import numpy as np
import pandas as pd

class create_hdf5():
    def __init__(self):
        print("Creating hdf5 files")
        self.invalid_images=[]
    def resize_image(self,path):
        try:
            image = Image.open(path)
            image = image.resize((224,224),Image.ANTIALIAS)
            image = np.asarray(image)
            return image
        except:
            self.invalid_images.append(path)
            pass    

    def read_from_hdf5(self,id):
        global image_array
        return np.array(image_array[id])

    def write_images_into_HDF5(self,images_path,file_path):
        HEIGHT = 224
        WIDTH = 224
        CHANNELS = 3
        names = [x.split("/")[-1].replace(".jpg","") for x in images_path.tolist()]
        # names = [x for x in images_path]
        batch = 512
        with h5py.File(file_path, 'w') as hf: 
            for i in tqdm(range(0,len(images_path),batch)):
                pool = ThreadPool(48)
                return_images = pool.map(self.resize_image,images_path[i:i+batch])
                pool.close()
                pool.join()
                for image,name in zip(return_images,names[i:i+batch]):
                    hf.create_dataset(
                    name=name,
                    data=image,
                    shape=(HEIGHT, WIDTH, CHANNELS),
                    maxshape=(HEIGHT, WIDTH, CHANNELS),
                    compression="gzip",
                    compression_opts=9)

        return self.invalid_images            
        
                    

        









