config={

#Hyperparameters   
"text_emb_size":300,
"batch_size":100,    
"train_hdf5_name":"myntra_trainin.h5",
"test_hdf5_name":"myntra_testing.h5",
"image_emb_size":2048,
"text_emb_size":300,
"middle_emb_size":1024,
"lamda":0.05,
"lr":1e-3,
"decayRate":0.99,
'n_epochs':200,
'print_freq':50,
"loss_function":"clip",
"fasttext_path":"cc.en.300.bin",
"optimizer":"adam",

#Files and directories
"source_dir":"/content/nfs/machine-learning/myntra_dataset/",
"train_csv_path":"/content/nfs/machine-learning/myntra_dataset/train.csv",
"test_csv_path":"/content/nfs/machine-learning/myntra_dataset/test.csv",
"hdf5_exists":True,
"load_pretrained":False

}