config={

#Hyperparameters   
"text_emb_size":300,
"batch_size":100,    
"train_hdf5_name":"myntra_training.h5",
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
"optimizer":"adam",
"scheduler":False,

#Files and directories
"source_dir":"/content/nfs/machine-learning/myntra_dataset/",
"train_csv_path":"/content/nfs/machine-learning/myntra_dataset/train.csv",
"test_csv_path":"/content/nfs/machine-learning/myntra_dataset/test.csv",
"fasttext_path":"cc.en.300.bin",
"model_name":"clip_demo_2.pth",
"hdf5_exists":True,
"load_pretrained":False,
"evaluate_n_results":10,

#wandb setup
"experiment_name":"clip_training_1",
"project_name":"clip-training",
"username":"sambal_123",
"API_KEY":"e18504279798c1429b9fc418d5cbf98056aa8da4"

}