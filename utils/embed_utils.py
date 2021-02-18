import random
import pickle 
import torch
import fasttext 
from config import config 
import torch
import torch.nn as nn

print("***Loading fasttext***")
fasttext_model = fasttext.load_model(config['fasttext_path'])
print("***Fasttext Loaded****")


class Fasttext(nn.Module):
    def __init__(self, fasttext_model):
        super(Fasttext,self).__init__()
        self.model = fasttext_model
        print("Fasttext Loaded")

    def forward(self, sentence):
        vector_sent = self.model.get_sentence_vector(sentence)
        # vector_split = np.mean([self.model.get_word_vector(word) for word in sentence.split(" ")],axis=0)
        return torch.tensor(vector_sent)