'''
create the dataset for whole training process
'''
# import dependent modules
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

class AllTextDataset(Dataset):
    def __init__(self, mode='all', **params):
        self.embedding_dir = params['embedding_dir']
        assert os.path.exists(self.embedding_dir), 'Wrong embedding directory'
        embeddings = np.load(self.embedding_dir, allow_pickle=True) 
        self.embeddings = embeddings['embeddings']
        self.labels = embeddings['labels']
        self.splits = embeddings['splits']
        if 'keywords' in embeddings.files:
            self.keywords = embeddings['keywords']
        else:
            self.keywords = False
        assert mode in ['all', 'train', 'valid', 'test', 'train-valid'], 'Wrong mode setting'
        if mode == 'train' or mode == 'valid' or mode == 'test':
            matching_indices = np.where(self.splits==mode)[0]
            self.labels = self.labels[matching_indices]
            self.embeddings = self.embeddings[matching_indices]
        elif mode == 'train-valid':
            matching_indices = np.where((self.splits=='train') | (self.splits == 'valid'))[0]
            self.labels = self.labels[matching_indices]
            self.embeddings = self.embeddings[matching_indices]
        if mode == 'train' and params['dataset'] == 'small':
            self.labels = self.labels[-10000:]
            self.embeddings = self.embeddings[-10000:]
        assert len(self.embeddings) == len(self.labels), 'Mismatch embeddings and labels'
        print('ok')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        index = np.int(index)
        return self.embeddings[index], self.labels[index]
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_labels(self):
        return self.labels
    
    def get_keywords(self):
        return self.keywords

class OriTextDataset(Dataset):
    def __init__(self, mode='all', **params):
        self.embedding_dir = params['embedding_dir']
        assert os.path.exists(self.embedding_dir), 'Wrong embedding directory'
        embeddings = np.load(self.embedding_dir, allow_pickle=True) 
        self.texts = embeddings['texts']
        self.labels = embeddings['labels']
        self.splits = embeddings['splits']
        assert mode in ['all', 'train', 'valid', 'test', 'train-valid'], 'Wrong mode setting'
        if mode == 'train' or mode == 'valid' or mode == 'test':
            matching_indices = np.where(self.splits==mode)[0]
            self.labels = self.labels[matching_indices]
            self.texts = self.texts[matching_indices]
        elif mode == 'train-valid':
            matching_indices = np.where((self.splits=='train') | (self.splits == 'valid'))[0]
            self.labels = self.labels[matching_indices]
            self.texts = self.texts[matching_indices]
        if params['dataset'] == 'small':
            self.labels = self.labels[-10000:]
            self.texts = self.texts[-10000:]
        assert len(self.texts) == len(self.labels), 'Mismatch embeddings and labels'
        print('ok')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        index = np.int(index)
        return self.texts[index], self.labels[index]
    
    def get_labels(self):
        return self.labels
    
    def get_texts(self):
        return self.texts
    
class TextImageDataset(Dataset):
    def __init__(self, mode='all', **params):
        self.embedding_dir = params['embedding_dir']
        assert os.path.exists(self.embedding_dir), 'Wrong embedding directory'
        embeddings = np.load(self.embedding_dir, allow_pickle=True) 
        self.text_embeddings = embeddings['text_embeddings']
        self.image_embeddings = embeddings['image_embeddings']
        self.labels = embeddings['labels']
        self.splits = embeddings['splits']
        if 'keywords' in embeddings.files:
            self.keywords = embeddings['keywords']
        else:
            self.keywords = False
        assert mode in ['all', 'train', 'valid', 'test', 'train-valid'], 'Wrong mode setting'
        if mode == 'train' or mode == 'valid' or mode == 'test':
            matching_indices = np.where(self.splits==mode)[0]
            self.labels = self.labels[matching_indices]
            self.text_embeddings = self.text_embeddings[matching_indices]
            self.image_embeddings = self.image_embeddings[matching_indices]
            
        elif mode == 'train-valid':
            matching_indices = np.where((self.splits=='train') | (self.splits == 'valid'))[0]
            self.labels = self.labels[matching_indices]
            self.text_embeddings = self.text_embeddings[matching_indices]
            self.image_embeddings = self.image_embeddings[matching_indices]
        
        if mode == 'train' and params['dataset'] == 'small':
            self.labels = self.labels[-10000:]
            self.text_embeddings = self.text_embeddings[-10000:]
            self.image_embeddings = self.image_embeddings[-10000:]
        assert len(self.text_embeddings) == len(self.labels), 'Mismatch embeddings and labels'
        print('ok')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        index = np.int(index)
        
        return (self.text_embeddings[index], self.image_embeddings[index]), self.labels[index]
    
    # def get_embeddings(self):
    #     return self.embeddings
    
    def get_labels(self):
        return self.labels
    
    def get_keywords(self):
        return self.keywords 