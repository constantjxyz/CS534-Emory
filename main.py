# import dependencies from module
import argparse
import time
import torch
import os
import wandb
import numpy as np
from setting.get_parameter import get_param_dict
from dataset.create_dataset import AllTextDataset, OriTextDataset, TextImageDataset
from engine.train_test_sklearn import run_train_test_sklearn
from engine.train_test_torch import run_train_test_torch

# set parameters from csv
parser = argparse.ArgumentParser()
parser.add_argument('--setting_dir', type=str, default='./setting/rnn/rnn_template.csv')
args = parser.parse_args()
params = get_param_dict(args.setting_dir)
random_state = int(params['rand_seed'])
torch.manual_seed(random_state)
np.random.seed(random_state)
torch.cuda.manual_seed_all(random_state)
print(f'random state: {random_state}')
# if params['gpu'] == '-1':
#     params['device'] = torch.device('cpu')
# else:
#     params['device'] = torch.device('cuda:'+str(params['gpu']) if torch.cuda.is_available() else 'cpu')
params['device']='cuda'
# params['device'] = 'cpu'
    
# write down the experiments by using wandb
if params['use_wandb'] == 'True':
    wandb.init(project='mlproject', name=params['ckpt'], entity="yuzhang_siyu")

# load the dataset
def main():
    assert params['text_modality'] in ['embedding', 'ori', 'text_image_embedding'], 'Wrong text_modality in the setting file'
    if params['text_modality'] == 'embedding':
        train_dataset = AllTextDataset(mode='train', **params)
        valid_dataset = AllTextDataset(mode='valid', **params)
        test_dataset = AllTextDataset(mode='test', **params)  
    elif params['text_modality'] == 'ori':
        train_dataset = OriTextDataset(mode='train', **params)
        valid_dataset = OriTextDataset(mode='valid', **params)
        test_dataset = OriTextDataset(mode='test', **params) 
    elif params['text_modality'] == 'text_image_embedding':
        train_dataset = TextImageDataset(mode='train', **params)
        valid_dataset = TextImageDataset(mode='valid', **params)
        test_dataset = TextImageDataset(mode='test', **params)   
    if params['module'] == 'sklearn':
        run_train_test_sklearn(train_dataset, valid_dataset, test_dataset, **params)
    elif params['module'] == 'torch': 
        run_train_test_torch(train_dataset, valid_dataset, test_dataset, **params)
    print('ok')
    

# execute main function
if __name__ == '__main__':
    print('Start')
    start = time.time()
    print('setting file:', args.setting_dir)
    main()
    end = time.time()
    print(f'Program end, total running time {end - start} s')

print('ok')
    
