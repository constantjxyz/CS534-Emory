# import dependencies from module
import argparse
import time
import torch
import os
#import wandb
import numpy as np
from setting.get_parameter import get_param_dict
from dataset.create_dataset import AllTextDataset, OriTextDataset
from engine.train_test_sklearn import run_train_test_sklearn
#from engine.train_test_torch import run_train_test_torch
import os
import logging
print("Current Working Directory:", os.getcwd())

# set parameters from csv
parser = argparse.ArgumentParser()
# parser.add_argument('--setting_dir', type=str, default='./setting/svm/svm_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/adaboost/adaboost_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/decision_tree/decision_tree_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/logistic_regression/logistic_regression_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/naive_bayes/naive_bayes_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/random_forest/random_forest_template.csv')
parser.add_argument('--setting_dir', type=str, default='./setting/mlp/mlp_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/rnn/rnn_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/bert/bert_template.csv.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/clip/clip_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/lstm/lstm_template.csv')
# parser.add_argument('--setting_dir', type=str, default='./setting/sentence_bert/sentence_bert_template.csv.csv')

args = parser.parse_args()
params = get_param_dict(args.setting_dir)
random_state = int(params['rand_seed'])
torch.manual_seed(random_state)
np.random.seed(random_state)
torch.cuda.manual_seed_all(random_state)
# if params['gpu'] == '-1':
#     params['device'] = torch.device('cpu')
# else:
#     params['device'] = torch.device('cuda:'+str(params['gpu']) if torch.cuda.is_available() else 'cpu')
# params['device']='cuda'
    
# write down the experiments by using wandb
# if params['use_wandb'] == 'True':
#     wandb.init(project='mlproject', name=params['ckpt'], entity="yuzhang_siyu")

# logging.basicConfig(filename='program_output_svm.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_adaboost.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_dt.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_lr.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_nb.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_rf.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
logging.basicConfig(filename='program_output_mlp.log', 
                    level=logging.INFO, 
                    format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_rnn.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_bert.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_clip.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_lstm.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')
# logging.basicConfig(filename='program_output_sentence_bert.log', 
#                     level=logging.INFO, 
#                     format='%(levelname)s:%(message)s')

# load the dataset
def main():
    assert params['text_modality'] in ['embedding', 'ori'], 'Wrong text_modality in the setting file'
    if params['text_modality'] == 'embedding':
        train_dataset = AllTextDataset(mode='train', **params)
        valid_dataset = AllTextDataset(mode='valid', **params)
        test_dataset = AllTextDataset(mode='test', **params)  
    elif params['text_modality'] == 'ori':
        train_dataset = OriTextDataset(mode='train', **params)
        valid_dataset = OriTextDataset(mode='valid', **params)
        test_dataset = OriTextDataset(mode='test', **params)  
    if params['module'] == 'sklearn':
        run_train_test_sklearn(train_dataset, valid_dataset, test_dataset, logging, **params)
    # elif params['module'] == 'torch': 
    #     #run_train_test_torch(train_dataset, valid_dataset, test_dataset, **params)
    print('ok')
    

# execute main function
if __name__ == '__main__':
    
    #logging.info('Start')
    print('Start')
    # logging.info(f"Current Working Directory: {os.getcwd()}")
    # logging.info(f"Setting file: {args.setting_dir}")
    start = time.time()
    print('setting file:', args.setting_dir)
    main()
    end = time.time()
    print(f'Program end, total running time {end - start} s')
    logging.info(f'Program end, total running time {end - start} s')

print('ok')
    
