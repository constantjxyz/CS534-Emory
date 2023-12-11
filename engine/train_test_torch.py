import torch
from torch import *
import numpy as np
from transformers import AutoModel, AutoTokenizer, get_scheduler, BertForSequenceClassification
from engine.model.model_bert import TextBertModel, SentenceBertFrozenModel
from engine.model.model_clip import ClipConcateModel
from engine.model.model_lstm import LSTMModel, RNNModel
import time
import sklearn
from sklearn import *

def run_train_test_torch(train_dataset, valid_dataset, test_dataset, **params):
    # set the dataloader of train_set, valid_set, and test_set
    y_train, y_valid, y_test = train_dataset.get_labels(), valid_dataset.get_labels(), test_dataset.get_labels()
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(params['batch_size']), 
    sampler=torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight)),
    # sampler=torch.utils.data.RandomSampler
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=int(params['batch_size']), shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(params['batch_size']), shuffle=False)
    print(len(test_dataset))
    print('length of training sest', len(train_dataset))
    
    # set device, choose model and optimizer
    print(params)
    device = params['device']
    method = params['method']
    learning_rate = float(params['learning_rate'])
    epochs = np.int(params['epochs'])
    weight_decay = float(params['weight_decay'])
    warmup_steps = int(params['warmup_steps'])
    t_total = len(train_dataloader) * epochs
    max_sequence_length = int(params['max_sequence_length'])
    
    # assert method in ['lstm']
    if method == 'bert':
        model = TextBertModel(num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif method == 'sentence_bert_frozen':
        model = SentenceBertFrozenModel(num_labels=2).to(device)
    elif method == 'clip_concate':
        model = ClipConcateModel(num_labels=2).to(device)
    elif method == 'lstm':
        model = LSTMModel(num_labels=2, sequence_length=20, input_dim=train_dataset[0][0].shape[1]).to(device)
    elif method == 'rnn':
        model = RNNModel(num_labels=2, sequence_length=20, input_dim=train_dataset[0][0].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_scheduler(name="cosine", optimizer=optimizer,num_warmup_steps=warmup_steps, num_training_steps=t_total)
    criterion = torch.nn.CrossEntropyLoss()
    # training and validation
    training_start_time = time.time()
    for epoch in np.arange(0, epochs):
        print('-'*10, 'epoch:', epoch, '_'*10)
        print('-'*10, 'training', '_'*10)
        model.train()
        epoch_total_loss = 0
        for step, batch in enumerate(train_dataloader):       
            b_text, b_labels = batch  
            if method == 'bert':          
                b_inputs = tokenizer(list(b_text), truncation=True, max_length=max_sequence_length, return_tensors="pt", padding=True)
                b_labels = b_labels.to(device)
                b_inputs = b_inputs.to(device)
            elif method in ['sentence_bert_frozen', 'lstm', 'rnn']:
                b_inputs = b_text
                b_labels = b_labels.to(device)
                b_inputs = b_inputs.to(device)
            elif method in ['clip_concate']:
                b_labels = b_labels.to(device)
                b_inputs = b_text
                b_inputs = [inputs.to(device) for inputs in b_inputs]

            model.zero_grad()        
            b_logits = model(b_inputs)
            loss = criterion(b_logits, b_labels)
            epoch_total_loss += loss.item()
            # Perform a backward pass to calculate the gradients
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_loss = epoch_total_loss/len(train_dataloader)
        print('epoch =', epoch)
        print('epoch_loss =', epoch_total_loss)
        print('avg_epoch_loss =', avg_loss)
        print('learning rate =', optimizer.param_groups[0]["lr"])
        print('time =', time.time()-training_start_time)
        
        model.eval()
        epoch_total_loss = 0
        all_probs = []
        all_labels = []
        all_predict_labels = []
        for step, batch in enumerate(valid_dataloader):        
            b_text, b_labels = batch           
            if method == 'bert':          
                b_inputs = tokenizer(list(b_text), truncation=True, max_length=max_sequence_length, return_tensors="pt", padding=True)
                b_labels = b_labels
                b_inputs = b_inputs.to(device)
            elif method in ['sentence_bert_frozen', 'lstm', 'rnn']:
                b_inputs = b_text
                b_labels = b_labels
                b_inputs = b_inputs.to(device)
            elif method in ['clip_concate']:
                b_labels = b_labels
                b_inputs = b_text
                b_inputs = [inputs.to(device) for inputs in b_inputs]

            with torch.no_grad():        
                b_logits = model(b_inputs)
                b_logits = b_logits.detach().cpu()
            loss = criterion(b_logits, b_labels)
            epoch_total_loss += loss.item()
            probs = torch.nn.functional.softmax(b_logits, dim=1)
            predicted_probs = probs[:, 1].detach().numpy()
            predicted_labels = torch.argmax(probs, dim=1).detach().numpy()
            all_predict_labels.extend(predicted_labels)
            all_probs.extend(predicted_probs)
            all_labels.extend(list(b_labels.numpy()))
        avg_loss = epoch_total_loss/len(valid_dataloader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_predict_labels = np.array(all_predict_labels)
        roc_score = sklearn.metrics.roc_auc_score(all_labels, all_probs)
        accuracy = sklearn.metrics.accuracy_score(all_labels, all_predict_labels)
        f1_score = sklearn.metrics.f1_score(all_labels, all_predict_labels, average='weighted')
        print('-'*10, 'validation', '_'*10)
        print('epoch =', epoch)
        print('epoch_loss =', epoch_total_loss)
        print('avg_epoch_loss =', avg_loss)
        print('validation roc:', roc_score)
        print('validation_acc:', accuracy)
        print('validation_f1_score:', f1_score)
        print('time =', time.time()-training_start_time)
        
    # testing evaluation 
    model.eval()
    epoch_total_loss = 0
    all_probs = []
    all_labels = []
    all_predict_labels = []
    for step, batch in enumerate(test_dataloader):        
        if method == 'bert':          
            b_inputs = tokenizer(list(b_text), truncation=True, max_length=max_sequence_length, return_tensors="pt", padding=True)
            b_labels = b_labels
            b_inputs = b_inputs.to(device)
        elif method in ['sentence_bert_frozen', 'lstm', 'rnn']:
            b_inputs = b_text
            b_labels = b_labels
            b_inputs = b_inputs.to(device)
        elif method in ['clip_concate']:
            b_labels = b_labels
            b_inputs = b_text
            b_inputs = [inputs.to(device) for inputs in b_inputs]

        with torch.no_grad():        
            b_logits = model(b_inputs)
            b_logits = b_logits.detach().cpu()
        loss = criterion(b_logits, b_labels)
        epoch_total_loss += loss.item()
        probs = torch.nn.functional.softmax(b_logits, dim=1)
        predicted_probs = probs[:, 1].detach().numpy()
        predicted_labels = torch.argmax(probs, dim=1).detach().numpy()
        all_predict_labels.extend(predicted_labels)
        all_probs.extend(predicted_probs)
        all_labels.extend(list(b_labels.numpy()))
    avg_loss = epoch_total_loss/len(valid_dataloader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_predict_labels = np.array(all_predict_labels)
    roc_score = sklearn.metrics.roc_auc_score(all_labels, all_probs)
    accuracy = sklearn.metrics.accuracy_score(all_labels, all_predict_labels)
    f1_score = sklearn.metrics.f1_score(all_labels, all_predict_labels, average='weighted')
    print('-'*10, 'test', '_'*10)
    print('epoch =', epoch)
    print('epoch_loss =', epoch_total_loss)
    print('avg_epoch_loss =', avg_loss)
    print('test roc:', roc_score)
    print('test_acc:', accuracy)
    print('test_f1_score:', f1_score)
    print('time =', time.time()-training_start_time)

        