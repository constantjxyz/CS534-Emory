import numpy as np
import pandas as pd
import os
import json
import sklearn
import transformers
import time

from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

start = time.time()
start_id = 130000
end_id = 150000
print(f'start: {start_id}, end: {end_id}')
embeddings = []
all_texts = np.load('all_text.npy', allow_pickle=True)
interval = 2000
for idx in range(start_id, end_id, interval):
    if idx+2000 < 150000:
        inputs = tokenizer(all_texts[idx:(idx+interval)].tolist(), padding=True, truncation=True, return_tensors="pt")
    else:
        inputs = tokenizer(all_texts[idx:].tolist(), padding=True, truncation=True, return_tensors="pt") 
    inputs = inputs
    t = model.get_text_features(**inputs).detach().cpu().numpy()
    print(idx, idx+interval)
    embeddings.append(t)
final_file = './dataset/embedding/clip/clip_text_start_' + str(start_id) + '_end_' +str(end_id) + '.npy'
np.save(final_file, np.array(embeddings))
print(final_file)
end = time.time()
print(f'finished time: {end - start}')