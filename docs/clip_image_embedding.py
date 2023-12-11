import numpy as np
import pandas as pd
import os
import json
import sklearn
import transformers
import time


data = pd.read_csv('./dataset/basic_info/all.csv')
data.head()
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
start = time.time()
start_id = 130000
end_id = 150000
print(f'start: {start_id}, end: {end_id}')
embeddings = []
for idx in range(start_id, end_id, 500):
    if idx+500 < 150000:
        inputs = processor(images=[Image.open(path) for path in data['image_path'].iloc[idx:(idx+500)]], return_tensors="pt")
    else:
        inputs = processor(images=[Image.open(path) for path in data['image_path'].iloc[idx:]], return_tensors="pt") 
    inputs = inputs.to('cuda')
    t = model.get_image_features(**inputs).detach().cpu().numpy()
    print(idx, idx+500)
    embeddings.append(t)
final_file = './dataset/embedding/clip_image/clip_image_start_' + str(start_id) + '_end_' +str(end_id) + '.npy'
np.save(final_file, np.array(embeddings))
print(final_file)
end = time.time()
print(f'finished time: {end - start}')

