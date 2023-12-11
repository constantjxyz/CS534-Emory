# import gensim.downloader as api
# import time
# start_time = time.time()
# model = api.load("glove-twitter-25")
# sentence = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
#  ['this', 'is', 'the', 'second', 'sentence'],]
# t = model['sentence']
import torch
import torchtext
import time
import numpy as np
# from transformers import AutoTokenizer
# from transformers import PreTrainedTokenizer
from torchtext.data import get_tokenizer
start_time = time.time()
glove = torchtext.vocab.GloVe(name='twitter.27B', dim=200)
max_sequence=100
all_texts = np.load('./dataset/basic_info/all_text.npy', allow_pickle=True)
tokenizer = get_tokenizer('moses')
all_tensors = []
start_id = 0
end_id = len(all_texts)
for idx in range(start_id, end_id):
    inputs = tokenizer(all_texts[idx])[:max_sequence]   #truncation
    tensor = glove.get_vecs_by_tokens(inputs, True)
    all_tensors.append(tensor)
    if idx % 10000 == 0:
        print(f'{idx} finished, running time {time.time()-start_time}s')
from torch.nn.utils.rnn import pad_sequence
all_tensors = pad_sequence(all_tensors, batch_first=True, padding_value=0)
final_file = './dataset/embedding/glove_200/glove_text_start_' + str(start_id) + '_end_' +str(end_id) + '.pt'
torch.save(all_tensors, final_file)

print('ok')
end_time = time.time()
print(f'finished, running time {end_time-start_time}s')


