import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_scheduler
from sentence_transformers import SentenceTransformer

class TextBertModel(nn.Module):
    def __init__(self, num_labels, text_pretrained='bert-base-uncased'):
        # text_pretrained: 'bert-base-uncased', 'deepset/sentence_bert', 'distilbert-base-uncased'
        super().__init__()
        self.num_labels = num_labels
        self.text_encoder = AutoModel.from_pretrained(text_pretrained)
        self.classifier = nn.Linear(
            self.text_encoder.config.hidden_size, num_labels)
        
    
    def forward(self, text):
        output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        logits = self.classifier(output.last_hidden_state[:, 0, :]) # CLS embedding
        return logits
    
class SentenceBertFrozenModel(nn.Module):
    def __init__(self, num_labels, max_sequence_length=500,text_pretrained='sentence-transformers/bert-base-nli-mean-tokens'):
        # text_pretrained: 'bert-base-uncased', 'deepset/sentence_bert', 'distilbert-base-uncased'
        super().__init__()
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(),
            nn.Linear(256, self.num_labels),
            )
    def forward(self, text_embeddings):
        logits = self.classifier(text_embeddings)
        return logits
    
class SentenceBertModel(nn.Module):
    def __init__(self, num_labels,text_pretrained='all-mpnet-base-v2'):
        # text_pretrained: 'bert-base-uncased', 'deepset/sentence_bert', 'distilbert-base-uncased'
        super().__init__()
        self.num_labels = num_labels
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(),
            nn.Linear(256, self.num_labels),
            )
    def forward(self, text_embeddings):
        logits = self.classifier(text_embeddings)
        return logits  