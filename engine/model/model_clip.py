import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_scheduler

class ClipConcateModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, self.num_labels),
            )
    def forward(self, embeddings):
        text_embeddings = embeddings[0]
        image_embeddings = embeddings[1]
        # mid_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        # logits = self.classifier(mid_embeddings)
        logits = self.classifier(image_embeddings)
        return logits 