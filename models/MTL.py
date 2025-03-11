import os
import torch
from torch import nn
from typing import List

from models.sentence_transformer import CustomSentenceTransformer
from models.classifier_head import ClassifierHead
from models.ner_head import NerHead

class MTL(nn.Module):
    def __init__(self,
                 config:dict,
                 device:str) -> None:
        
        super(MTL, self).__init__()
        
        self.sentenceTransformer = CustomSentenceTransformer(config, device)
        self.classifier_head     = ClassifierHead(config)
        self.ner_head            = NerHead(config)
        self.tokenizer           = self.sentenceTransformer.tokenizer

        # load pre-trained model of encoder (BERT)
        if os.path.isfile(config["model"]["sentence_transformer_pretrained"]):
            self.sentenceTransformer.load_state_dict(torch.load(config["model"]["sentence_transformer_pretrained"]))
            print("Encoder Weight Loaded: ", config["model"]["sentence_transformer_pretrained"])
    
    def forward(self, input_text:List[str]):
        sentence_embedded, encoder_embedding = self.sentenceTransformer(input_text) # Output sentence embedding Output Shape: [batch_size, output_dim]
        output_cls = self.classifier_head(sentence_embedded)            # Classification head Output Shape: [batch_size, n_classes]
        output_ner = self.ner_head(encoder_embedding)                   # NER head outputs classes per token

        return output_cls, output_ner