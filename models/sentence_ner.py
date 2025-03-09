from torch import nn

import os
import torch
from torch import nn, Tensor
from typing import List

from models.sentence_transformer import CustomSentenceTransformer

class NER(nn.Module):
    def __init__(self,
                 backbone_model:str,
                 n_classes:str,
                 embedding_size:int,
                 pooling_method:str,
                 encoder_pretrained:str='',
                 device='cpu') -> None:
        super(NER, self).__init__()
        self.sentence_transformer = CustomSentenceTransformer(backbone_model, 
                                                              embedding_size, 
                                                              pooling_method, 
                                                              device, 
                                                              output_encoder_embedding=True) # output_encoder_embedding=False output the intermediate token level embeddings
        
        # Required a transformer hidden size
        self.linear               = nn.Linear(self.sentence_transformer.backbone_transformer_encoder.config.hidden_size,   
                                              n_classes)   # Input [transformer_hidden_size(768), seq_len]
        self.dropout              = nn.Dropout(0.3)
        self.st_pretrained        = encoder_pretrained
        self.tokenizer            = self.sentence_transformer.tokenizer

        # load pre-trained model of encoder
        if os.path.isfile(self.st_pretrained):
            self.sentence_transformer.load_state_dict(torch.load(self.st_pretrained))
            print("Encoder Weight Loaded: ", self.st_pretrained)

    def ner_head(self, sentence_embedded: Tensor):
        """
        Constructs a classification head for sentence NER using dropout and linear layers.
        """
        x = self.dropout(sentence_embedded)  # Using dropout for regualarization
        x = self.linear(x)                   # First linear transformation

        return x
    
    def forward(self, input_text:List[str]):
        encoder_output = self.sentence_transformer(input_text)
        classes_output = self.ner_head(encoder_output)

        return classes_output