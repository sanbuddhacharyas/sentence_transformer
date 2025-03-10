import os
import torch
from torch import nn, Tensor
from typing import List

from models.sentence_transformer import CustomSentenceTransformer

class SentenceClassifier(nn.Module):
    def __init__(self,
                 backbone_model:str,
                 n_classes:str,
                 embedding_size:int,
                 pooling_method:str,
                 encoder_pretrained:str='',
                 device='cpu') -> None:
        super(SentenceClassifier, self).__init__()
        self.encoder            = CustomSentenceTransformer(backbone_model, embedding_size, pooling_method, device)
        self.linear1            = nn.Linear(embedding_size,    embedding_size//2)    # 512 / 2 = 256
        self.linear2            = nn.Linear(embedding_size//2, embedding_size//4)    # 256 / 2 = 126
        self.output             = nn.Linear(embedding_size//4, n_classes)            # number of classes
        self.activation         = nn.GELU()
        self.softmax            = nn.Softmax()
        self.encoder_pretrained = encoder_pretrained

        # load pre-trained model of encoder
        print(self.encoder_pretrained)
        if os.path.isfile(self.encoder_pretrained):
            self.encoder.load_state_dict(torch.load(self.encoder_pretrained))
            print("Encoder Weight Loaded: ", self.encoder_pretrained)

    def classifier(self, sentence_embedded: Tensor):
        """
        Constructs a classification head for sentence embeddings using linear layers and activation functions.
        
        Args:
            sentence_embedded (Tensor): Input tensor containing sentence embeddings.
        
        Returns:
            Tensor: Probability distribution over output classes after applying softmax.
        """
        x = self.linear1(sentence_embedded)  # First linear transformation
        x = self.activation(x)               # Apply GELU activation
        x = self.linear2(x)                  # Second linear transformation
        x = self.activation(x)               # Apply GELU activation
        x = self.output(x)                   # Project to output class logits
        output = self.softmax(x)             # Convert logits to probabilities

        return output
    
    def forward(self, input_text:List[str]):
        encoder_output = self.encoder(input_text)
        classes_output = self.classifier(encoder_output)

        return classes_output