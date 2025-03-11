from torch import nn, Tensor


class NerHead(nn.Module):
    def __init__(self, config:dict) -> None:
        
        super(NerHead, self).__init__()
      
        # Required a transformer hidden size
        self.linear               = nn.Linear(config['model']['encoder_embedding_size'], 
                                              config['model']['n_classes_ner']) # Input [transformer_hidden_size(768), seq_len]
        self.dropout              = nn.Dropout(0.3)                              # Use Dropout for Regularization
       
    def forward(self, sentence_embedded: Tensor):
        """
        Constructs a classification head for sentence NER using dropout and linear layers.
        """
        x = self.dropout(sentence_embedded)  # Using dropout for regualarization
        x = self.linear(x)                   # First linear transformation

        return x