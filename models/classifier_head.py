from torch import nn

class ClassifierHead(nn.Module):
    def __init__(self,
                 config:str) -> None:
        super(ClassifierHead, self).__init__()

        sentence_embedding_size = config["model"]["sentence_embedding_size"]
        n_classes_cls           = config["model"]["n_classes_cls"]

        self.linear1            = nn.Linear(sentence_embedding_size,    sentence_embedding_size//2)    # 512 / 2 = 256
        self.linear2            = nn.Linear(sentence_embedding_size//2, sentence_embedding_size//4)    # 256 / 2 = 126
        self.output             = nn.Linear(sentence_embedding_size//4, n_classes_cls)                 # number of classes
        self.activation         = nn.GELU()
        self.softmax            = nn.Softmax()

    def forward(self, sentence_embedded):
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