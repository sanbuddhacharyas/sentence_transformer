import torch
import torch.nn as nn
from transformer import AutoModel, AutoTokenizer

# This class takes a transformer model as backbone and implement an end to end sentence transformer.
class CustomSentenceTransformer(nn.Module):
    
    def __init__(self, 
                 backbone_model:str,
                 pooling_method:str='mean') -> None:
        super(CustomSentenceTransformer, self).__init__()
        self.backbone_transformer_encoder = AutoModel.from_pretrained(backbone_model)        # Load the pretrained transformer model from the hugging face
        self.tokenizer                    = AutoTokenizer.from_pretrained(backbone_model)    # Load the tokenizer for transformer model
        self.pooling_method               = pooling_method

    def mean_pooling_layer(self, 
                     encoder_embeddings: torch.Tensor, 
                     attention_mask:     torch.Tensor) -> torch.Tensor:
        """
            This function computes mean pooling from the encoder_embeddings.
        """
        attention_mask       = attention_mask.unsqueeze(-1).expand(self.tokenizer.size()).float()  # Expand the attention mask to match the shape of encoder embeddings
        embeddings_sum       = torch.sum(encoder_embeddings * attention_mask, 1)                   # Sum the embeddings of valid tokens
        count_valid_tokens   = attention_mask.sum(1)                                               # Count total valid tokens per sentence
        count_valid_tokens   = torch.clamp(count_valid_tokens, 1e-10)                              # Add small number to prevent from zero division
        embeddings_mean      = torch.div(embeddings_sum, count_valid_tokens)                       # Calculates the mean of the embeddings

        return embeddings_mean
    
    def forward(self, input_text):

        encoded_inputs       = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")     # Encode the input text
        encoder_embeddings   = self.backbone_transformer_encoder(**encoded_inputs).last_hidden_state              # Shape: [batch_size, sequence_length, hidden_dim]

        # Apply mean pool  
        sentence_embedded    = self.mean_pooling_layer(encoder_embeddings, encoded_inputs["attention_mask"])

        return sentence_embedded


