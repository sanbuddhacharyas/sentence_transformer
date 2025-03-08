import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from collections.abc import Iterable

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = (1 - label) * distance.pow(2) + (label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()


class CustomMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, 
                 model, 
                 device,
                 scale: float = 20.0) -> None:
        
        super().__init__()
        self.model  = model.to(device)
        self.scale  = scale
        self.device = device
        self.find_similarity     = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cross_entropy_loss  = nn.CrossEntropyLoss()

    def normalize_embeddings(self, embeddings: Tensor) -> Tensor:
        """
            Normalizes the embeddings matrix, so that each sentence embedding has unit length.
        """
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def cos_sim(self, a:Tensor, b: Tensor) -> Tensor:
        """
        Computes the cosine similarity between two tensors.
            output Matrix mul res[i][j]  = cos_sim(anchor[i], canditate[j])
        """

        a_norm = self.normalize_embeddings(a)
        b_norm = self.normalize_embeddings(b)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def forward(self, 
                sentence_features: Iterable[dict[str, torch.Tensor]]) -> torch.Tensor:
        
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_features[sentence_feature]) for sentence_feature in sentence_features]
        anchors    = embeddings[0]              # (batch_size, embedding_dim)
        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        # For every anchor, we compute the similarity to all other candidates (positives and negatives),
        # also from other anchors. This gives us a lot of in-batch negatives.
        scores = self.cos_sim(anchors, candidates) * self.scale                # (batch_size, batch_size * (1 + num_negatives))

        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i
        range_labels = torch.arange(0, scores.size(0), device=scores.device)

        return self.cross_entropy_loss(scores, range_labels)

