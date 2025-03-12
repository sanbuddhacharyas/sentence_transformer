import torch
import torch.nn as nn
from torch import optim

from utils.metrics import metric_evaluation

# Training loop
def train_model(model, dataloader_cls, dataloader_ner, lr=0.00001, epochs=5, alpha=0.5, batch_size=8):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ner = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=lr)

    device      = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        iteration  = 0

        # Training and Testing dataloader
        train_cls, test_cls = dataloader_cls
        train_ner, test_ner = dataloader_ner

        # Alternate batches from Dataset A and Dataset B
        for data_cls, data_ner in zip(train_cls, train_ner):

            # Pop the labels from input tokens id
            labels_cls =  data_cls.pop('labels')
            labels_ner =  data_ner.pop('labels')

            optimizer.zero_grad()

            # Forward pass
            outputs_cls, _ = model(data_cls)
            _, outputs_ner = model(data_ner)

            # Calculate losses
            loss_cls       = criterion_cls(outputs_cls, labels_cls.to(device))

            # NER has shape of [batch_size, seq_len, seq_len] but  CrossEntropyLoss takes Shape: [batch_size, n_classes, seq_len]
            loss_ner       = criterion_ner(outputs_ner.permute(0, 2, 1), labels_ner.to(device)) 

            # Weighted loss combination
            loss = alpha * loss_cls + (1 - alpha) * loss_ner
            loss.backward()
            optimizer.step()

            # Metrics calculation
            total_loss += loss.item()
            iteration  += 1

            print(f"Epoch {epoch+1}/{epochs}, Iteration {iteration+1}/{batch_size} Loss: {total_loss / iteration:.4f}")
        
        avg_loss = total_loss / (len(dataloader_cls) + len(dataloader_ner))
        acc_cls, precision_score_cls, recall_score_cls, f1_score_cls, acc_ner, precision_score_ner, recall_score_ner, f1_score_ner = metric_evaluation(model, test_cls, test_ner)
        print(f"Epoch {epoch+1}/{epochs}| Loss: {avg_loss:.4f} | Accuracy_Cls: {acc_cls:.4f} | Precision_Cls: {precision_score_cls:.4f} | Recall_cls:{recall_score_cls:.4f} | f1_score_cls:{f1_score_cls:.4f}")
        print(f"Epoch {epoch+1}/{epochs}| Loss: {avg_loss:.4f} | Accuracy_ner: {acc_ner:.4f} | Precision_ner: {precision_score_ner:.4f} | Recall_ner:{recall_score_ner:.4f} | f1_score_ner:{f1_score_ner:.4f}")

        