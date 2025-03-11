import torch
import torch.nn as nn
from torch import optim


# Training loop
def train_model(model, dataloader_cls, dataloader_ner, lr=0.00001, epochs=5, alpha=0.5):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ner = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=lr)

    device      = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_cls, correct_ner = 0, 0
        total_samples_cls, total_samples_ner = 0, 0

        # Alternate batches from Dataset A and Dataset B
        for data_cls, data_ner in zip(dataloader_cls, dataloader_ner):

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
            correct_cls += (outputs_cls.argmax(1) == labels_cls).sum().item()
            correct_ner += (outputs_ner.argmax(2) == labels_ner).sum().item()
            
            total_samples_cls += labels_cls.size(0)
            total_samples_ner += labels_ner.size(0)

        avg_loss = total_loss / (len(dataloader_cls) + len(dataloader_ner))
        acc_cls = correct_cls / total_samples_cls
        acc_ner = correct_ner / total_samples_ner

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy A: {acc_cls:.4f}, Accuracy B: {acc_ner:.4f}")
