import sys
sys.path.insert(0, './')

import torch
from torch.utils.data import DataLoader, Dataset

def DataLoaderClassifier(config):
     # Load datasets
    train_dataset, test_dataset  = dummyDataloaderGenerator(data_size=2000, seq_len=config['model']['max_token_size'], num_classes=config['model']['n_classes_cls']), \
                                   dummyDataloaderGenerator(data_size=128, seq_len=config['model']['max_token_size'], num_classes=config['model']['n_classes_cls'])
    
    dataloader_train            = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,  num_workers=0)
    dataloader_test             = DataLoader(test_dataset,  batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    return dataloader_train, dataloader_test

# For Task A (Sentence Classification)
# Creates a dummy dataset
class dummyDataloaderGenerator(Dataset):
    def __init__(self, data_size=512, seq_len=128, num_classes=4):
        self.seq_len = seq_len
        self.data    = torch.randint(0, 10000, (data_size, seq_len))  # Random data token ids
        self.labels  = torch.randint(0, num_classes, (data_size,))    # Single label per sequence

        # Generate random padding start positions (from 10 to seq_len)
        self.pad_start_positions = torch.randint(10, seq_len, (data_size,))
        self.masks = torch.ones(data_size, seq_len, dtype=torch.long)  # Default mask (all 1s)

        # Apply padding & masking
        for i, pad_start in enumerate(self.pad_start_positions):
            self.data[i, pad_start:]  = 0  # Zero out padding region
            self.masks[i, pad_start:] = 0  # Masking (1 = valid, 0 = padding)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx],
            "attention_mask": self.masks[idx],
            "labels":self.labels[idx]}
    

if __name__ == '__main__':

    dataset = dummyDataloaderGenerator()
    print(dataset[0])

