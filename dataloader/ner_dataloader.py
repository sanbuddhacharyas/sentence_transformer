
import sys
sys.path.insert(0, './')

from torch.utils.data import DataLoader, Dataset

import torch
def DataLoaderNER(config):
     # Load datasets
    train_dataset, test_dataset  =  dummyNERGenerator(data_size=64, seq_len=config['model']['max_token_size'], num_classes=config['model']['n_classes_ner']), \
                                    dummyNERGenerator(data_size=8,  seq_len=config['model']['max_token_size'],  num_classes=config['model']['n_classes_ner'])
    
    dataloader_train             = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,  num_workers=0)
    dataloader_test              = DataLoader(test_dataset,  batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    return dataloader_train, dataloader_test

# Dummy NER Dataset Loader with special tokens and padding
class dummyNERGenerator(Dataset):
    def __init__(self, data_size=512, seq_len=128, num_classes=4):
        self.seq_len = seq_len
        self.data   = torch.randint(0, 10000, (data_size, seq_len))
        self.labels = torch.randint(0, num_classes, (data_size, seq_len))

        # Adding special tokens for [CLS] and [SEP]
        self.labels[:, 0] = 0  # [CLS] token label
        self.labels[:, -1] = 0  # [SEP] token label

        # Generate random padding region start (from position 10 to seq_len)
        self.pad_start_positions = torch.randint(10, seq_len, (data_size,))
        self.masks = torch.ones(data_size, seq_len, dtype=torch.long)  # Default to 1 (valid)

        # Apply padding & masking
        for i, pad_start in enumerate(self.pad_start_positions):
            self.data[i, pad_start:] = 0  # Zero out padded data
            self.labels[i, pad_start:] = -100  # Label -100 to ignore padding in loss functions
            self.masks[i, pad_start:] = 0  # Set mask to 0 for padding region

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx],
            "attention_mask": self.masks[idx],
            "labels":self.labels[idx]}
    

"""Future Use
def preprocess_data(examples, tokenizer):
    # Tokenize the sentences
    tokenized_inputs = tokenizer(examples['tokens'], 
                                 truncation=True, 
                                 padding='max_length', 
                                 is_split_into_words=True, 
                                 return_tensors="pt")
    
    # Align the labels
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        # Expand the labels for tokens split by the tokenizer
        word_ids  = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(torch.tensor(label_ids))
    
    # Add the labels and return the tokenized inputs
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def collate_fn(batch):
    # Initialize a dictionary to hold batched data
    batched_data = {}
    
    # Iterate over all keys in the dictionary (assumes all examples have the same keys)
    for key in batch[0].keys():
        # Stack the tensors for each key while preserving their original shape
        batched_data[key] = torch.stack([torch.tensor(item[key]) for item in batch], dim=0)
    
    return batched_data


def DataloaderNER(dataset_path, batch_size, tokenizer):
     # Load datasets
    train_dataset, test_dataset = load_huggingFace_dataset(dataset_path, dataset_type='NER')

    # Preprocess data and tokenize the inputs, padding is done to make batch size equal
    train_dataset = train_dataset.map(preprocess_data, batched=True, fn_kwargs={"tokenizer":  tokenizer}).remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags","token_type_ids"])
    test_dataset  = test_dataset.map(preprocess_data,  batched=True, fn_kwargs={"tokenizer":  tokenizer}).remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags","token_type_ids"])
    
    # Create a dataloader
    dataloader_train            = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    dataloader_test             = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    return dataloader_train, dataloader_test
"""
    

if __name__ == '__main__':

    dataset = dummyNERDataloader()
    print(dataset[0][0]["input_ids"], dataset[0][1])