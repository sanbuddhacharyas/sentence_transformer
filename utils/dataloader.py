from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

import torch

def load_huggingFace_dataset(dataset_name:str, dataset_type='pair'):
    train_dataset, test_dataset = [None], [None]
    if dataset_type=='sentence_transformer':
        train_dataset = load_dataset(dataset_name, 'pair', split='train')
        test_dataset  = load_dataset(dataset_name, 'pair', split='test')

    elif (dataset_type=='classification') or (dataset_type=='NER'):
        train_dataset = load_dataset(dataset_name, split='train')
        test_dataset  = load_dataset(dataset_name, split='test')

    return train_dataset, test_dataset

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



if __name__ == '__main__':
    dataset_name = 'eriktks/conll2003'
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    
    train_dataset, test_dataset = load_huggingFace_dataset(dataset_name, dataset_type='NER')
    
    train_dataset = train_dataset.map(preprocess_data, batched=True, fn_kwargs={"tokenizer": tokenizer}).remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags","token_type_ids"])
    
    print(train_dataset[0]['labels'])
    # tokenized_inputs = tokenizer(test_dataset[0]["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    # output_token                = tokenize_and_align_labels(train_dataset[0], tokenizer)
   
    # tokenized_inputs = tokenizer(train_dataset[0]['tokens'], 
    #                              truncation=True, 
    #                              padding='max_length', 
    #                              is_split_into_words=True, 
    #                              return_tensors="pt")
    
    # dataloader_train            = DataLoader(train_dataset, batch_size=2, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    
    # for batch_ind, (train_batch) in enumerate(dataloader_train):
    #     print(train_batch)
    #     break
    