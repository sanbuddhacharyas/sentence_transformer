from datasets import load_dataset
from torch.utils.data import DataLoader

def load_huggingFace_dataset(dataset_name:str, dataset_type='pair'):
    train_dataset, test_dataset = [None], [None]
    if dataset_type=='sentence_transformer':
        train_dataset = load_dataset(dataset_name, 'pair', split='train')
        test_dataset  = load_dataset(dataset_name, 'pair', split='test')

    elif dataset_type=='classification':
        train_dataset = load_dataset(dataset_name, split='train')
        test_dataset  = load_dataset(dataset_name, split='test')

    return train_dataset, test_dataset




if __name__ == '__main__':
    dataset_name = 'fancyzhx/ag_news'
    train_dataset, test_dataset = load_huggingFace_dataset(dataset_name, dataset_type='classification')
    dataloader_train            = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=0)
    
    for batch_ind, (train_batch) in enumerate(dataloader_train):
        print(type(train_batch['label']))
    