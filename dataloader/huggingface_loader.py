from datasets import load_dataset
def load_huggingFace_dataset(dataset_name:str, dataset_type='pair'):
    train_dataset, test_dataset = [None], [None]
    if dataset_type=='sentence_transformer':
        train_dataset = load_dataset(dataset_name, 'pair', split='train')
        test_dataset  = load_dataset(dataset_name, 'pair', split='test')

    elif (dataset_type=='classification') or (dataset_type=='NER'):
        train_dataset = load_dataset(dataset_name, split='train')
        test_dataset  = load_dataset(dataset_name, split='test')

    return train_dataset, test_dataset