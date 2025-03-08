from datasets import load_dataset


def load_huggingFace_dataset(dataset_name:str, dataset_type='pair'):
    train_dataset = load_dataset(dataset_name, dataset_type, split='train')
    test_dataset  = load_dataset(dataset_name, dataset_type, split='test')

    return train_dataset, test_dataset





if __name__ == '__main__':
    dataset_name = 'sentence-transformers/all-nli'
    train_dataset, test_dataset = load_huggingFace_dataset(dataset_name)
    