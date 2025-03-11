
import torch

from models.MTL import MTL

from utils.config_parser import load_config
from dataloader.classification_dataloader import DataLoaderClassifier
from dataloader.ner_dataloader import DataLoaderNER
from training.MTL_trainer import train_model


if __name__ == '__main__':
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MTL_model = MTL(config, device)

    tokenizer = MTL_model.tokenizer

    # Load datasets
    dataloader_train_cls, dataloader_test_cls  =  DataLoaderClassifier(config)
    dataloader_train_ner, dataloader_test_ner  =  DataLoaderNER(config)

    train_model(MTL_model, 
                dataloader_train_cls, 
                dataloader_train_ner, 
                config['training']['learning_rate'],
                config['training']['epochs'],
                config['training']['alpha'])