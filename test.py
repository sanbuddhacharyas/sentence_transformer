import torch
import numpy as np

from torch.utils.data import DataLoader

from utils.metrics import accuracy, precision, recall
from utils.config_parser import load_config
from models.sentence_transformer import CustomSentenceTransformer
from models.sentence_classification import SentenceClassifier
from models.sentence_ner import NER
from utils.dataloader import load_huggingFace_dataset


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config()

    sentence_classifier         = SentenceClassifier(backbone_model=config["model"]["backbone_model_name"], 
                                                     n_classes=4, 
                                                     embedding_size=config["model"]["output_embedding"], 
                                                     pooling_method=config["model"]["pooling_method"], 
                                                     encoder_pretrained='',
                                                     device=device)
    
    sentence_classifier.load_state_dict(torch.load(config['model']['save_model_classifier']))

    train_dataset, test_dataset = load_huggingFace_dataset(config["data"]["classification_dataset_path"], dataset_type='classification')
    dataloader_test             = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=4)

    y_pred       = []
    y_truth      = []

    sentence_classifier = sentence_classifier.to(device)
    for batch_ind, test_batch in enumerate(dataloader_test):
        y_pred.append(np.argmax(sentence_classifier(test_batch['text']).detach().cpu(), axis=-1).numpy())
        y_truth.append(test_batch['label'].numpy())

    y_pred  = np.concatenate(y_pred)
    y_truth = np.concatenate(y_truth)
    print(y_pred.shape, y_truth.shape)


    print(f"Accuracy: {accuracy(y_truth, y_pred)}")
    print(f"Precision: {precision(y_truth, y_pred, 'macro')}")
    print(f"Recall: {recall(y_truth, y_pred, 'macro')}")
