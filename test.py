import torch
import numpy as np

import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import calculate_all_metrics
from utils.config_parser import load_config
from models.sentence_transformer import CustomSentenceTransformer
from models.sentence_classification import SentenceClassifier
from models.sentence_ner import NER
from utils.dataloader import load_huggingFace_dataset, preprocess_data, collate_fn


def test_classification_model(config:dict, 
                              n_classes:int,
                              pretrained_model:str,
                              batch_size:int=8,
                              device:str='cpu') -> None:
    """This function calculates performance metrics of classification model on test dataset
    """
    sentence_classifier         = SentenceClassifier(backbone_model=config["model"]["backbone_model_name"], 
                                                     n_classes=n_classes, 
                                                     embedding_size=config["model"]["output_embedding"], 
                                                     pooling_method=config["model"]["pooling_method"], 
                                                     encoder_pretrained='',
                                                     device=device)
    
    sentence_classifier.load_state_dict(torch.load(pretrained_model, map_location=torch.device(device)))

    _, test_dataset = load_huggingFace_dataset(config["data"]["classification_dataset_path"], dataset_type='classification')
    dataloader_test = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

    y_pred, y_truth = [], []

    sentence_classifier = sentence_classifier.to(device)
    sentence_classifier.eval()
    for test_batch in tqdm(dataloader_test):
        y_pred.append(np.argmax(sentence_classifier(test_batch['text']).detach().cpu(), axis=-1).numpy())
        y_truth.append(test_batch['label'].numpy())

    y_pred  = np.concatenate(y_pred)
    y_truth = np.concatenate(y_truth)

    calculate_all_metrics(y_truth, y_pred)
   

def test_NER(config:dict, 
             n_classes:int,
             pretrained_model:str,
             batch_size:int=8,
             device:str='cpu'):
    """This function calculates performance metrics of Name Entity Recognition model on test dataset
    """

    ner_model         = NER(backbone_model=config["model"]["backbone_model_name"], 
                            n_classes=n_classes, 
                            embedding_size=config["model"]["output_embedding"], 
                            pooling_method=config["model"]["pooling_method"], 
                            encoder_pretrained='',
                            device=device)
    
    ner_model.load_state_dict(torch.load(pretrained_model, map_location=torch.device(device)))

    # Load the model
    _, test_dataset = load_huggingFace_dataset(config["data"]["ner_dataset_path"], dataset_type='NER')

    # Preprocess data and tokenize the inputs, padding is done to make batch size equal
    test_dataset    = test_dataset.map(preprocess_data,  batched=True, fn_kwargs={"tokenizer":  ner_model.tokenizer}).remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags","token_type_ids"])
    
    # Create a dataloader
    dataloader_test = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    y_pred, y_truth = [], []

    ner_model = ner_model.to(device)     # Move a model to device (i.e cuda)
    ner_model.eval()                     # Keep the model in evaluation mode

    for text_batch in tqdm(dataloader_test):
        labels           = text_batch.pop('labels').to('cpu')
        predicted_labels = np.argmax(ner_model(text_batch).detach().cpu().numpy(), axis=-1).squeeze() # Shape:[Batch_size, Seq_len]
       
        # Remove the padded tokens with label=-100
        for ind, label in enumerate(labels):
            mask = torch.tensor(label !=-100, dtype=torch.bool)  # Mask only the valid tokens
            y_pred.append(predicted_labels[ind][mask])          # Add predicted labels to list
            y_truth.append(labels[ind][mask])                   # Add ground truth labels to list

    y_pred  = np.concatenate(y_pred)
    y_truth = np.concatenate(y_truth)

    calculate_all_metrics(y_truth, y_pred)

if __name__ == '__main__':
    
    # Initialize the parser
    parser = argparse.ArgumentParser(description="")

    # Positional Argument
    parser.add_argument("--model_name", type=str, choices=["classifier", "NER"], help="classifier or NER")
    parser.add_argument("--pretrained_path", type=str, help="Path to saved model weight")
    parser.add_argument("--n_classes", type=int, help="number of classes to predict")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size of the model")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config()          # Load the hyperparameters and configurations

    if (args.model_name == 'NER'):
        test_NER(config, n_classes=args.n_classes, pretrained_model=args.pretrained_path, batch_size=args.batch_size, device=device)

    elif (args.model_name == 'classifier'):
        test_classification_model(config, n_classes=args.n_classes, pretrained_model=args.pretrained_path, batch_size=args.batch_size, device=device)

    else:
        print("Enter Correct Model!!!")

