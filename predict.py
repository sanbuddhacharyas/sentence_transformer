import torch
import numpy as np

import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm


from utils.metrics import calculate_all_metrics, cosine_similarity
from utils.config_parser import load_config
from models.sentence_transformer import CustomSentenceTransformer
from models.sentence_classification import SentenceClassifier
from models.sentence_ner import NER
from utils.dataloader import load_huggingFace_dataset


def predict_classification_model(config:dict, 
                              n_classes:int,
                              pretrained_model:str,
                              input_sentence:str,
                              device:str='cpu') -> str:
    """This function calculates performance metrics of classification model on test dataset
    """
    sentence_classifier         = SentenceClassifier(backbone_model=config["model"]["backbone_model_name"], 
                                                     n_classes=n_classes, 
                                                     embedding_size=config["model"]["output_embedding"], 
                                                     pooling_method=config["model"]["pooling_method"], 
                                                     encoder_pretrained='',
                                                     device=device)
    
    sentence_classifier.load_state_dict(torch.load(pretrained_model, map_location=torch.device(device)))

    output = np.argmax(sentence_classifier([input_sentence]).detach().cpu(), axis=-1).numpy()[0]

    return config['data']['ag_news']['ind2class'][output]
   

def predict_NER(config:dict, 
            n_classes:int,
            pretrained_model:str,
            input_sentence:str,
            device:str='cpu') -> None:
    """This function predicts the Name Entity of the input sentence
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

    tokenized_inputs = ner_model.tokenizer([input_sentence], 
                                 truncation=True, 
                                 padding='max_length', 
                                 is_split_into_words=False, 
                                 return_tensors="pt")
    
    predicted_output = np.argmax(ner_model(tokenized_inputs).detach().cpu().numpy(), axis=-1).squeeze() # Shape:[Batch_size, Seq_len]
    print(predicted_output)

    return predicted_output

def find_sentence_embedding(config:dict, 
                              pretrained_model:str,
                              input_sentence:str,
                              device:str='cpu') -> str:
    """This function calculates performance metrics of classification model on test dataset
    """
    sentence_transformer = CustomSentenceTransformer(config['model']['backbone_model_name'], 
                                                     config['model']['output_embedding'], 
                                                     config['model']['pooling_method'], 
                                                     device)
    
    sentence_transformer.load_state_dict(torch.load(pretrained_model, map_location=torch.device(device)))

    return sentence_transformer(input_sentence).detach().cpu().numpy()


if __name__ == '__main__':
    
    # Initialize the parser
    parser = argparse.ArgumentParser(description="")

    # Positional Argument
    parser.add_argument("--model_name", type=str, choices=["classifier", "NER", "embedding"], help="classifier or NER")
    parser.add_argument("--pretrained_path", type=str, help="Path to saved model weight")
    parser.add_argument("--n_classes", type=int, help="number of classes to predict")
    parser.add_argument("--input", type=str, help="Enter the sentence to test")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config()          # Load the hyperparameters and configurations

   
    if (args.model_name == 'NER'):
        output = predict_NER(config, n_classes=args.n_classes, pretrained_model=args.pretrained_path, input_sentence=args.input, device=device)
        print(output)
    elif (args.model_name == 'classifier'):
        output = predict_classification_model(config, n_classes=args.n_classes, pretrained_model=args.pretrained_path, input_sentence=args.input, device=device)
        print(output)

    elif (args.model_name == 'embedding'):
        input_seq = eval(args.input)
        print(input_seq)
        output = find_sentence_embedding(config=config, pretrained_model=args.pretrained_path, input_sentence=input_seq, device=device)
        print(output.shape)
        print("Similariy: ", cosine_similarity(output[0], output[1]))

    else:
        print("Enter Correct Model!!!")

