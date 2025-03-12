import torch
import argparse
import numpy as np

from utils.metrics import cosine_similarity
from utils.config_parser import load_config
from utils.plot import pca_visualization
from models.sentence_transformer import CustomSentenceTransformer


def find_sentence_embedding(config:dict, 
                              pretrained_model:str,
                              input_sentence:str,
                              device:str='cpu') -> str:
    """This function calculates performance metrics of classification model on test dataset
    """
    sentence_transformer = CustomSentenceTransformer(config, device)
    
    sentence_transformer.load_state_dict(torch.load(pretrained_model, map_location=torch.device(device)))
    return sentence_transformer(sentence_transformer.tokenizer(input_sentence, 
                                                               padding='max_length', 
                                                               max_length=config['model']['max_token_size'], 
                                                               return_tensors="pt"))[0].detach().cpu().numpy()


if __name__ == '__main__':
    
    # Initialize the parser
    parser = argparse.ArgumentParser(description="")

    # Positional Argument
    parser.add_argument("--model_name", type=str, choices=["embedding"], help="")
    parser.add_argument("--pretrained_path", type=str, help="Path to saved model weight")
    parser.add_argument("--n_classes", default=4, type=int, help="number of classes to predict")
    parser.add_argument("--input", type=str, help="Enter the sentence to test")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config()          # Load the hyperparameters and configurations

   
    if (args.model_name == 'embedding'):
        sequences  = eval(args.input)  # Convert string to list of strings
        outputs    = []
        for input_seq in sequences:
            outputs.append(find_sentence_embedding(config=config, 
                                                   pretrained_model=args.pretrained_path, 
                                                   input_sentence=input_seq[0], 
                                                   device=device).squeeze())
        
        print("Embeddings  1st sentnece: ", outputs[0])
        print("Embeddings  2nd sentnece: ", outputs[1])
        print("Similariy: ", cosine_similarity(outputs[0], outputs[1])) # Find the cosine similarity between the two embeddings

        pca_visualization(np.array(outputs), sequences)
    else:
        print("Enter Correct Model!!!")

