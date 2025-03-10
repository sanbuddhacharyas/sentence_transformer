import sys
sys.path.insert(0, '../')

import wandb
import torch
import os
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader

from models.sentence_ner import NER
from utils.config_parser import load_config
from utils.dataloader import load_huggingFace_dataset, collate_fn, preprocess_data


if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all the hyperparametes and configuration
    config = load_config('../config.yaml')

    # Create a wandb to log
    wandb.init(project="NER", name=f"Version_{config['project']['version']}")

    # Load required variables from configuration
    learning_rate       = config["training"]["learning_rate"]
    epochs              = config["training"]["epochs"]
    batch_size          = config["training"]["batch_size"]
    n_classes           = config["training"]["n_classes"]

    # Log arguments to W&B
    wandb.config.backbone_model_name = config["model"]["backbone_model_name"]

    # Create Sentence Classifier Model
    ner_model         = NER(backbone_model=config["model"]["backbone_model_name"], 
                            n_classes=n_classes, 
                            embedding_size=config["model"]["output_embedding"], 
                            pooling_method=config["model"]["pooling_method"], 
                            encoder_pretrained=config["model"]["sentence_transformer_pretrained"],
                            device=device)
    # Set Optimizer to traing the model
    optimizer                   = optim.AdamW(ner_model.parameters(), lr=learning_rate)

    # Load datasets
    train_dataset, test_dataset = load_huggingFace_dataset(config["data"]["ner_dataset_path"], dataset_type='NER')

    # Preprocess data and tokenize the inputs, padding is done to make batch size equal
    train_dataset = train_dataset.map(preprocess_data, batched=True, fn_kwargs={"tokenizer": ner_model.tokenizer}).remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags","token_type_ids"])
    test_dataset  = test_dataset.map(preprocess_data,  batched=True, fn_kwargs={"tokenizer":  ner_model.tokenizer}).remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags","token_type_ids"])
    # Create a dataloader
    dataloader_train            = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    dataloader_test             = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    total_test_loop             = len(test_dataset)//batch_size

    print("Total Training Dataset: ", len(train_dataset))
    print("Total Testing  Dataset: ", len(test_dataset))

    # Define Loss Function
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100) # ignore padding with ignore index -100 

    # Use GPU
    ner_model = ner_model.to(device)

    # Load pretrained dataset:
    if os.path.isfile(config["model"]["save_model_NER"]):
        ner_model.load_state_dict(torch.load(config["model"]["save_model"]))
        print("Model Loaded: ", config["model"]["save_model"])

    iteration = 0
    for epoch in range(epochs):
        # Evaluate test loss
        test_loss = 0.0
        ner_model.eval()
        for text_batch in dataloader_test:
            labels = text_batch.pop('labels').to('cpu')
            logits = ner_model(text_batch).permute(0, 2, 1).detach().cpu()
            
            test_loss += loss_fn(logits, labels).item()
            
        wandb.log({'test_loss': test_loss / total_test_loop})
        print(f"test_loss: {test_loss / total_test_loop}")
        
        
        ner_model.train()

        total_loss = 0.0
        for batch_ind, train_batch in enumerate(dataloader_train):
            
            optimizer.zero_grad()                                          # Reset gradients
            labels = train_batch.pop('labels')
            output = ner_model(train_batch)                                # Run Sentence Classifier model with input text sequence
            loss   = loss_fn(output.permute(0, 2, 1), labels.to(device))   # Computer loss
            loss.backward()                                                # Backpropagate
            optimizer.step()                                               # Update weights
            total_loss +=  loss.item()
            iteration  += 1

            # Log Socres every 200 iteration
            if (batch_ind % 200) == 0:
                wandb.log({"train_loss:": total_loss/(batch_ind+1),
                           "epoch": epoch,
                           "iteration": iteration
                           })
                
                print(f"train_loss: {total_loss/(batch_ind+1)}, epoch: {epoch}, iteration: {iteration}")
            
        # Save the model after epoch 
        torch.save(ner_model.state_dict(), f'../weights/NER_{epoch}.pth')