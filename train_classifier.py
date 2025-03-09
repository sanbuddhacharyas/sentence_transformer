import wandb
import torch
import os
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader

from models.sentence_classification import SentenceClassifier
from utils.config_parser import load_config
from utils.dataloader import load_huggingFace_dataset


if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all the hyperparametes and configuration
    config = load_config()

    # Create a wandb to log
    wandb.init(project="SentenceClassification", name=f"Version_{config['project']['version']}")

    # Load required variables from configuration
    learning_rate       = config["training"]["learning_rate"]
    epochs              = config["training"]["epochs"]
    batch_size          = config["training"]["batch_size"]

    # Log arguments to W&B
    wandb.config.backbone_model_name = config["model"]["backbone_model_name"]

    # Create sentence transformer model
    sentence_classifier         = SentenceClassifier(backbone_model=config["model"]["backbone_model_name"], 
                                                     n_classes=config["training"]["n_classes"], 
                                                     embedding_size=config["model"]["output_embedding"], 
                                                     pooling_method=config["model"]["pooling_method"], 
                                                     encoder_pretrained=config["model"]["save_model"],
                                                     device=device)
    
    optimizer                   = optim.AdamW(sentence_classifier.parameters(), lr=learning_rate)

    # Load dataset
    train_dataset, test_dataset = load_huggingFace_dataset(config["data"]["classification_dataset_path"], dataset_type='classification')
    dataloader_train            = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    dataloader_test             = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    total_test                  = len(test_dataset)
    print("Total number of test loop: ", total_test)

    # Define Loss Function
    loss_fn   = nn.CrossEntropyLoss()
    iteration = 0

    # Use GPU
    sentence_classifier = sentence_classifier.to(device)

    # Load pretrained dataset:
    if os.path.isfile(config["model"]["save_model_classifier"]):
        sentence_classifier.load_state_dict(torch.load(config["model"]["save_model"]))
        print("Model Loaded: ", config["model"]["save_model"])

    for epoch in range(epochs):
        # Evaluate test loss
        test_loss = 0.0
        sentence_classifier.eval()
        for text_batch in dataloader_test:
            pred_class = sentence_classifier(text_batch['text'])
            test_loss += loss_fn(pred_class, text_batch['label']).item()

        wandb.log({'test_loss': test_loss / total_test})
        print(f"test_loss: {test_loss / total_test}")
        
        sentence_classifier.train()

        total_loss = 0.0
        for batch_ind, train_batch in enumerate(dataloader_train):
            
            optimizer.zero_grad()                             # Reset gradients
            output = sentence_classifier(train_batch['text'])
            loss   = loss_fn(output, train_batch['label'])     # Computer loss
            loss.backward()                                    # Backpropagate
            optimizer.step()                                   # Update weights
            total_loss +=  loss.item()
            iteration  += 1

            if (batch_ind % 200) == 0:

                wandb.log({"train_loss:": total_loss/(batch_ind+1),
                           "epoch": epoch,
                           "iteration": iteration
                           })
                
                print(f"train_loss: {total_loss/(batch_ind+1)}, epoch: {epoch}, iteration: {iteration}")
            
        # Save the model after epoch 
        torch.save(sentence_classifier.state_dict(), f'weights/sentence_classifier_{epoch}.pth')
        
        



        



    