import wandb
import torch
import os

import torch.optim as optim
from torch.utils.data import DataLoader

from models.sentence_transformer import CustomSentenceTransformer
from utils.config_parser import load_config
from utils.dataloader import load_huggingFace_dataset
from utils.loss import CustomMultipleNegativesRankingLoss



if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all the hyperparametes and configuration
    config = load_config()

    # Create a wandb to log
    wandb.init(project=config["project"]["name"], name=f"Version_{config['project']['version']}")

    backbone_model_name = config["model"]["backbone_model_name"]
    pooling_method      = config["model"]["pooling_method"]
    dataset_name        = config["data"]["dataset_path"]
    learning_rate       = config["training"]["learning_rate"]
    epochs              = config["training"]["epochs"]
    batch_size          = config["training"]["batch_size"]

    # Log arguments to W&B
    wandb.config.backbone_model_name = backbone_model_name
    wandb.config.backbone_model_name = backbone_model_name

    # Create sentence transformer model
    sentence_transformer        = CustomSentenceTransformer(backbone_model_name, pooling_method=pooling_method, device=device)
    optimizer                   = optim.AdamW(sentence_transformer.parameters(),lr=learning_rate)

    # Load dataset
    train_dataset, test_dataset = load_huggingFace_dataset(dataset_name)
    dataloader_train            = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    dataloader_test             = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    total_test                  = len(test_dataset)
    print("Total number of test loop: ", total_test)

    # Define Loss Function
    loss_MNRL = CustomMultipleNegativesRankingLoss(sentence_transformer, device)
    iteration = 0

    # Use GPU
    sentence_transformer = sentence_transformer.to(device)

    # Load pretrained dataset:
    if os.path.isfile(config["model"]["save_model"]):
        
        sentence_transformer.load_state_dict(torch.load(config["model"]["save_model"]))
        print("Model Loaded: ", config["model"]["save_model"])

    for epoch in range(epochs):
        epoch += 3
        sentence_transformer.train()
        total_loss = 0.0

        # Evaluate test loss
        test_loss = 0.0
        total_test = 0
        sentence_transformer.eval()
        for test_batch in dataloader_test:
            test_loss += loss_MNRL(test_batch).item()
            total_test += 1

        wandb.log({'test_loss': test_loss / total_test})
        print(f"test_loss: {test_loss / total_test}")
        
        for batch_ind, input_sequence in enumerate(dataloader_train):
            
            optimizer.zero_grad()                            # Reset gradients
            loss = loss_MNRL(input_sequence)                 # Computer loss
            loss.backward()                                  # Backpropagate
            optimizer.step()                                 # Update weights
            total_loss +=  loss.item()
            iteration += 1

                
            if (batch_ind % 200) == 0:

                wandb.log({"train_loss:": total_loss/(batch_ind+1),
                           "epoch": epoch,
                           "iteration": iteration
                           })
                
                print(f"train_loss: {total_loss/(batch_ind+1)}, epoch: {epoch}, iteration: {iteration}")
            
        # Save the model after epoch 
        torch.save(sentence_transformer.state_dict(), f'weights/sentence_transformer_{epoch}.pth')
        
        



        



    