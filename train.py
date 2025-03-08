# import wandb

import torch.optim as optim
from torch.utils.data import DataLoader

from models.sentence_transformer import CustomSentenceTransformer
from utils.config_parser import load_config
from utils.dataloader import load_huggingFace_dataset
from utils.loss import CustomMultipleNegativesRankingLoss



if __name__=='__main__':

    # Load all the hyperparametes and configuration
    config = load_config()

    # Create a wandb to log
    # wandb.init(project=config["project"]["name"], name=f"Version_{config['project']['version']}")


    backbone_model_name = config["model"]["backbone_model_name"]
    pooling_method      = config["model"]["pooling_method"]
    dataset_name        = config["data"]["dataset_path"]
    learning_rate       = config["training"]["learning_rate"]
    epochs              = config["training"]["epochs"]
    batch_size          = config["training"]["batch_size"]

    # Log arguments to W&B
    # wandb.config.backbone_model_name = backbone_model_name
    # wandb.config.backbone_model_name = backbone_model_name

    print(learning_rate)
    # Create sentence transformer model
    sentence_transformer        = CustomSentenceTransformer(backbone_model_name, pooling_method=pooling_method)
    optimizer                   = optim.AdamW(sentence_transformer.parameters(),lr=learning_rate)

    # Load dataset
    train_dataset, test_dataset = load_huggingFace_dataset(dataset_name)
    dataloader_train            = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    dataloader_test             = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

    total_test                  = len(test_dataset)

    # Define Loss Function
    loss_MNRL = CustomMultipleNegativesRankingLoss(sentence_transformer)
    iteration = 0

    for epoch in range(epochs):
        sentence_transformer.train()
        total_loss = 0.0
        
        for batch_ind, input_sequence in enumerate(dataloader_train):
            optimizer.zero_grad()                 # Reset gradients
            loss = loss_MNRL(input_sequence)      # Computer loss
            loss.backward()                       # Backpropagate
            optimizer.step()                      # Update weights
            total_loss +=  loss.item()
            iteration += 1

            if (batch_ind % 2000) == 0:
                sentence_transformer.eval()

                # wandb.log({"train_loss:": total_loss/(batch_ind+1),
                #            "epoch": epoch,
                #            "iteration": iteration
                #            })
                
                print(f"train_loss: {total_loss/(batch_ind+1)}, epoch: {epoch}, iteration: {iteration}")
            
        # Evaluate test loss
        test_loss = 0.0
    
        for test_batch in dataloader_test:
            loss = loss_MNRL(test_batch).item()
            test_loss += loss

        # wandb.log({'test_loss': loss.item() / total_test})
        print(f"test_loss: {loss.item() / total_test}")


        



    