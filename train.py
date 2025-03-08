import torch.optim as optim
from torch.utils.data import DataLoader

from models.sentence_transformer import CustomSentenceTransformer
from utils.config_parser import load_config
from utils.dataloader import load_huggingFace_dataset
from utils.loss import MultipleNegativesRankingLoss



if __name__=='__main__':

    # Load all the hyperparametes and configuration
    config = load_config()

    backbone_model_name = config["model"]["backbone_model_name"]
    pooling_method      = config["model"]["pooling_method"]
   
    dataset_name        = config["data"]["dataset_path"]

    learning_rate       = config["training"]["learning_rate"]
    epochs              = config["training"]["epochs"]
    batch_size          = config["training"]["batch_size"]

    print(learning_rate)
    # Create sentence transformer model
    sentence_transformer = CustomSentenceTransformer(backbone_model_name, pooling_method=pooling_method)
    optimizer            = optim.AdamW(sentence_transformer.parameters(),lr=learning_rate)

    # Load dataset
    train_dataset, test_dataset = load_huggingFace_dataset(dataset_name)
    dataloader                  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Define Loss Function
    loss_MNRL = MultipleNegativesRankingLoss(sentence_transformer)


    for epoch in range(epochs):
        sentence_transformer.train()
        total_loss = 0.0
        
        for batch_ind, input_sequence in enumerate(dataloader):
            optimizer.zero_grad()                 # Reset gradients
            loss = loss_MNRL(input_sequence)      # Computer loss
            loss.backward()                       # Backpropagate
            optimizer.step()                      # Update weights
            total_loss +=  loss.item()

            if batch_ind % 1000 == 0:
                print("Total Loss: ", total_loss/(batch_ind + 1))
        



    