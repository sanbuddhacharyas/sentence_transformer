
# Sentence Transformer & Multi-Task Learning

This repository contains code to train, test and predict the sentence transformer, sentence classification and Name Entity Recognition models.

## Prerequisites
- Python 3.10 or later
- Anaconda or Miniconda (optional, can use a python virtualenv instead)

## Installation
### 1. Create a conda environment
```
conda create -n sentenceTransformer python=3.10
conda activate sentenceTransformer 
```
### 2. Install Dependencis

```
pip install -r requirements.txt
```
## Docker Installation
Download docker from docker hub. It contains the code and pre-trained weights of sentence transformer model.

**There is no need to download the weights or configure parameters.**
```
docker pull sangamman/fetch-sentence-transformer:latest
```
OR build from docker file
```
docker build -t fetch-sentence-transformer .
```

## Run Docker Environment
```docker run -it sangamman/fetch-sentence-transformer:latest /bin/sh```

To generate the sentence embedding inside tthe docker environment use the following command:
```python predict.py --model_name 'embedding' --pretrained_path 'weights/sentence_transformer_5.pth' --input "['Fetch is 100% safe to use. All you are doing with Fetch is getting rewarded for the shopping you are already doing, no credit card is needed, just sign up with a phone number and you’re good to start earning free gift cards.', 'Fetch is a mobile application that allows you to snap your grocery, shopping & restaurant receipts and lets you earn gift cards and rewards for your purchases.']"```

Generate PCA Sentence Visualization (Saved as embedding_):
```python predict.py --model_name 'embedding' --pretrained_path 'weights/sentence_transformer_5.pth' --input "['A person on a horse jumps over a broken down airplane.', 'A person is outdoors, on a horse.', 'There are women showing affection.', 'The family is outside.', 'A skier slides along a metal rail.', 'A skier is near the rail.']"```

To train the MTL model use the following command inside the docker environment:
```python train.py```

# Predict Sentence Embedding:
Create a folder name `weight` and download the pre-trained sentence transformer model.
```https://drive.google.com/file/d/1MYjFvMCfXl-GMzpMdNEhWVGTHI4UWbPh/view?usp=sharing```

Use python script`predict.py` to outputs sentence embedding of two input sentence and calculates cosine similarity.

```python predict.py --model_name 'embedding' --pretrained_path '<path to sentence embedding trained weight>' --input "['sentence-1', 'sentence-2']" ```

For Example:

```python predict.py --model_name 'embedding' --pretrained_path 'weights/sentence_transformer_5.pth' --input "['Fetch is 100% safe to use. All you are doing with Fetch is getting rewarded for the shopping you are already doing, no credit card is needed, just sign up with a phone number and you’re good to start earning free gift cards.', 'Fetch is a mobile application that allows you to snap your grocery, shopping & restaurant receipts and lets you earn gift cards and rewards for your purchases.']"```


## Configuration
Configure hyperparameters and paths in `config.yaml`.
```
    project: 
    name: sentence_transformer
    version: 1.0
    
    training:
        batch_size: 32             <Batch Size of Dataset>
        epochs: 20.                <Number of Epoch to train the model>
        learning_rate: 0.00001     <Learning Rate of the model>
        alpha: 0.5                 <Weight to combine classifier loss and NER loss>

    model:
        backbone_model_name: "google-bert/bert-base-uncased"  <Hugging Face Transformer Backbone>
        pooling_method : "mean"     <Pooling Method in Sentence Transformer>
        sentence_transformer_pretrained: "../weights/sentence_transformer_5.pth"
        encoder_embedding_size: 768    <Backbone(BERT) Output Embedding Size>
        sentence_embedding_size: 512   <Sentence Embedding Output Size>
        n_classes_cls: 4               <Number of classes in Classifier model>
        max_token_size: 128            <Maximum token size (Require for padding the input sequence)>
        n_classes_ner: 9               <Number of classes in NER>
        dropout_ner: 0.3               <Dropout layer in NER for Regularization>

    data:
        dataset_path: "sentence-transformers/all-nli" <Training the Sentence Transformer model>
```

# Dummy Dataset
`dataloader/classification_dataloader.py and dataloader/ner_dataloader.py` contains dummy dataset loader that generates random tokenized data and respecitve labels


# Model Training 
There is two training code one to train the `sentence tranformer` and another to train the `Multi task learning` model.


### 1. Train Sentence_Transformer model
Set following parameters: 
```
cd training
conda activate sentenceTransformer
python train_sentence_transformer.py
```

### 2. Train Multi Task Learning Model
Set following parameters: 

```
conda activate sentenceTransformer
python train.py
```
