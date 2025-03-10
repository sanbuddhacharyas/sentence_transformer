
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

## Configuration
Configure hyperparameters and paths in `config.yaml`.
```
    project: 
        name: sentence_transformer1
        version: 1.0
        
    training:
        batch_size: <training batch size>
        epochs: <number of epoch to train>
        learning_rate: <models initial learning rate>
        n_classes: <number of classes for classification and NER models>

    model:
        backbone_model_name: <huggingface transformer model>"google-bert/bert-base-uncased" 
        pooling_method : <method of pooling in sentence transformer>
        sentence_transformer_pretrained: <path to trained sentence transformer model>
        save_model_classifier: <path to trained classification  model>
        save_model_NER: "path to trained NER  model"
        output_embedding: <Output Sentence Embedding Size>

    data:
        dataset_path: <huggingface dataset for training sentence transformer>
        classification_dataset_path: <huggingface dataset for training classification model>
        ner_dataset_path: <huggingface dataset for training NER model>
```

# Dataset Preperation
Following dataset from huggingface is used to train the models:<br>
1. Sentence Transformer: ["sentence-transformers/all-nli"](https://huggingface.co/datasets/sentence-transformers/all-nli)<br>
``` 
    data:
        dataset_path: "sentence-transformers/all-nli"
```
2. Sentence Classification: ["fancyzhx/ag_news"](https://huggingface.co/datasets/fancyzhx/ag_news)<br>
``` 
    data:
        classification_dataset_path: ""fancyzhx/ag_news""
```

3. Name Entity Recognition (NER): ["eriktks/conll2003"](https://huggingface.co/datasets/eriktks/conll2003)
``` 
    data:
        ner_dataset_path: "eriktks/conll2003"
```

# Model Training 
Each models has different code to train the model which is in the folder train.
```
    ├── train
    │   ├── train_classifier.py
    │   ├── train_ner.py
    │   └── train_sentence_transformer.py
```
### 1. Train Sentence_Transformer model
Set following parameters: 
```
project: 
    name: sentence_transformer1
    version: 1.0

training:
    batch_size: 32
    epochs: 20
    learning_rate: 0.00001

model:
    backbone_model_name: "google-bert/bert-base-uncased"
    pooling_method : "mean"
    output_embedding: 512

data:
    dataset_path: "sentence-transformers/all-nli"
```
```
cd train
conda activate sentenceTransformer
python train train_sentence_transformer.py
```

### 2. Train Sentence Classification model
Set following parameters: 
```
project: 
    name: Sentence_classification
    version: 1.0

training:
    batch_size: 32
    epochs: 20
    learning_rate: 0.00001
    n_classes:4

model:
    backbone_model_name: "google-bert/bert-base-uncased"
    pooling_method : "mean"
    sentence_transformer_pretrained: "weights/<name of sentence transformer weight>"
    save_model_classifier: "" 
    save_model_NER: ""
    output_embedding: 512

data:
    classification_dataset_path: "fancyzhx/ag_news"
```
```
cd train
conda activate sentenceTransformer
python train train_classifier.py
```

### 3. Train Name Entity Recognition Model
Set following parameters: 
```
project: 
    name: NER
    version: 1.0

training:
    batch_size: 32
    epochs: 20
    learning_rate: 0.00001
    n_classes:9

model:
    backbone_model_name: "google-bert/bert-base-uncased"
    pooling_method : "mean"
    sentence_transformer_pretrained: "weights/<name of sentence transformer weight>"
    save_model_classifier: "" 
    save_model_NER: ""
    output_embedding: 512

data:
    classification_dataset_path: "fancyzhx/ag_news"
```
```
cd train
conda activate sentenceTransformer
python train train_classifier.py
```

# Evaluate Performace Metrics
Performace metrics can be evaluated with `test.py` python script
Pass the following arguments

```
python test.py --model_name <Model Name NER or classifier> --pretrained_path <path to model saved weights> --n_classes <Number of clases to predict> --batch_size <batch size to load dataset>
```
Example:
 
**Name Entity Recognition:**
```

python test.py --model_name 'NER' --pretrained_path 'weights/sentence_classifier_19.pth' --n_classes 9 --batch_size 32
```

**Sentence Classifier:**
```
python test.py --model_name 'NER' --pretrained_path 'weights/sentence_classifier_8.pth' --n_classes 4 --batch_size 32
```