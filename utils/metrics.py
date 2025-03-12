import numpy as np

def accuracy(y_true, y_pred):
    """Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)"""
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred, average='binary'):
    """Calculate precision: TP / (TP + FP)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    elif average in ['macro', 'micro', 'weighted']:
        unique_classes = np.unique(y_true)
        precision_scores = []

        for cls in unique_classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            precision_scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)

        return np.mean(precision_scores) if average == 'macro' else np.sum(np.array(precision_scores) * (np.bincount(y_true) / len(y_true)))

def recall(y_true, y_pred, average='binary'):
    """Calculate recall: TP / (TP + FN)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    elif average in ['macro', 'micro', 'weighted']:
        unique_classes = np.unique(y_true)
        recall_scores = []

        for cls in unique_classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

        return np.mean(recall_scores) if average == 'macro' else np.sum(np.array(recall_scores) * (np.bincount(y_true) / len(y_true)))

def F1_Score(precision:float, recall:float):
    """This function caclulates f1-score from precision and recall
    """
    return (2 * precision * recall) / (precision + recall)

def calculate_all_metrics(y_truth:np.array, y_pred:np.array):
    """This function calcualtes all the classification score and print them
    """
    acc             = accuracy(y_truth, y_pred)
    precision_score = precision(y_truth, y_pred, 'macro')
    recall_score    = recall(y_truth, y_pred, 'macro')
    f1_score        = F1_Score(precision_score, recall_score)

    return acc, precision_score, recall_score, f1_score
   

def cosine_similarity(vec1, vec2):
    # Dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    
    # Magnitudes (norms) of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Cosine similarity formula
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cos_sim

import torch
import numpy as np
from tqdm import tqdm

from utils.metrics import calculate_all_metrics

def metric_evaluation(model,test_cls, test_ner):
    # Evaluation on test performance
    y_pred_cls, y_truth_cls = [], []
    y_pred_ner, y_truth_ner = [], []

    model.eval()
    for test_batch_cls, test_batch_ner in tqdm(zip(test_cls, test_ner)):

        # Pop the labels from input tokens id
        labels_cls =  test_batch_cls.pop('labels')
        labels_ner =  test_batch_ner.pop('labels')

        # Forward pass
        outputs_cls, _ = model(test_batch_cls)
        _, outputs_ner = model(test_batch_ner)

        # Classiciation Metric
        predict_cls = np.argmax(outputs_cls.detach().cpu().numpy(), axis=-1)
        y_pred_cls.append(predict_cls.squeeze())
        y_truth_cls.append(labels_cls.numpy().squeeze())

        # NER Metrics
        predicted_labels = np.argmax(outputs_ner.detach().cpu().numpy(), axis=-1).squeeze()

        # Remove the padded tokens with label=-100
        for ind, label in enumerate(labels_ner):
            mask = torch.tensor(label !=-100, dtype=torch.bool)   # Mask only the valid tokens
            y_pred_ner.append(predicted_labels[ind][mask])        # Add predicted labels to list
            y_truth_ner.append(labels_ner[ind][mask])             # Add ground truth labels to list

    y_pred_cls  = np.concatenate(y_pred_cls)
    y_truth_cls = np.concatenate(y_truth_cls)

    y_pred_ner  = np.concatenate(y_pred_ner)
    y_truth_ner = np.concatenate(y_truth_ner)
    
    acc_cls, precision_score_cls, recall_score_cls, f1_score_cls = calculate_all_metrics(y_pred_cls, y_truth_cls)
    acc_ner, precision_score_ner, recall_score_ner, f1_score_ner = calculate_all_metrics(y_pred_ner, y_truth_ner)

    return acc_cls, precision_score_cls, recall_score_cls, f1_score_cls, acc_ner, precision_score_ner, recall_score_ner, f1_score_ner