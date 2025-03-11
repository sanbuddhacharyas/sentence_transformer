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
    precision_score = precision(y_truth, y_pred, 'macro')
    recall_score    = recall(y_truth, y_pred, 'macro')
    print(f"Accuracy: {accuracy(y_truth, y_pred)}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1: {F1_Score(precision_score, recall_score)}")

def cosine_similarity(vec1, vec2):
    # Dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    
    # Magnitudes (norms) of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Cosine similarity formula
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cos_sim