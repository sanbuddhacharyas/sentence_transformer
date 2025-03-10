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
