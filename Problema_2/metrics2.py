import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def confusion_matrix(y_true, y_pred, num_classes):
    """
    Calcula la matriz de confusión para un problema de múltiples clases.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def accuracy_score(y_true, y_pred):
    """
    Calcula la precisión (accuracy) general.
    """
    correct = np.sum(np.array(y_true) == np.array(y_pred))
    total = len(y_true)
    return correct / total


def precision_recall_fscore(conf_matrix):
    """
    Calcula la precisión, el recall y el F1-score por clase.
    """
    num_classes = conf_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        true_positive = conf_matrix[i, i]
        false_positive = np.sum(conf_matrix[:, i]) - true_positive
        false_negative = np.sum(conf_matrix[i, :]) - true_positive

        precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1_score



def roc_curve_multiclass(y_true, y_score, num_classes):
    """
    Calcula puntos de la curva ROC y AUC para cada clase.
    """
    roc_points = []
    auc = np.zeros(num_classes)
    
    for i in range(num_classes):
        # Tratamos la clase i como clase positiva y las demás como negativas (One-vs-All)
        y_true_bin = np.where(y_true == i, 1, 0)  # Etiquetas binarizadas
        y_score_bin = y_score[:, i]  # Puntajes para la clase i

        # Ordenamos las predicciones por confianza
        thresholds = np.sort(np.unique(y_score_bin))[::-1]
        tpr = []  # True Positive Rate
        fpr = []  # False Positive Rate
        
        for threshold in thresholds:
            y_pred_bin = np.where(y_score_bin >= threshold, 1, 0)
            tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
            fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
            tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
            fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        roc_points.append((fpr, tpr))
        auc[i] = np.trapz(tpr, fpr)  # AUC-ROC por clase
    
    return roc_points, auc

def pr_curve_multiclass(y_true, y_score, num_classes):
    """
    Calcula puntos de la curva PR y AUC para cada clase.
    """
    pr_points = []
    auc = np.zeros(num_classes)
    
    for i in range(num_classes):
        y_true_bin = np.where(y_true == i, 1, 0)
        y_score_bin = y_score[:, i]
        
        thresholds = np.sort(np.unique(y_score_bin))[::-1]
        precision = []
        recall = []

        for threshold in thresholds:
            y_pred_bin = np.where(y_score_bin >= threshold, 1, 0)
            tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
            fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
            fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

            precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

        pr_points.append((recall, precision))
        auc[i] = np.trapz(precision, recall)  # AUC-PR por clase
    
    return pr_points, auc
