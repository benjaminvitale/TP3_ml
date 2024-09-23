import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(y_true, y_pred):
    """
    Calcula la matriz de confusión.
    y_true: Vector de etiquetas verdaderas
    y_pred: Matriz de probabilidades predichas
    """
    y_true = np.asarray(y_true)
    y_true.reshape(-1,1)
    y_pred = np.asarray(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    tp = np.sum((y_true == 1) & (y_pred_classes == 1))
    fp = np.sum((y_true == 0) & (y_pred_classes == 1))
    fn = np.sum((y_true == 1) & (y_pred_classes == 0))
    tn = np.sum((y_true == 0) & (y_pred_classes == 0))
    return np.array([[tp, fp], [fn, tn]])


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    correct_predictions = np.sum(y_true == y_pred_classes)
    return round(correct_predictions / len(y_true),3)



def precision_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true.reshape(-1,1)
    
    y_pred_classes = np.argmax(y_pred,axis=1)
    
    tp = np.sum((y_true == 1) & (y_pred_classes == 1))
    fp = np.sum((y_true == 0) & (y_pred_classes == 1))
    return round(tp / (tp + fp),3) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true.reshape(-1,1)
    
    y_pred_classes = np.argmax(y_pred,axis=1)
    
    tp = np.sum((y_true == 1) & (y_pred_classes == 1))
    fn = np.sum((y_true == 1) & (y_pred_classes == 0))
    return round(tp / (tp + fn),3) if (tp + fn) > 0 else 0


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return round(2 * (precision * recall) / (precision + recall),2) if (precision + recall) > 0 else 0


def roc_curve_custom(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true.reshape(-1,1)
    """
    Calcula la curva ROC.
    """

    thresholds = np.sort(np.unique(y_prob))[::-1]
    TPR = []  # True Positive Rate
    FPR = []  # False Positive Rate

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        
        TPR.append(TP / (TP + FN) if (TP + FN) > 0 else 0)  # Sensitivity
        FPR.append(FP / (FP + TN) if (FP + TN) > 0 else 0)  # 1 - Specificity
    
    return np.array(FPR), np.array(TPR)

def auc_custom(x, y):
    """
    Calcula el área bajo la curva (AUC) usando el método del trapecio.
    """
    return np.trapz(y, x)

def precision_recall_curve_custom(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true.reshape(-1,1)
    """
    Calcula la curva de precisión-recall.
    """
    thresholds = np.sort(np.unique(y_prob))[::-1]
    precision_vals = []
    recall_vals = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision_vals.append(precision)
        recall_vals.append(recall)
    
    return np.array(recall_vals), np.array(precision_vals)



def plot_confusion_matrix(conf_matrix, class_names, tittle):
    """
    Plotea la matriz de confusión utilizando seaborn y matplotlib.
    conf_matrix: Matriz de confusión
    class_names: Lista con los nombres de las clases
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión' + tittle)
    plt.show()






def compute_roc_curve(y_true, y_proba):

    thresholds = np.sort(np.unique(y_proba))[::-1]  # Ordenar umbrales de mayor a menor
    tpr_list = []  # True Positive Rate
    fpr_list = []  # False Positive Rate
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)  # Predicciones binarizadas con el umbral actual
        tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positives
        fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positives
        fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negatives
        tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negatives
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensibilidad o TPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds

def compute_auc(x, y):

    return np.trapz(y, x)  # Calcula el área usando la regla del trapecio

def compute_precision_recall_curve(y_true, y_proba):

    thresholds = np.sort(np.unique(y_proba))[::-1]  # Ordenar umbrales de mayor a menor
    precision_list = []
    recall_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)  # Predicciones binarizadas con el umbral actual
        tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positives
        fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positives
        fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    return np.array(recall_list), np.array(precision_list), thresholds

def get_metrics(y_true, y_proba):
    y_proba_positive = y_proba[:, 1] 

    # ROC
    fpr, tpr, roc_thresholds = compute_roc_curve(y_true, y_proba_positive)
    auc_roc = compute_auc(fpr, tpr)

    # Precision-Recall
    recall, precision, pr_thresholds = compute_precision_recall_curve(y_true, y_proba_positive)
    auc_pr = compute_auc(recall, precision)

    # Resultados
    metrics = {
        "ROC_curve": (fpr, tpr, roc_thresholds),
        "AUC_ROC": auc_roc,
        "PR_curve": (recall, precision, pr_thresholds),
        "AUC_PR": auc_pr
    }

    # ROC Curve
    fpr, tpr, _ = metrics["ROC_curve"]
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['AUC_ROC']:.2f})")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # PR Curve
    recall, precision, _ = metrics["PR_curve"]
    plt.plot(recall, precision, label=f"PR Curve (AUC = {metrics['AUC_PR']:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
    return metrics["AUC_PR"],metrics["AUC_ROC"]
        




   
 
