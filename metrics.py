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



def plot_confusion_matrix(conf_matrix, class_names):
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
    plt.title('Matriz de Confusión')
    plt.show()


# Función principal para calcular y graficar las métricas
def calculate_metrics_custom(y_true,y_prob):
    # Curva ROC y AUC-ROC
    fpr, tpr = roc_curve_custom(y_true, y_prob)
    auc_roc = auc_custom(fpr, tpr)
    
    # Curva PR y AUC-PR
    recall_vals, precision_vals = precision_recall_curve_custom(y_true, y_prob)
    auc_pr = auc_custom(recall_vals, precision_vals)
    
    # Graficar Curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (False Positive Rate)')
    plt.ylabel('Tasa de Verdaderos Positivos (True Positive Rate)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

    # Graficar Curva PR
    plt.figure()
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'Curva PR (AUC = {auc_pr:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precisión (Precision)')
    plt.title('Curva PR')
    plt.legend(loc="lower left")
    plt.show()
    

