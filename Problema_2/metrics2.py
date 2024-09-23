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







import numpy as np
import matplotlib.pyplot as plt

def binary_classification_metrics(y_true, y_pred_proba, thresholds):
    """
    Calcula las métricas TPR, FPR, Precision y Recall para un conjunto de umbrales.
    
    y_true: etiquetas verdaderas (n_samples,)
    y_pred_proba: probabilidades predichas para la clase positiva (n_samples,)
    thresholds: conjunto de umbrales a evaluar
    """
    tpr = []
    fpr = []
    precision = []
    recall = []
    
    for threshold in thresholds:
        # Convertir probabilidades en etiquetas (clasificación binaria)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calcular TP, FP, TN, FN
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calcular TPR y FPR
        tpr_value = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr_value = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        # Calcular Precision y Recall
        precision_value = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_value = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        tpr.append(tpr_value)
        fpr.append(fpr_value)
        precision.append(precision_value)
        recall.append(recall_value)
    
    return np.array(tpr), np.array(fpr), np.array(precision), np.array(recall)

def auc(x, y):
    """
    Calcula el área bajo la curva (AUC) usando la regla del trapecio.
    x: array de puntos en el eje X
    y: array de puntos en el eje Y
    """
    return np.trapz(y, x)

def plot_roc_curve(fpr, tpr):
    """Generar la gráfica de la curva ROC."""
    plt.figure()
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve(precision, recall):
    """Generar la gráfica de la curva Precision-Recall."""
    plt.figure()
    plt.plot(recall, precision, color='b', label=f'PR curve (AUC = {auc(recall, precision):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

# Ejemplo de uso para multiclase usando One-vs-Rest (OvR)
def calculate_multiclass_metrics_manual(y_true, y_pred_proba, classes):
    """
    Calcula métricas ROC y PR para problemas multiclase usando One-vs-Rest.
    
    y_true: etiquetas verdaderas (n_samples,)
    y_pred_proba: matriz de probabilidades predichas (n_samples, n_classes)
    classes: lista de clases
    """
    roc_aucs = []
    pr_aucs = []

    for i, class_ in enumerate(classes):
        print(f'Clase {class_}:')
        
        # Convertir etiquetas en binario (One-vs-Rest)
        y_true_bin = (y_true == class_).astype(int)
        y_pred_proba_class = y_pred_proba[:, i]
        
        # Umbrales desde 0.0 hasta 1.0
        thresholds = np.linspace(0, 1, num=100)
        
        # Calcular TPR, FPR, Precision y Recall
        tpr, fpr, precision, recall = binary_classification_metrics(y_true_bin, y_pred_proba_class, thresholds)
        
        # AUC-ROC y AUC-PR
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        
        # Graficar curvas ROC y PR
        plot_roc_curve(fpr, tpr)
        plot_pr_curve(precision, recall)
    
    # Promedios AUC-ROC y AUC-PR
    roc_auc_macro = np.mean(roc_aucs)
    pr_auc_macro = np.mean(pr_aucs)
    
    print(f'AUC-ROC (macro): {roc_auc_macro:.2f}')
    print(f'AUC-PR (macro): {pr_auc_macro:.2f}')
    
    return roc_auc_macro, pr_auc_macro

