import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix_custom(y_true, y_pred):
    """
    Calcula la matriz de confusión.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def accuracy_custom(y_true, y_pred):
    """
    Calcula la exactitud (accuracy).
    """
    return np.mean(y_true == y_pred)

def precision_custom(y_true, y_pred):
    """
    Calcula la precisión (precision).
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall_custom(y_true, y_pred):
    """
    Calcula el recall.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score_custom(y_true, y_pred):
    """
    Calcula el F1-Score.
    """
    precision = precision_custom(y_true, y_pred)
    recall = recall_custom(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def roc_curve_custom(y_true, y_prob):
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
    """
    Calcula la curva de precisión-recall.
    """
    thresholds = np.sort(np.unique(y_prob))[::-1]
    precision_vals = []
    recall_vals = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        precision = precision_custom(y_true, y_pred)
        recall = recall_custom(y_true, y_pred)
        precision_vals.append(precision)
        recall_vals.append(recall)
    
    return np.array(recall_vals), np.array(precision_vals)

# Función principal para calcular y graficar las métricas
def calculate_metrics_custom(y_true, y_pred, y_prob):
    # Matriz de confusión
    cm = confusion_matrix_custom(y_true, y_pred)
    print("Matriz de Confusión:")
    print(cm)

    # Exactitud (Accuracy)
    accuracy = accuracy_custom(y_true, y_pred)
    print(f"\nExactitud (Accuracy): {accuracy:.4f}")

    # Precisión (Precision)
    precision = precision_custom(y_true, y_pred)
    print(f"Precisión (Precision): {precision:.4f}")

    # Recall
    recall = recall_custom(y_true, y_pred)
    print(f"Recall: {recall:.4f}")

    # F1-Score
    f1 = f1_score_custom(y_true, y_pred)
    print(f"F1-Score: {f1:.4f}")
    
    # Curva ROC y AUC-ROC
    fpr, tpr = roc_curve_custom(y_true, y_prob)
    auc_roc = auc_custom(fpr, tpr)
    print(f"\nAUC-ROC: {auc_roc:.4f}")
    
    # Curva PR y AUC-PR
    recall_vals, precision_vals = precision_recall_curve_custom(y_true, y_prob)
    auc_pr = auc_custom(recall_vals, precision_vals)
    print(f"AUC-PR: {auc_pr:.4f}")
    
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

