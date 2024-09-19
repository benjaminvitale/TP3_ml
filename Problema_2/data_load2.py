import models2 as md2
import curacion2 as cc2
import metrics2 as mt2
import numpy as np
from collections import Counter

X_dev = np.array(cc2.X_dev)
X_test = np.array(cc2.X_test)

y_dev = np.array(cc2.Y_dev)
y_test = np.array(cc2.Y_test)

"""model = md2.LDA()
model.fit(X_dev,y_dev)
y_pred1 = model.predict(X_test)
y_score1 = model.predict_proba(X_test)
"""
model2 = md2.RandomForest()
model2.fit(X_dev,y_dev)
y_pred2 = model2.predict(X_test)
#y_score2 = model2.predict_proba(X_test)
y_score2 = []
unique, counts = np.unique(y_dev, return_counts=True)
#print(Counter(y_dev))

def find_best(x,y,x_test,c):
    best_model = None
    best_f1 = 0
    best_lambda = None
    best_lr = None
    
    for l2 in lambdas:
        for lr in learning_rates:
            for th in threshold:
                # Crear una instancia de tu modelo con el lambda y learning rate actuales
                model = mods.LogisticRegression(th, 2000, lr, l2)
                
                # Entrenar el modelo
                model.fit(x, y, c)
                
                # Hacer predicciones en el conjunto de validación
                y_val_pred = model.predict(x_test)
                
                # Calcular el F1-score
                f1 = mt.f1_score(target_test, y_val_pred)
                #f1 = sm.f1_score(target_test,y_val_pred)
                
                # Actualizar el mejor modelo si el F1-score mejora
                if f1 > best_f1:
                    best_f1 = f1
                    best_lambda = l2
                    best_lr = lr
                    best_model = model
    print(best_f1,best_lambda,best_lr)
    return best_model
def show_data(y_pred,y_test,y_score):
    num_classes = 3
    conf_matrix = mt2.confusion_matrix(y_test, y_pred, num_classes)
    print("Matriz de confusión:\n", conf_matrix)

    # Paso 2: Accuracy
    acc = mt2.accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Paso 3: Precision, Recall y F1-Score
    precision, recall, f1 = mt2.precision_recall_fscore(conf_matrix)
    print("Precision por clase:", precision)
    print("Recall por clase:", recall)
    print("F1-Score por clase:", f1)


    """roc_points, auc_roc = mt2.roc_curve_multiclass(y_test, y_score, num_classes)
    print("AUC-ROC por clase:", auc_roc)

    # Paso 5: Curvas PR y AUC-PR
    pr_points, auc_pr = mt2.pr_curve_multiclass(y_test, y_score, num_classes)
    print("AUC-PR por clase:", auc_pr)"""
show_data(y_pred2,y_test,y_score2)