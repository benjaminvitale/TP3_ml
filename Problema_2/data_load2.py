import models2 as md2
import curacion2 as cc2
import metrics2 as mt2
import numpy as np
from collections import Counter

X_dev = np.array(cc2.X_dev)
X_test = np.array(cc2.X_test)

y_dev = np.array(cc2.Y_dev)
y_test = np.array(cc2.Y_test)



unique, counts = np.unique(y_dev, return_counts=True)
print(Counter(y_dev))
depths = [3,4]
leaf_nums = [3,4]
infos_gain = [0.0,0.1]
entropy = [1e7,1e8]
def find_best(x,y,x_test):
    best_model = None
    best_f1 = 0
    best_leaf = None
    best_depth = None
    best_info = None
    best_entropy = None
    
    for depth in depths:
        for leaf_num in leaf_nums:
            for info in infos_gain:
                for entropy_ in entropy:
                # Crear una instancia de tu modelo con el lambda y learning rate actuales
                    model = md2.RandomForest(10, depth, leaf_num, info,entropy_)
                    #model = md2.DecisionTree(depth,leaf_num,info,entropy_)
                    # Entrenar el modelo
                    model.fit(x, y)
                    
                    # Hacer predicciones en el conjunto de validación
                    y_val_pred = model.predict(x_test)
                    
                    # Calcular el F1-score
                    num_classes = 3
                    conf_matrix = mt2.confusion_matrix(y_test, y_val_pred, num_classes)

                    f1 = mt2.precision_recall_fscore(conf_matrix)[2]
                    #f1 = sm.f1_score(target_test,y_val_pred)
                    
                    # Actualizar el mejor modelo si el F1-score mejora
                    if np.mean(f1) > best_f1:
                        best_f1 = np.mean(f1)
                        best_depth = depth
                        best_leaf = leaf_num
                        best_model = model
                        best_info = info
                        best_entropy = entropy_
    print(best_f1,best_depth,best_leaf,best_info,best_entropy)
    return best_model
def find_logistic(x,y):
    best_model = None
    best_f1 = 0
    best_lambda = None
    best_lr = None
    lambdas = [0.01,0.1,0.5]
    learning_rates = [0.01,0.1]

    
    for l2 in lambdas:
        for lr in learning_rates:
            # Crear una instancia de tu modelo con el lambda y learning rate actuales
            model = md2.LogisticRegressionMulticlass(1000, lr, l2)
            
            # Entrenar el modelo
            model.fit(x, y)
            
            # Hacer predicciones en el conjunto de validación
            y_val_pred = model.predict(X_test)
            num_classes = 3
            conf_matrix = mt2.confusion_matrix(y_test, y_val_pred, num_classes)

            f1 = mt2.precision_recall_fscore(conf_matrix)[2]
            
            # Calcular el F1-score
            
            # Actualizar el mejor modelo si el F1-score mejora
            if np.mean(f1) > best_f1:
                best_f1 = np.mean(f1)
                best_lambda = l2
                best_lr = lr
                best_model = model
    print(best_f1,best_lambda,best_lr)
    return best_model 

model = md2.LDA()
model.fit(X_dev,y_dev)
y_pred1 = model.predict(X_test)
y_score1 = model.predict_proba(X_test)

model2= find_best(X_dev,y_dev,X_test)
y_pred2 = model2.predict(X_test)
y_score2 = model2.predict_proba(X_test)



model3 = find_logistic(X_dev,y_dev)
model3.fit(X_dev,y_dev)
y_pred3 = model3.predict(X_test)
y_score3 = model3.predict_proba(X_test)


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

print("metricas para LDA:")
show_data(y_pred1,y_test,y_score1)

roc_auc_macro, pr_auc_macro = mt2.calculate_multiclass_metrics_manual(y_test, y_score1, np.unique(y_test))


print("metricas para random forest:")
show_data(y_pred2,y_test,y_score2)

roc_auc_macro, pr_auc_macro = mt2.calculate_multiclass_metrics_manual(y_test, y_score2, np.unique(y_test))

print("Metricas para regresion logistica:")
show_data(y_pred3,y_test,y_score3)

roc_auc_macro, pr_auc_macro = mt2.calculate_multiclass_metrics_manual(y_test, y_score3, np.unique(y_test))
