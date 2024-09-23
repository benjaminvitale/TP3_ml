import curacion as cc
import models as mods
import numpy as np
import metrics as mt
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd

Data1 = np.array(cc.Data_1)
Data2 = np.array(cc.Data_2)
Data3 = np.array(cc.Data_3)
Data4 = np.array(cc.Data_4)

t1 = np.array(cc.target_1)
t2 = np.array(cc.target_2)
t3 = np.array(cc.target_3)
t4 = np.array(cc.target_4)

lambdas = [0.1,0.5,1,5,10]  # Probar valores de lambda desde 10^-4 hasta 10^4
learning_rates = [0.0005,0.001, 0.01, 0.1, 1] 
threshold = [0.35,0.37,0.4,0.45]


Data_test1 = np.array(cc.Data_test1)
Data_test2 = np.array(cc.Data_test2)
Data_test3 = np.array(cc.Data_test3)
Data_test4 = np.array(cc.Data_test4)

target_test = np.array(cc.target_test)


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
y = []
log_regr1 = find_best(Data1,t1,Data_test1,0)

log_regr2 = find_best(Data2,t2,Data_test2,0)

log_regr3 = find_best(Data3,t3,Data_test3,0)

log_regr4 = find_best(Data4,t4,Data_test4,0)

log_regr5 = find_best(Data1,t1,Data_test1,1)


y_pred1 = log_regr1.predict(Data_test1)
y_proba1 = log_regr1.predict_proba(Data_test1)
y.append((y_pred1,y_proba1))

y_pred2 = log_regr2.predict(Data_test2)
y_proba2 = log_regr2.predict_proba(Data_test2)
y.append((y_pred2,y_proba2))

y_pred3 = log_regr3.predict(Data_test3)
y_proba3 = log_regr3.predict_proba(Data_test3)
y.append((y_pred3,y_proba3))

y_pred4 = log_regr4.predict(Data_test4)
y_proba4 = log_regr4.predict_proba(Data_test4)
y.append((y_pred4,y_proba4))

y_pred5 = log_regr5.predict(Data_test1)
y_proba5 = log_regr5.predict_proba(Data_test1)
y.append((y_pred5,y_proba5))


acc = []
precision = []
recall = []
f_score = []
auc_pr = []
auc_roc = []
names = ['Sin rebalanceo', 'Undersampling', 'Oversampling Duplicate', 'Oversampling Smote', 'Cost re weighting']
def mostrar_datos():
    print("Sin rebalanceo:")
    x = mt.get_metrics(target_test,y_proba1)
    auc_pr.append(x[0])
    auc_roc.append(x[1])

    print("Undersampling:")
    x = mt.get_metrics(target_test,y_proba2)
    auc_pr.append(x[0])
    auc_roc.append(x[1])
    print("Oversampling Duplicate:")
    x = mt.get_metrics(target_test,y_proba3)
    auc_pr.append(x[0])
    auc_roc.append(x[1])
    print("Oversampling Smote:")
    x = mt.get_metrics(target_test,y_proba4)
    auc_pr.append(x[0])
    auc_roc.append(x[1])
    print("Weighted Classes:")
    x = mt.get_metrics(target_test,y_proba5)
    auc_pr.append(x[0])
    auc_roc.append(x[1])
    counter = 0
    for i in y:
        mt.plot_confusion_matrix(mt.confusion_matrix(target_test, i[0]), ["1", "0"],names[counter])
        counter += 1
        #mt.calculate_metrics_custom(target_test,i[1])
        acc.append(mt.accuracy_score(target_test,i[0]))
        precision.append(mt.precision_score(target_test,i[0]))
        recall.append(mt.recall_score(target_test,i[0]))
        f_score.append(mt.f1_score(target_test,i[0]))

    # Datos de la tabla
    data = {
        'Modelo': [
            'Sin rebalanceo', 
            'Undersampling', 
            'Oversampling duplicate',
            'Oversampling smote',
            'Cost re weighting'
        ],
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F-Score': f_score,
        'auc-pr': auc_pr,
        'auc-roc': auc_roc
    }
    # Crear el DataFrame
    df = pd.DataFrame(data)

    # Mostrar la tabla en consola
    print(df)

