import csv 
import numpy as np
from imblearn.over_sampling import SMOTE 




def open_csv(direccion):
    with open(direccion, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        i = 0
        Datas = []
        y = []
        
        for row in csv_reader:
            if i != 0:
                Datas.append(row[1:])
                y.append(int(float(row[0])))


            i += 1
        for row in range(len(Datas)):
            for j in range(len(Datas[0])):
                Datas[row][j] = float(Datas[row][j])
        return Datas,y

X_dev,Y_dev = open_csv('Problema_2/diabetes_data/diabetes_dev.csv')
X_test,Y_test = open_csv('Problema_2/diabetes_data/diabetes_test.csv')

#smote = SMOTE(sampling_strategy='auto', random_state=42)
#X_dev,Y_dev = smote.fit_resample(X_dev, Y_dev)
def min_max_scaling(X):
    vals_min = []
    vals_max = []

    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    vals_min.append(min_val)
    vals_max.append(max_val)
    
    # Evitar división por cero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    X_scaled = (X - min_val) / range_val
    return X_scaled,np.array(vals_min),np.array(vals_max)

def normalize_test(X,vals_min,vals_max):
    X_scaled = (X - vals_min) / (vals_max - vals_min)
    return X_scaled

X_dev,min,max = min_max_scaling(X_dev)
X_test = normalize_test(X_test,min,max)
