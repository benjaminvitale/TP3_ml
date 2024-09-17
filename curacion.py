import csv
import random
from imblearn.over_sampling import SMOTE 
import numpy as np
Data_1 = []
Data_2 = []
Data_3 = []
Data_4 = []

target_1 = []
target_2 = []
target_3 = []
target_4 = []

Data_test = []

target_test = []


with open('breast_cancer_data/breast_cancer_dev.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    i = 0
    
    for row in csv_reader:
        if i != 0:
            Data_1.append(row[:-1])
            Data_2.append(row[:-1])
            Data_3.append(row[:-1])
            Data_4.append(row[:-1])
            target_1.append(int(row[-1]))
            target_2.append(int(row[-1]))
            target_3.append(int(row[-1]))
            target_4.append(int(row[-1]))

        i += 1

cont = 0
cont2 = 0
for i in target_1:
    if i == 0:
        cont += 1
    if i == 1:
        cont2 += 1
while len(Data_2) > 172:
    i = random.randint(0,len(Data_2)-1)
    if target_2[i] == 0:
        Data_2.pop(i)
        target_2.pop(i)
while len(Data_3) < 776:
    i = random.randint(0,len(Data_3)-1)
    if target_3[i] == 1:
        Data_3.append(Data_3[i])
        target_3.append(target_3[i])
for i in range(len(Data_4)):
    for j in range(6):
        Data_1[i][j] = float(Data_1[i][j])
        Data_4[i][j] = float(Data_4[i][j])
for i in range(len(Data_3)):
    for j in range(6):
        Data_3[i][j] = float(Data_3[i][j])

for i in range(len(Data_2)):
    for j in range(6):
        Data_2[i][j] = float(Data_2[i][j])

smote = SMOTE(sampling_strategy='auto', random_state=42)
Data_4, target_4 = smote.fit_resample(Data_4, target_4)


with open('breast_cancer_data/breast_cancer_test.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    i = 0
    
    for row in csv_reader:
        if i != 0:
            Data_test.append(row[:-1])
            target_test.append(int(row[-1]))
        i += 1


for i in range(len(Data_test)):
    for j in range(6):
        Data_test[i][j] = float(Data_test[i][j])

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

Data_1,min_1,max_1 = min_max_scaling(Data_1)
Data_2,min_2,max_2 = min_max_scaling(Data_2)
Data_3,min_3,max_3 = min_max_scaling(Data_3)
Data_4,min_4,max_4 = min_max_scaling(Data_4)

Data_test1 = normalize_test(Data_test,min_1,max_1)
Data_test2 = normalize_test(Data_test,min_2,max_2)
Data_test3 = normalize_test(Data_test,min_3,max_3)
Data_test4 = normalize_test(Data_test,min_4,max_4)