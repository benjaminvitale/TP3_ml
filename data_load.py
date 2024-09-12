import curacion as cc
import models as mods
import numpy as np
import metrics as mt
import matplotlib.pyplot as plt
Data1 = np.array(cc.Data_1)
Data2 = np.array(cc.Data_2)
Data3 = np.array(cc.Data_3)
Data4 = np.array(cc.Data_4)

t1 = np.array(cc.target_1)
t2 = np.array(cc.target_2)
t3 = np.array(cc.target_3)
t4 = np.array(cc.target_4)

Data_test = np.array(cc.Data_test)
target_test = np.array(cc.target_test)

log_regr = mods.LogisticRegression()
log_regr.fit(Data1,t1)
y_pred1 = log_regr.predict(Data_test)
y_proba1 = log_regr.predict_proba(Data_test)

log_regr2 = mods.LogisticRegression()
log_regr2.fit(Data2,t2)
y_pred2 = log_regr2.predict(Data_test)
y_proba2 = log_regr2.predict_proba(Data_test)

log_regr3 = mods.LogisticRegression()
log_regr3.fit(Data3,t3)
y_pred3 = log_regr3.predict(Data_test)
y_proba3 = log_regr3.predict_proba(Data_test)

log_regr4 = mods.LogisticRegression()
log_regr4.fit(Data4,t4)
y_pred4 = log_regr4.predict(Data_test)
y_proba4 = log_regr4.predict_proba(Data_test)






