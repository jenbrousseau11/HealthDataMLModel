# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:55:23 2023

@author: jbrousseau
"""
import os


import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import Functions as c
from sklearn.model_selection import train_test_split

model_df = pd.read_csv("workout_sleep_classifier_model_dataset.csv")
model_df = model_df.drop("Unnamed: 0",axis=1)

## train model
outcome = model_df['BelowAvg']
predictors = model_df.loc[:, model_df.columns != 'BelowAvg']

X_train, X_test, y_train, y_test = train_test_split(predictors, outcome, test_size=0.4, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


c.cm_metrics(y_test,y_pred_rf)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
cm_disp_rf = ConfusionMatrixDisplay(conf_matrix_rf)
cm_disp_rf.plot()
plt.grid(False)
plt.show()

for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)