# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:20:56 2023

@author: jbrousseau
"""
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import Functions as c

model_df = pd.read_csv("workout_sleep_classifier_model_dataset.csv")
model_df = model_df.drop("Unnamed: 0",axis=1)

## train model
outcome = model_df['BelowAvg']
predictors = model_df.loc[:, model_df.columns != 'BelowAvg']

X_train, X_test, y_train, y_test = train_test_split(predictors, outcome, test_size=0.4, random_state=42)

#lets do xgboost
sleep_workout_xgb = xgb.XGBClassifier()
sleep_workout_xgb.fit(X_train, y_train)

y_pred = sleep_workout_xgb.predict(X_test)


tree_text_class = xgb.to_graphviz(sleep_workout_xgb,  rankdir='LR', format='png')
conf_matrix_xgb = confusion_matrix(y_test, y_pred,labels=sleep_workout_xgb.classes_)

cm_disp_xgb = ConfusionMatrixDisplay(conf_matrix_xgb)
cm_disp_xgb.plot()
plt.grid(False)
plt.show()

c.cm_metrics(y_test,y_pred)

# Tune
param_grid = {
    'max_depth': [3,4,5],
    'learning_rate': [.1,.01,.05],
    'gamma': [0,.25,.1],
    'reg_lambda': [0,1,10],
    'scale_pos_weight':[1,3,5]
    }

optimal_params = GridSearchCV(
    estimator = xgb.XGBClassifier(),
    param_grid=param_grid,
    scoring = 'roc_auc',
    n_jobs=10,
    cv=3
    )

optimal_params.fit(X_train, y_train)
optimal_params.best_params_


param_grid2 = {
    'max_depth': [2,3,4],
    'learning_rate': [.075,.05,.025],
    'gamma': [0,.05,.1],
    'reg_lambda': [10,12,14],
    'scale_pos_weight':[5,7,9]
    }

optimal_params2 = GridSearchCV(
    estimator = xgb.XGBClassifier(),
    param_grid=param_grid2,
    scoring = 'roc_auc',
    n_jobs=10,
    cv=3
    )

optimal_params2.fit(X_train, y_train)
optimal_params2.best_params_


param_grid3 = {
    'max_depth': [3],
    'learning_rate': [.05],
    'gamma': [0],
    'reg_lambda': [12],
    'scale_pos_weight':[7,9,11]
    }

optimal_params3 = GridSearchCV(
    estimator = xgb.XGBClassifier(),
    param_grid=param_grid3,
    scoring = 'roc_auc',
    n_jobs=10,
    cv=3
    )

optimal_params3.fit(X_train, y_train)
optimal_params3.best_params_



clf_xgb_tuned = xgb.XGBClassifier(seed=42
                            ,learn_rate=.05
                            ,max_depth=3
                            #,gamma=0
                            ,reg_lambda=13
                            # ,scale_pos_weight=9
                            ,n_estimators = 1
                            )
clf_xgb_tuned.fit(X_train, y_train)

y_pred_tuned = clf_xgb_tuned.predict(X_test)




conf_matrix = confusion_matrix(y_test, y_pred_tuned)

print("Confusion Matrix:")
print(conf_matrix)
c.cm_metrics(y_test,y_pred_tuned)
