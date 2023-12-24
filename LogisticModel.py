# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:04:54 2023

@author: jbrousseau
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn import linear_model
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import Functions as c

model_df = pd.read_csv("workout_sleep_classifier_model_dataset.csv")
model_df = model_df.drop("Unnamed: 0",axis=1)

## train model
outcome = model_df['BelowAvg']
predictors = model_df.loc[:, model_df.columns != 'BelowAvg']

X_train, X_test, y_train, y_test = train_test_split(predictors, outcome, test_size=0.4, random_state=42)
### logistic
logr = linear_model.LogisticRegression()
logr.fit(X_train,y_train)

y_pred_log = logr.predict(X_test)


c.cm_metrics(y_test,y_pred_log)
conf_matrix_log = confusion_matrix(y_test, y_pred_log)

cm_disp_log = ConfusionMatrixDisplay(conf_matrix_log)
cm_disp_log.plot()
plt.grid(False)
plt.show()

# can try oversampling using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Resample the training data using SMOTE
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a model on the resampled training data
model_smote = linear_model.LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

# Make predictions on the test data
y_pred_smote = model_smote.predict(X_test)

c.cm_metrics(y_test,y_pred_smote)
conf_matrix_log_smote = confusion_matrix(y_test, y_pred_smote)

cm_disp_log_smote = ConfusionMatrixDisplay(conf_matrix_log_smote)
cm_disp_log_smote.plot()
plt.grid(False)
plt.show()

# can try adjusting decision threshold
y_pred_smote_p = model_smote.predict_proba(X_test)[:, 1]

# Define the new threshold
new_threshold = 0.44  # You can set this to your desired threshold

# Create an array of predicted labels based on the new threshold
predicted_labels = (y_pred_smote_p > new_threshold).astype(int)

c.cm_metrics(y_test,predicted_labels)

conf_matrix_log_nt = confusion_matrix(y_test, predicted_labels)
cm_disp_log_nt = ConfusionMatrixDisplay(conf_matrix_log_nt)
cm_disp_log_nt.plot()
plt.grid(False)
plt.show()