# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:31:10 2023

@author: jbrousseau
"""

from tabulate import tabulate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score

#Get Confusion Matrix Metrics
def cm_metrics(y_test,predicted_labels):
    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    sensitivity = recall_score(y_test, predicted_labels)
    auc_roc = roc_auc_score(y_test, predicted_labels)
    tn_log_nt, fp_log_nt, fn_log_nt, tp_log_nt = confusion_matrix(y_test, predicted_labels).ravel()
    specificity = tn_log_nt / (tn_log_nt + fp_log_nt)
    
    all_metrics = [
                    ["Accuracy",accuracy],
                    ["Precision",precision],
                    ["Sensitivity",sensitivity],
                    ["Specificity",specificity],
                    ["AUC",auc_roc]
                    ]
    headers = ['Metric',"Score"]
    
    print(tabulate(all_metrics, headers=headers, tablefmt="grid"))