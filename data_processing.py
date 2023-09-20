# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:06:58 2023

@author: ferra
"""

# data_processing.py

def process_batch(X_batch, clf):
    return clf.predict_proba(X_batch)