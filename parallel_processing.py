# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:07:30 2023

@author: ferra
"""

# parallel_processing.py
from joblib import Parallel, delayed, cpu_count
from data_processing import process_batch

def process_batch_parallel(X_batch, clf):
    return process_batch(X_batch, clf)

def parallel_process_batch(X_batch_list, clf, n_cores):
    # n_cores = cpu_count()
    return Parallel(n_jobs=n_cores)(delayed(process_batch_parallel)(X_batch, clf) for X_batch in X_batch_list)
