# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:29:26 2023

@author: arianeg
"""

import numpy as np
def BP_avg(window_size, stride, Y_pred, Y_real):
    ''' Function to apply a moving average to arrays'''
    
    
    Y1 = np.array([np.nanmean(Y_pred[i:i+window_size]) for i in range(0, len(Y_pred), stride) if i+window_size <= len(Y_pred) ])


    Y2 = np.array([np.nanmean(Y_real[i:i+window_size]) for i in range(0, len(Y_real), stride) if i+window_size <= len(Y_real) ])


    return Y1, Y2 