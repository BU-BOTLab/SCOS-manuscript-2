# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:26:09 2024

@author: arianeg
"""

import numpy as np
from sklearn.feature_selection import SelectFromModel


def feat_selection(X_train, Y_train, X_test, Y_test, reg, feature_names, max_features):
    
    reg.fit(X_train, Y_train[:])

    selector = SelectFromModel(reg, threshold=-np.inf, max_features = max_features).fit(X_train, Y_train[:])


    select_X_train = selector.transform(X_train)
    select_X_test = selector.transform(X_test)

    
    return select_X_test, select_X_train 
    
