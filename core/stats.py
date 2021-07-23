#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py

This script holds all the statistic functions. 

Created on: Fri Jul 16 14:55:52 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.
"""

import numpy as np

# Importing statistics
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# Function to get statistics
def get_stats(predict_df,
              df_act_key='y_act',
              df_pred_key='y_pred',
              ):
    """
    This function generates statistics for a prediction. It will output 
    a dictionary with many parameters used to deduce the strength of the 
    prediction. 

    Parameters
    ----------
    predict_df : dataframe
        Prediction dataframe containing 'y_act' and 'y_pred'
    df_act_key: str, optional
        actual key for dataframe. The default value is 'y_act'.
    df_pred_key: str, optional
        prediction key for dataframe. The default value is 'y_pred'.

    Returns
    -------
    stats_dict: [dict]
        dictionary containing statistics

    """
    # Defining x and y 
    actual = predict_df[df_act_key]
    pred = predict_df[df_pred_key]
    
    # Defining stats dict
    stats_dict = {
        'r2': r2_score(actual, pred),
        'mse': mean_squared_error(actual, pred),
        'pearsonr': pearsonr(actual, pred)[0],
        }
    
    # Adding RMSE
    stats_dict['rmse'] = np.sqrt(stats_dict['mse'])
    
    return stats_dict
