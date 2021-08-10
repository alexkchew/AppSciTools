#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qspr_post_analysis.py

This script contains all code necessary for post analysis after QSPR models have 
been developed.

Created on: Tue Aug 10 14:34:48 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.

Function list:
    detect_outliers_from_df:
        detects outliers from a dataframe

"""
import operator
import numpy as np

# Function to detect outliers given a dataframe of predicted and expected
def detect_outliers_from_df(predict_df,
                            criteria = ('abs_deviation', operator.gt, 0.07),
                            act_key = 'y_act',
                            pred_key = 'y_pred',
                            label_key = 'label',
                            ):
    """
    This function will detect outliers given a dataframe of predicted and 
    experimental values. 
    
    Args:
        predict_df (dataframe): 
            predicted and experimental values inside a dataframe
        criteria (tuple, shape = 3),
            criteria for classifying an outlier. Examples are:
                ('abs_deviation', '>', 0.07):
                    means that the absolute deviation between predicted 
                    and experimental values is greater than 0.07
            
        act_key (str, optional): 
            Key for the actual values. 
            Default value is 'y_act'
        pred_key (str, optional):
            Key for predicted values.
            Default value is 'y_pred'
        label_key (str, optional):
            Key for label values.
            Default value is 'label'
            
    Returns:
        outlier_df (dataframe):
            dataframe with all outlier information
    """
    # Copying the dataframe
    outlier_df = predict_df.copy()
    
    # Getting deviation
    outlier_df['abs_deviation'] = np.abs(predict_df[pred_key] - predict_df[act_key])
    
    # Using criteria to determine 
    outlier_df['outlier'] = criteria[1](outlier_df[criteria[0]], criteria[2])
    
    return outlier_df