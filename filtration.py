#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filtration.py

This stores all filtration tools necessary for machine learning analysis. 

Created on: Wed Jul  7 13:16:35 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.

Functions:
    filter_by_variance_threshold:
        filters dataframe by removing variances <= 0.1
"""
import pandas as pd
## IMPORTING VARIANCE THRESHOLD FUNCTION
from sklearn.feature_selection import VarianceThreshold

# Function to filter by variance threshold
def filter_by_variance_threshold(X_df,
                                 variance_threshold = 0.1,
                                 verbose = True):
    '''
    This function filters based on variance threshold.
    INPUTS:
        X_df: [pd.DataFrame]
            pandas dataframe with the input data for training. 
    OUTPUTS:
        X_df: [pd.DataFrame]
            Updated dataframe with features within variance threshold removed
    '''
    ## GETTING VARIANCE THRESHOLD
    constant_filter = VarianceThreshold(threshold=variance_threshold)
    
    ## FITTING
    constant_filter.fit(X_df)
    
    ## IDENTIFYING CONSTANT COLS
    constant_columns = [column for column in X_df.columns
                        if column not in X_df.columns[constant_filter.get_support()]]
    
    ## GETTING COLUMNS
    if verbose is True:
        print("Filtering variance 0.1: Total features before and after: %d vs. %d"%(len(X_df.columns),
                                                                                    len(X_df.columns[constant_filter.get_support()])
                                                                                ) )
    
    ## DROPING CONSTANT
    output_df = X_df.drop(columns = constant_columns)
    
    return output_df