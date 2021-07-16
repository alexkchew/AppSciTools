#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qspr_models.py

This script contains all qspr functions. 

Created on: Fri Jul 16 14:53:32 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.
"""

# Importing modules
import numpy as np
import pandas as pd
import copy

# 5-CV cross validation functions
from sklearn.model_selection import KFold

# Standardization procedures (maybe not necessary)
from sklearn.preprocessing import StandardScaler

# Generate models
from lightgbm import LGBMRegressor # Module isn't working correctly at this time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

# Setting random seed
np.random.seed(0)

# Defining defaults
RF_DEFAULTS = {
    'n_estimators': 200,
    }

# Function to split the dataset
def split_dataset_KFolds(X, 
                         y, 
                         labels, 
                         n_folds = 5,
                         verbose = False):
    """
    This function generates datasets for k folds

    Parameters
    ----------
    X : np.array, shape = N, M
        Contains all input values for QSPR models
    y : np.array, shape = N
        Contains all output values for QSPR models
    labels : np.array
        Contains all labels for each row of the model
    n_folds : str, optional
        Number of desired folds. The default is 5.
    verbose: logical, optional
        Prints out train / test set split verbosely. The default value is False.
    Returns
    -------
    fold_list : list
        list of each fold with the corresponding X, y, and labels

    """
    # Getting k folds
    kf = KFold(n_splits=n_folds) # Turning off shuffling
    # Getting the splits
    kf.get_n_splits(X)
    
    # Storing fold list
    fold_list = []
    
    # Generating trainind and validation sets
    for train_index, test_index in kf.split(X):
        if verbose is True:
            print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Getting laels
        labels_train, labels_test = labels[train_index], labels[test_index]
        
        # Storing the information
        output_dict = dict(
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            labels_train = labels_train, 
            labels_test = labels_test,
            train_index = train_index,
            test_index = test_index
            )
        
        # Storing fold list
        fold_list.append(output_dict)
        
    return fold_list

# Function to generate model based on type
def generate_model_based_on_type(model_type):
    """
    This function generates a model based on the type that you want. 

    Parameters
    ----------
    model_type : str
        type of model that you want. The full list is shown below:
            'RF':
                random forest regressor
            'lightgbm':
                Light gradient boosted machine learning model, e.g.
                    https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/
            'linear':
                Classical linear regression model
            'lasso':
                Lasso model
            'svr':
                support vector regression

    Returns
    -------
    model: [obj]
        model object

    """
    # Iteratively find model types
    if model_type == 'RF':
        model = RandomForestRegressor(**RF_DEFAULTS)
    elif model_type == 'lightgbm':
        model = LGBMRegressor() # **RF_DEFAULTS
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'lasso':
        model = Lasso(max_iter=10000000)
    elif model_type == 'svr':
        model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    else:
        print("Error! Model type (%s) not found!"%(model_type))
    return model

# Function to generate predictions for each model
def predict_for_KFolds(model_type,
                       fold_list,
                       return_models = False):
    """
    This function predicts using K-folds.

    Parameters
    ----------
    model: obj
        model to use for the K-fold predictions
    fold_list: list
        list of training and testing sets
    return_models: logical, optional
        True if you want to return the models for each fold. Default is False.
    Returns
    -------
    predict_df: df
        prediction dataframe

    """
    # Storing dataframe
    predict_df = []
    
    # Storing models
    if return_models is True:
        model_list = []
    
    # Looping through each
    for train_test_dict in fold_list:
        # Creating a new model
        model = generate_model_based_on_type(model_type = model_type)
        
        # Fitting model
        model.fit(train_test_dict['X_train'],
                  train_test_dict['y_train'])
        
        # Predicting test set
        yhat = model.predict(train_test_dict['X_test'])
        
        # Creating dataframe
        df = pd.DataFrame(np.array([yhat, train_test_dict['y_test'], train_test_dict['labels_test']]).T,
                          columns = ['y_pred', 'y_act', 'label']
                          )
        
        # Storing dataframes
        predict_df.append(df)
        
        # Storing model
        if return_models is True:
            model_list.append(copy.deepcopy(model))
        
    # Concat for all
    predict_df = pd.concat(predict_df, axis = 0)
        
    if return_models is True:
        return predict_df, model_list
    else:
        return predict_df
    

