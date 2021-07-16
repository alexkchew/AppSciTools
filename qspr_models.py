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

# Importing relative stats
from .core.stats import get_stats

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

# Function to strip title and etc to get numerical descriptors only
def strip_df_index(df,
                   col2remove = DEFAULT_INDEX_COLS):
    """
    This function strips the dataframe from the index information. 

    Parameters
    ----------
    df : dataframe
        pandas dataframe containing descriptor information. 
    col2remove: list]
        list of columns to remove from the dataframe.
    Returns
    -------
    df_clean: dataframe]
        pandas dataframe without any "Title" or index information
    """
    # Dropping the columns
    df_clean = df.drop(columns = col2remove,
                       errors='ignore')
    
    return df_clean

# Function to generate X_df from descriptor list
def generate_X_df_from_descriptor_list(descriptor_list):
    """
    This function generates the combined dataframe from descriptor list. 
    This assumes that all descriptors have been generated with the same 
    indexes. We may need to check this in the future!

    Parameters
    ----------
    descriptor_list : list
        List of descriptors that are desired.

    Returns
    -------
    X_df: [df]
        dataframe containing all descriptors
    """
    
    # Getting descriptors
    descriptor_df_dict = load_multiple_descriptor_data(descriptor_list = descriptor_list)

    # Cleaning each dictionary
    descriptor_df_clean = {each_key: strip_df_index(descriptor_df_dict[each_key]) 
                                              for each_key in descriptor_df_dict}
    
    # Combining dataframes
    X_df = pd.concat([descriptor_df_clean[each_key] for each_key in descriptor_df_clean], axis=1)
    
    return X_df
    

# Defining main function to perform calculations
def main_generate_qspr_models_CV(descriptor_keys_to_use,
                                 output_property_list,
                                 want_normalize = True,
                                 model_type_list = ['RF']):
    """
    This function runs QSPR models and outputs prediction accuracies using 
    K-fold cross validation. Note! We are assuming that the columns in X and 
    y data are in the same order. We may need to double-check this in the future.

    Parameters
    ----------
    descriptor_keys_to_use: list
        List of descriptor keys to use
    output_property_list: list
        list of output properties
    want_normalize: logical
        True if you want to normalize the X array by subtracting mean and dividing by 
        standard deviation. 
    model_type_list: [list]
        list of models that you want to perform.
    Returns
    -------
    storage_descriptor_sets: [dict]
        dictionary of dicts that store all information, e.g. [property][descriptor set]

    """

    # Defining storage for each descriptor
    storage_descriptor_sets = {}
    
    # Looping through property labels
    for property_label in output_property_list:
        
        # Creating empty dictionary for property labels
        storage_descriptor_sets[property_label] = {}
        
        # Looping through each dataframe key
        for descriptor_df_key in descriptor_keys_to_use:
            # Getting descriptor list
            descriptor_list = DESCRIPTOR_DICT[descriptor_df_key]['descriptor_list']
            
            # Generating combined dataframe
            X_df = generate_X_df_from_descriptor_list(descriptor_list = descriptor_list)
            
            # Re-normalize all columns in the X array
            if want_normalize is True:
                scalar_transformation = StandardScaler()
                X = scalar_transformation.fit_transform(X_df)
            else:
                scalar_transformation = None
                X = X_df.values
            
            # Getting y properties
            csv_data = load_property_data(csv_data_path = DEFAULT_CSV_PATHS["raw_data"],
                                          keep_list = DEFAULT_INDEX_COLS + [property_label])
            
            # Getting location at which csv data exists. True if value exists
            csv_data_exists = ~csv_data[property_label].isnull() # list of True/False
            
            # Defining y array
            y_array = csv_data[property_label][csv_data_exists].values
            labels_array = csv_data[DEFAULT_INDEX_COLS[0]][csv_data_exists].values
            
            # Getting X for area of existing
            X_df = X_df[csv_data_exists]
            X = X[csv_data_exists]
            
            # Getting the different folds
            fold_list = split_dataset_KFolds(X = X, 
                                             y = y_array, 
                                             labels = labels_array, 
                                             n_folds = 5)
            
            # Generating models
            # model = RandomForestRegressor(**RF_DEFAULTS)
            
            # Storing model
            model_storage = {}
            
            # Looping through each model
            for model_type in model_type_list:
                    
                # Generating prediction dataframe
                predict_df, model_list = predict_for_KFolds(model_type = model_type,
                                                            fold_list = fold_list,
                                                            return_models = True)
                
                # Getting the statistics
                stats_dict = get_stats(predict_df)
                
                # Storing for each model
                model_storage[model_type] = {
                    'predict_df': predict_df,
                    'model_list': model_list,
                    'stats_dict': stats_dict,
                    }
            
            ######################################################################
            # Storing all variables
            storage_descriptor_sets[property_label][descriptor_df_key] = {
                'X_df': X_df,
                'X': X,
                'y': y_array,
                'fold_list': fold_list,
                'model_storage': model_storage,
                }
            
    return storage_descriptor_sets