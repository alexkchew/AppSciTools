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
from scipy import stats

# 5-CV cross validation functions
from sklearn.model_selection import KFold

# Importing load functions
from .load_csv import load_multiple_descriptor_data, load_property_data

# Standardization procedures (maybe not necessary)
from sklearn.preprocessing import StandardScaler

# Generate models
from sklearn.linear_model import Lasso

# Importing modules
from sklearn.model_selection import RepeatedKFold, GridSearchCV

# Importing relative stats
from .core.stats import get_stats

# Getting stacking regressor
from sklearn.ensemble import StackingRegressor

# Setting random seed
np.random.seed(0)

# Defining defaults
RF_DEFAULTS = {
    'n_estimators': 200,
    }

# Defining default index
DEFAULT_INDEX_COLS = ['Title']

# Getting default grid search
DEFAULT_GRID_SEARCH = {
        'lasso':{
                'alpha': np.arange(0.01, 2, 0.01),
                },
        'RF':{
                'n_estimators': np.arange(100, 500, 100),
                },
        'lightgbm':{
                'n_estimators': np.arange(100, 500, 100),
                },
        'pls': {
                'n_components': np.arange(2,20),
                },
        'knn': {
                'n_neighbors': np.arange(5, 15),
                }
        }
        
# Getting all available models
AVAILABLE_MODELS = [
        'linear',
        'lasso',
        'RF',
        'lightgbm', 
        'pls', 
        'knn', 
        'gpr', 
        'svr',
        'gradient_boost_regression',
        'baysian_ridge',
        'elastic_net',
        'kernel_ridge',
        'xgboost',
        'stacked_LGBM_RF_GBR',
        'stacked_LGBM_RF',
        ]

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
            'pls':
                partial least squares regression
            'knn':
                K nearest regressor
            'gpr':
                Gaussian process regression
                Example link: https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319
            'gradient_boost_regression':
                gradient boost regression
            'baysian_ridge':
                Baysian ridge regression
            'elastic_net':
                Linear elastic net model
            'kernel_ridge':
                kernel ridge regression
            'xgboost':
                XGBoost regressor
            
        Many examples are available:
            https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods#regression
            
    Returns
    -------
    model: [obj]
        model object

    """
    # Iteratively find model types
    if model_type == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**RF_DEFAULTS)
    elif model_type == 'lightgbm':
        from lightgbm import LGBMRegressor # Module isn't working correctly at this time
        model = LGBMRegressor() # **RF_DEFAULTS
    elif model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_type == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(max_iter=10000000)
    elif model_type == 'svr':
        from sklearn.svm import SVR
        model = SVR()
        # model = make_pipeline(StandardScaler(), SVgradient_boost_regressionR(C=1.0, epsilon=0.2))
    elif model_type == 'pls':
        # Partial least squares
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression()
    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor()
    elif model_type == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        model = GaussianProcessRegressor()
    elif model_type == 'gradient_boost_regression':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor()
    elif model_type == 'baysian_ridge':
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge()
    elif model_type == 'elastic_net':
        from sklearn.linear_model import ElasticNet
        model = ElasticNet()
    elif model_type == 'kernel_ridge':
        from sklearn.kernel_ridge import KernelRidge
        model = KernelRidge()
    elif model_type == 'xgboost':
        from xgboost.sklearn import XGBRegressor
        model = XGBRegressor()
    
    # Getting model types
    elif model_type == 'stacked_LGBM_RF_GBR':
        # Generating stacked regressor
        model = generate_stacked_regressor(model_strings = ['gradient_boost_regression', 
                                                            'lightgbm', 
                                                            'RF' ],
                                           final_estimator_string = 'linear')
        
    # Getting model types
    elif model_type == 'stacked_LGBM_RF':
        # Generating stacked regressor
        model = generate_stacked_regressor(model_strings = [
                                                            'lightgbm', 
                                                            'RF' ],
                                           final_estimator_string = 'linear')

    
    else:
        print("Error! Model type (%s) not found!"%(model_type))
    return model

# Function to generate stacked mode
def generate_stacked_regressor(model_strings,
                               final_estimator_string = 'linear'):
    """
    This function generates stacked regressor given model strings.
    
    Parameters
    ----------
    model_strings: list
        List of model strings for the regressor.
    final_estimator_string: str, optional
        Final estimate to place at the end. Default value is 'linear'. 
    
    Returns
    -------
    model: obj
        model object
    
    """
    # Getting tuple
    estimators = [ (each_model, generate_model_based_on_type(each_model) ) for each_model in model_strings]
    # Getting final estimator
    final_estimators = generate_model_based_on_type('linear')
    
    # Outputting the model
    model = StackingRegressor(estimators=estimators, 
                              final_estimator=final_estimators)
    
    
    return model



# Function to convert dataframe values
def convert_data_for_better_skew(data,
                                 transform_type = None):
    """
    This function converts the data for a better skew
    
    Parameters
    ----------
    
    data: dataframe
        dataframe containing information of a property you want to transform. 
    transform_type: str, optional
        transformation type. The default value is None
    Returns
    -------
    transformed data according to your desired type. 
    
    Resource:
        https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45
    
    """
    # Defining available types
    available_transforms_ = [
            None,
            'log',
            'sqrt',
            'box-cox',
            ]
    
    if transform_type is None:
        return data
    elif transform_type == 'log':
        return np.log(data)
    elif transform_type == 'sqrt':
        return np.sqrt(data)
    elif transform_type == 'box-cox':
        return stats.boxcox(data.values[:,0])
    else:
        print("Error! Transform type (%s) is not defined."%(transform_type))
        print("Available types are: %s"%(', '.join(available_transforms_)))

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

# Function to generate predictions for each model
def predict_for_KFolds(model_type,
                       fold_list,
                       model_input = None,
                       return_models = False,
                       return_train_test_dfs = False,):
    """
    This function predicts using K-folds.

    Parameters
    ----------
    model: obj
        model to use for the K-fold predictions
    fold_list: list
        list of training and testing sets
    model: obj, optional
    
    return_models: logical, optional
        True if you want to return the models for each fold. Default is False.
    return_train_test_dfs: logical, optional
        True if you want to return training and test set dataframes for each 
        fold. Default is False. 
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
    
    # Storing predictions for each fold
    if return_train_test_dfs is True:
        train_test_dfs = []
    
    # Looping through each
    for train_test_dict in fold_list:
        # Creating a new model
        if model_input is None:
            model = generate_model_based_on_type(model_type = model_type)
        else:
            model = copy.deepcopy(model_input)
        
        # Fitting model
        model.fit(train_test_dict['X_train'],
                  train_test_dict['y_train'])
        
        
        # Predicting the test set
        yhat = model.predict(train_test_dict['X_test'])
        
        # Checking if the prediction is a shape (N, 1)
        if len(yhat.shape) == 2:
            # Fixing y hat for when predictions have multiple columns
            yhat = yhat[:,0]
        
        # Creating dataframe
        df_test = pd.DataFrame(np.array([yhat, train_test_dict['y_test'], train_test_dict['labels_test']]).T,
                          columns = ['y_pred', 'y_act', 'label']
                          )
        
        # Storing dataframes
        predict_df.append(df_test)
        
        # Storing model
        if return_models is True:
            model_list.append(copy.deepcopy(model))
            
        # Predicting the training set
        if return_train_test_dfs is True:
            # Predicting the training set
            y_pred_train = model.predict(train_test_dict['X_train'])
            # Dealing with broadcasting issues
            y_pred_train = np.squeeze(y_pred_train)
            
            # Creating dataframe
            df_train = pd.DataFrame(np.array([y_pred_train, train_test_dict['y_train'], train_test_dict['labels_train']]).T,
                                    columns = ['y_pred', 'y_act', 'label']
                                    )
            # Adding to the list
            train_test_dfs.append({
                    'train_df': df_train,
                    'test_df': df_test,
                    })
            
        
    # Concat for all test set dataframes
    predict_df = pd.concat(predict_df, axis = 0)
        
    # Getting output tuple
    output_list = [predict_df]
    
    if return_models is True:
        output_list.append(model_list)
    
    if return_train_test_dfs is True:
        output_list.append(train_test_dfs)
    
    return tuple(output_list)
    
#    if return_models is True:
#        return predict_df, model_list
#    else:
#        return predict_df

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
def generate_X_df_from_descriptor_list(descriptor_list,
                                       default_csv_paths,
                                       col2remove = DEFAULT_INDEX_COLS,
                                       **args,
                                       ):
    """
    This function generates the combined dataframe from descriptor list. 
    This assumes that all descriptors have been generated with the same 
    indexes. We may need to check this in the future!

    Parameters
    ----------
    descriptor_list : list
        List of descriptors that are desired.
    default_csv_paths: dict
        default csv path dictionary
    col2remove: list
        list of columns to remove from index file
        
    **args goes into loading descriptor data
    
    Returns
    -------
    X_df: [df]
        dataframe containing all descriptors
    """
    
    # Getting descriptors
    descriptor_df_dict = load_multiple_descriptor_data(default_csv_paths = default_csv_paths,
                                                       descriptor_list = descriptor_list,
                                                       **args)

    # Cleaning each dictionary
    descriptor_df_clean = {each_key: strip_df_index(descriptor_df_dict[each_key],
                                                    col2remove = col2remove,
                                                    ) 
                                              for each_key in descriptor_df_dict}
    
    # Combining dataframes
    X_df = pd.concat([descriptor_df_clean[each_key] for each_key in descriptor_df_clean], axis=1)
    
    return X_df
    
# Function to optimize based on inputs
def optimize_hyperparams_given_cv(X,
                                  y,
                                  estimator = Lasso(),
                                  param_grid = {
                                          'alpha': np.arange(0.01, 2, 0.01)
                                          },
                                  scoring = 'neg_mean_absolute_error',
                cv_inputs = dict(
                    n_splits = 5,
                    n_repeats = 1,
                    random_state = 1,
                    ),
                
                        ):
    """
    This function optimizes hyperparameters for the models. 
    
    Parameters
    ----------
    X: [np.array]
        X array
    y: [np.array]
        y array

    estimator: [obj]
        estimator object
    param_grid: dict
        dictionary containing grid to vary
    cv_inputs: dict, optional
        dictionary containing cross validation inputs
        
    
        
    Returns
    -------
    grid_cv: obj
        grid cross validation object
        Outputs:
            grid_cv.best_estimator_: Best model object
            grid_cv.best_params_: Best parameters
        
    """
    # Identifying cross validatoin
    cv = RepeatedKFold(**cv_inputs)

    # Running cross validation CV
    grid_cv = GridSearchCV(estimator = estimator,
                           param_grid =param_grid,
                           cv = cv,
                           scoring = scoring,
                           )
    
    # Fitting
    grid_cv.fit(X,y)
    
    return grid_cv

# Function to remove correlated descriptors above a threshold
def remove_corr_descriptors_from_df(X_df,
                                    threshold = 0.80,
                                    verbose = True,
                                    want_details = False):
    """
    This function removes correlated descriptors given a dataframe.
    
    Parameters
    ----------
    X_df: dataframe
        X descriptors dataframe
    threshold: float, optional
        threshold of Pearson's r to be considered correlated. The default 
        value is 0.80.
    verbose: logical, optional
        True if you want to output the removal of descriptors. Default is True.
    want_details: logical, optional
        True if you want a dataframe of details. 
    Returns
    -------
    X_df_uncorr: dataframe
        dataframe with columns that are uncorrelated. 
    corr_details: dataframe, optional based on want_details
        dataframe containing correlated descriptors. 
    """
    # Getting Pearson's r correlation matrix 
    corr_matrix = X_df.corr()
    
    # Getting names of all deleted columns
    col_corr = set()
    
    # Storing details of correlation
    corr_details = []
    
    # Looping over rows
    for i in range(len(corr_matrix.columns)):
        # Looping over columns
        for j in range(i):
            # Checking if the correlation matrix is greater than or equal to threshold, 
            # as well as whether the columns are not within the correlation cols
            if (np.abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                # Storing to details
                corr_details.append({
                        'i': corr_matrix.columns[i],
                        'j': corr_matrix.columns[j],
                        'Pearson r': corr_matrix.iloc[i, j],
                        })
    
    # Creating dataframe
    corr_details = pd.DataFrame(corr_details)
    
    # Getting X df uncor
    X_df_uncorr = X_df.drop(columns=col_corr)
    
    if verbose is True:
        print("Removing correlated descriptors >= %.2f"%(threshold))
        print("   %d correlated descriptors"%(len(col_corr)))
        print("   %d original descriptors"%(len(X_df.columns)))
        print("   %d output descriptors"%(len(X_df_uncorr.columns)))
    if want_details is True:
        return X_df_uncorr, corr_details
    else:
        return X_df_uncorr

# Defining main function to perform calculations
def main_generate_qspr_models_CV(descriptor_keys_to_use,
                                 descriptor_dict,
                                 output_property_list,
                                 default_csv_paths,
                                 exp_data_name = 'raw_data',
                                 default_index_cols = DEFAULT_INDEX_COLS,
                                 want_normalize = True,
                                 model_type_list = ['RF'],
                                 property_conversion = [],
                                 hyperparam_tuning = False,
                                 remove_des_corr_float = None):
    """
    This function runs QSPR models and outputs prediction accuracies using 
    K-fold cross validation. Note! We are assuming that the columns in X and 
    y data are in the same order. We may need to double-check this in the future.

    Parameters
    ----------
    descriptor_keys_to_use: list
        List of descriptor keys to use
    descriptor_dict: [dict]
        dictionary of descriptors that each contain a 'descriptors_list' and 'label'. 
    output_property_list: list
        list of output properties
    want_normalize: logical
        True if you want to normalize the X array by subtracting mean and dividing by 
        standard deviation. 
    model_type_list: list
        list of models that you want to perform.
    property_conversion: list, optional
        conversion of property to account for skewness. If this is empty, then 
        no property corrections are made. Otherwise, this is a list that 
        corresponds to the list in output_property_list. Hence, input of 
        ['sqrt'] would correspond to the first property. 
    hyperparam_tuning: logical, optional
        True if you want to tune hyper parameters using 5-CV. The default value 
        is False. 
    remove_des_corr_float: float, optional
        Pearson's r correlaton coefficient at which correlated descriptors are
        removed. 
    Returns
    -------
    storage_descriptor_sets: [dict]
        dictionary of dicts that store all information, e.g. [property][descriptor set]

    """

    # Defining storage for each descriptor
    storage_descriptor_sets = {}
    
    # Looping through property labels
    for property_idx, property_label in enumerate(output_property_list):
        
        # Creating empty dictionary for property labels
        storage_descriptor_sets[property_label] = {}
        
        # Looping through each dataframe key
        for descriptor_df_key in descriptor_keys_to_use:
            # Getting descriptor list
            descriptor_list = descriptor_dict[descriptor_df_key]['descriptor_list']
            
            # Generating combined dataframe
            X_df = generate_X_df_from_descriptor_list(descriptor_list = descriptor_list,
                                                      default_csv_paths = default_csv_paths,
                                                      col2remove = default_index_cols,)
            
            # Storing orig dataframe ( before any transformations, which is useful 
            # for debugging purposes ). 
            X_df_orig = X_df.copy()
            
            # Removing correlated descriptors
            if remove_des_corr_float is not None:
                
                # Getting new dataframe
                X_df, corr_details = remove_corr_descriptors_from_df(X_df = X_df,
                                                       threshold = remove_des_corr_float,
                                                       verbose = True,
                                                       want_details = True)
            else:
                corr_details = None
            
            # Re-normalize all columns in the X array
            if want_normalize is True:
                scalar_transformation = StandardScaler()
                X = scalar_transformation.fit_transform(X_df)
            else:
                scalar_transformation = None
                X = X_df.values
            
            # Getting y properties
            csv_data = load_property_data(csv_data_path = default_csv_paths[exp_data_name],
                                          keep_list = default_index_cols + [property_label])
            
            # Getting location at which csv data exists. True if value exists
            csv_data_exists = ~csv_data[property_label].isnull() # list of True/False
            
            # Defining y array
            y_array = csv_data[property_label][csv_data_exists].values
            
            # Modulinating y array based on the inputs
            if len(property_conversion) != 0:
                # Getting property type
                property_conversion_type = property_conversion[property_idx]
                # Converting the y array
                y_array = convert_data_for_better_skew(data = y_array,
                                                       transform_type = property_conversion_type)
            else:
                property_conversion_type = None
                
            # Getting label arrays
            labels_array = csv_data[default_index_cols[0]][csv_data_exists].values
            
            # Getting X for area of existing
            X_df = X_df[csv_data_exists]
            X = X[csv_data_exists]
            
            # Getting the different folds
            fold_list = split_dataset_KFolds(X = X, 
                                             y = y_array, 
                                             labels = labels_array, 
                                             n_folds = 5)
            
            # Storing model
            model_storage = {}
            
            # Looping through each model
            for model_type in model_type_list:
                    
                # Tuning hyperparameters
                if hyperparam_tuning is True:
                    
                    # Checking if parameter is within
                    if model_type in DEFAULT_GRID_SEARCH:
                        # Printing
                        print("Tuning %s..."%(model_type))
                        
                        # Generating the model
                        model = generate_model_based_on_type(model_type = model_type)
                        
                        # Getting grid cross validation
                        grid_cv = optimize_hyperparams_given_cv(X = X,
                                                                y = y_array,
                                                                estimator = model,
                                                                param_grid = DEFAULT_GRID_SEARCH[model_type],
                                                          scoring= 'neg_mean_absolute_error',
                                                        cv_inputs = dict(
                                                            n_splits = 5,
                                                            n_repeats = 1,
                                                            random_state = 1,
                                                            )
                                                        )
                        
                        # Getting best model
                        best_model = grid_cv.best_estimator_
                        
                    else:
                        print("%s not tuned, since it is not in DEFAULT_GRID_SEARCH"%(model_type))
                        best_model = None
                        grid_cv = None
                    

                
                # Generating prediction dataframe
                predict_df, model_list, train_test_dfs = predict_for_KFolds(model_type = model_type,
                                                                            fold_list = fold_list,
                                                                            model_input = best_model,
                                                                            return_models = True,
                                                                            return_train_test_dfs = True,)
                
                # Getting the statistics
                stats_dict = get_stats(predict_df)
                
                # Storing for each model
                model_storage[model_type] = {
                    'predict_df': predict_df,
                    'model_list': model_list,
                    'stats_dict': stats_dict,
                    'grid_cv': grid_cv,
                    'train_test_dfs': train_test_dfs,
                    }
            
            ######################################################################
            # Storing all variables
            storage_descriptor_sets[property_label][descriptor_df_key] = {
                'X_df': X_df,
                'X': X,
                'y': y_array,
                'labels_array': labels_array,
                'scalar_transformation': scalar_transformation,
                'fold_list': fold_list,
                'model_storage': model_storage,
                'property_conversion_type': property_conversion_type,
                'X_df_orig': X_df_orig,
                'corr_details': corr_details,
                }
            
    return storage_descriptor_sets