#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_qspr_models.py

This script contains all plotting information for QSPR models.


Created on: Wed Jul 21 09:58:36 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.
"""

# Importing plot tools
from . import plot_tools


# Defining statistics global var
STATS_NAME_CONVERSION_DICT = {
        'pearsonr': "Pearson's $r$",
        'r2' : "R$^2$",
        'rmse': 'RMSE'
        }

# Defining model dict
MODEL_NAME_CONVERSION_DICT = {
        'linear': 'Linear',
        'lasso' : 'LASSO',
        'RF'    : 'Random forest',
        'lightgbm' : 'LightGBM',
        'pls'   : 'Partial least squares',
        'knn'   : 'k nearest neighbors',
        'gpr'   : 'Gaussian process regression',
        'svr'   : 'Support vector regression',
        'gradient_boost_regression': 'Gradient boost regression',
        'baysian_ridge': 'Baysian ridge regression',
        'elastic_net'  : 'Elastic net regression',
        'kernel_ridge' : 'Kernel ridge regression',
        'xgboost'      : 'XGBoost'
        }

# Function to extract error from models
def extract_model_errors(storage_descriptor_sets,
                         model_type_list,
                         descriptor_key = '2d_and_qm_descriptors',
                         stat_var = 'pearsonr',
                         ):
    """
    This function extracts the model statistics for given properties.
    
    Parameters
    ----------
    storage_descriptor_sets: dict
        contains all the raw information from model training.
    model_type_list: list
        list of models that you have trained the models on.
    descriptor_key: str
        descriptor key to use
    Returns
    -------
    error_storage: dict
        dictionary containing all the errors associated with the model.
    """
    # Storing r2
    error_storage = {}
    # {each_model: [] for each_model in model_type_list}
    
    # Plotting for each descriptor set
    for prop_idx, each_property in enumerate(storage_descriptor_sets):
        # Getting predicted df
        output_dict = storage_descriptor_sets[each_property][descriptor_key]
        
        # Getting empty dict for property
        error_storage[each_property] = {}
        
        # Looping through model list
        for each_model in model_type_list:
            error_value = output_dict['model_storage'][each_model]['stats_dict'][stat_var]
            # Storing
            error_storage[each_property][each_model] = error_value
    
    return error_storage
    
# Function to plot model prediction for a single property
def plot_model_comparison(error_storage,
                          property_name,
                          xlabel = 'stat',
                          width = 0.2,
                          fig_size_cm = plot_tools.FIGURE_SIZES_DICT_CM['1_col']
                          ):
    """
    This function plots the model performance for a particular statistic as  
    as bar plot. This is useful for testing which of the models performed 
    the best. 
    
    Parameters
    ----------
    error_storage: dict
        dictionary containing error information. 
    property_name: str
        property name that you are interested in
    width: float, optional
        width of the distributions
    fig_size_cm: tuple, optional
        figure size. The default values are stored within FIGURE_SIZES_DICT_CM.
    xlabel: str
        xlabel to plot on
        
    Returns
    -------
    fig, ax: obj
        fig and axis object
    """        
    # Creating plots
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Getting the error dictionary
    error_for_property = error_storage[property_name]
    
    # Getting the model labels
    x_labels = list(error_for_property.keys())
    
    # Converting x labels if within model
    x_labels = [ MODEL_NAME_CONVERSION_DICT[each_model] if each_model in MODEL_NAME_CONVERSION_DICT else each_model for each_model in x_labels ]
    
    
    # Getting multiple bar plots
    # ind = np.arange(len(x_labels))
    
    # Getting y values
    y = [error_for_property[each_model] for each_model in error_for_property.keys()]
    
    # Plotting horizontal plot
    ax.barh(x_labels,
            y,
            color = 'k',
            )
    
    # Reverse axis
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Setting x axis
    if xlabel in STATS_NAME_CONVERSION_DICT:
        ax.set_xlabel(STATS_NAME_CONVERSION_DICT[xlabel])
    else:
        ax.set_xlabel(xlabel)
    
    # Setting title
    ax.set_title(property_name, fontsize = 10)
    
    # Plotting x = 0 line
    ax.axvline(x = 0 , color = 'k', linewidth = 1)
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax

# Single liner function to plot model comparison
def plot_model_comparison_for_property(storage_descriptor_sets,
                                       model_type_list,
                                       property_name = 'Experimental E(S1T1)',
                                       stat_var = 'pearsonr',
                                       descriptor_key = '2d_and_qm_descriptors',
                                       width = 0.2,
                                       fig_size_cm = plot_tools.FIGURE_SIZES_DICT_CM['1_col'],
                                       ):
    """
    This function plots model comparison for a property. 
    
    Parameters
    ----------
    storage_descriptor_sets: dict
        contains all the raw information from model training.
    model_type_list: list
        list of models that you have trained the models on.
    descriptor_key: str
        descriptor key to use
    fig_size_cm: tuple, optional
        figure size. The default values are stored within FIGURE_SIZES_DICT_CM.
    width: float, optional
        width of the distributions
        
    Returns
    -------
    fig, ax: obj
        fig and axis object
    """
    # Getting error storage
    error_storage = extract_model_errors(storage_descriptor_sets = storage_descriptor_sets,
                                         model_type_list = model_type_list,
                                         descriptor_key = descriptor_key,
                                         stat_var = stat_var,
                                         )
    
    # Plotting model comparisons
    fig, ax = plot_model_comparison(error_storage = error_storage,
                                    property_name = property_name,
                                    xlabel = stat_var,
                                    width = width,
                                    fig_size_cm = fig_size_cm,
                                    )
    
    return fig, ax
