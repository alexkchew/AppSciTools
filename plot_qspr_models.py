#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_qspr_models.py

This script contains all plotting information for QSPR models.


Created on: Wed Jul 21 09:58:36 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.
"""
import matplotlib.pyplot as plt
import pandas as pd

# Importing plot tools
from . import plot_tools

# Importing specific plot tolols
from .plot_tools import STATS_NAME_CONVERSION_DICT

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
                          fig_size_cm = plot_tools.FIGURE_SIZES_DICT_CM['1_col'],
                          ascending = None
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
    ascending: logical, optional
        True if you want to sort models by ascending or decending (True/False).
        The default value is None.
        
    Returns
    -------
    fig, ax: obj
        fig and axis object
    """        
    # Creating plots
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Getting the error dictionary
    error_for_property = pd.Series(error_storage[property_name])
    
    if ascending is not None:
        error_for_property = error_for_property.sort_values(ascending = ascending)
    
    # Getting the model labels
    x_labels = list(error_for_property.keys())
    
    # Converting x labels if within model
    x_labels = [ MODEL_NAME_CONVERSION_DICT[each_model] if each_model in MODEL_NAME_CONVERSION_DICT else each_model for each_model in x_labels ]
    
    # Getting y values
    y = error_for_property.values
    # [error_for_property[each_model] for each_model in error_for_property.keys()]
    
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
                                       ascending = None,
                                       ):
    """
    This function plots model comparison for a property. 
    
    Parameters
    ----------
    storage_descriptor_sets: dict
        contains all the raw information from model training.
    model_type_list: list
        list of models that you have trained the models on.
    property_name: str
        property name that you are interested in
    descriptor_key: str
        descriptor key to use
    fig_size_cm: tuple, optional
        figure size. The default values are stored within FIGURE_SIZES_DICT_CM.
    width: float, optional
        width of the distributions        
    ascending: logical, optional
        True if you want to sort models by ascending or decending (True/False).
        The default value is None.
        
    Returns
    -------
    fig, ax: obj
        fig and axis object
    error_storage: dict
        dictionary containig error storage
    
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
                                    ascending = ascending,
                                    )
    
    return fig, ax, error_storage

# Function to plot specific models in sub plots
def plot_multiple_parity_for_property_as_subplots(plot_specific_models,
                                                  storage_descriptor_sets,
                                                  property_name = 'Experimental E(S1T1)',
                                                  descriptor_key = '2d_and_qm_descriptors',
                                                  figsize = plot_tools.cm2inch(*(24, 12)),
                                                  MAX_STRING_LENGTH = 20,
                                                  stats_desired = ['r2', 'rmse']):
    """
    This function plots multiple parities as a subplot. 
    
    Parameters
    ----------
    plot_specific_models: list
        list of specific models that you would like to plot
    property_name: str
        property name that you are interested in
    descriptor_key: str
        descriptor key to use
    figsize: tuple, optional
        figure size in inches
    MAX_STRING_LENGTH: int
        maximum string length to use
    Returns
    -------
    fig, ax: obj
        fig and axis object
        
        
    Examples
    -------
    
    
    fig, ax = plot_qspr_models.plot_multiple_parity_for_property_as_subplots(storage_descriptor_sets = storage_descriptor_sets,
                                                                             plot_specific_models = plot_specific_models,
                                                                             property_name = 'Experimental E(S1T1)',
                                                                             descriptor_key = '2d_and_qm_descriptors',
                                                                             figsize = plot_tools.cm2inch(*(20, 12)),
                                                                             MAX_STRING_LENGTH = None,
                                                                             stats_desired = ['pearsonr', 'rmse'])    
    
    
    """
    # Defining figure and axis
    fig, axs = plt.subplots(nrows = 1,
                            ncols = len(plot_specific_models),
                            sharey = False,
                            sharex = False,
                            figsize = figsize)
    
    # Looping through each model
    for idx_model, each_model in enumerate(plot_specific_models):
        # Getting axis
        ax = axs[idx_model]
        
        # Getting results
        prediction_dict = storage_descriptor_sets[property_name][descriptor_key]['model_storage'][each_model]
        predict_df = prediction_dict['predict_df']
        stats_dict = prediction_dict['stats_dict']
        
        # Plotting figure and axis
        fig, ax = plot_tools.plot_parity(predict_df = predict_df,
                                  stats_dict = stats_dict,
                                  xlabel = "Actual",
                                  ylabel = "Predicted",
                                  fig = fig,
                                  ax = ax,
                                  want_extensive = True,
                                  stats_desired = stats_desired
                                  )
        # Getting title
        if each_model in MODEL_NAME_CONVERSION_DICT:
            output_title = MODEL_NAME_CONVERSION_DICT[each_model]                
        else:
            output_title = each_model
        
        # Checking if it is not none
        if MAX_STRING_LENGTH is not None:
            output_title = output_title[:MAX_STRING_LENGTH]
        
        # Adding title
        ax.set_title("%s"%(output_title),
                     fontsize = 10)
    
    # Tight layout figure
    fig.tight_layout()
    
    return fig, ax
