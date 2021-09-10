#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_qspr_models.py

This script contains all plotting information for QSPR models.


Created on: Wed Jul 21 09:58:36 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.
"""

import pandas as pd
import operator
import numpy as np

# Importing plot tools
from . import plot_tools

from .plot_tools import plt

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

# Function to generate array of rgb colors
def generate_rgb_colors(n_colors,
                        colormap = 'hot',
                        ):
    """
    This function generates rgb color array based on inputs. 
    
    Parameters
    ----------
    n_colors: int
        number of colors that you want
    colormap: str
        colormap that is desired
    Returns
    -------
    colors_array: np.array, shape = n_colors, 4
        colors array from the matplotlib colors
    
    """
    from matplotlib import cm
    colors_array = getattr(cm, 'hot')(range(n_colors))
    return colors_array

# Function to plot model prediction for a single property
def plot_model_comparison(error_storage,
                          property_name,
                          xlabel = 'stat',
                          width = 0.2,
                          fig_size_cm = plot_tools.FIGURE_SIZES_DICT_CM['1_col'],
                          ascending = None,
                          colors = None,
                          ):
    """
    This function plots the model performance for a particular statistic as  
    as bar plot. This is useful for testing which of the models performed 
    the best. 
    
    Parameters
    ----------
    error_storage: pd.dataframe
        dataframe containing error information. Columns are the different 
        descriptor keys. 
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
    colors: list
        list of colors for each bar plot. If None, then we will generate colors 
        based on your inputs. 
        
    Returns
    -------
    fig, ax: obj
        fig and axis object
    """        
    # Creating plots
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Getting the error dictionary
    error_for_property = [ pd.Series(error_storage[descriptor_keys][property_name]) for descriptor_keys in error_storage.columns ]
    
    # Getting the model labels
    x_labels = list(error_for_property[0].keys())
    
    # Converting x labels if within model
    x_labels = np.array([ MODEL_NAME_CONVERSION_DICT[each_model] 
                         if each_model in MODEL_NAME_CONVERSION_DICT else each_model 
                         for each_model in x_labels ])
    
    # Getting indices
    x_ind = np.arange(len(x_labels))
    
    # Generating colors
    if colors is None:
        if len(error_for_property) == 1:
            colors_array = ['k']
        else:
            # Changing the color array
            colors_array = generate_rgb_colors(n_colors = len(error_for_property))
    else:
        colors_array = colors[:]
    
    # Looping through y
    for idx, (each_error_property, color) in enumerate(zip(error_for_property, colors_array)):
        
        # Sorting based on the first property
        if ascending is not None and idx == 0:
            # Arg sorting
            idx_sorted = np.argsort(each_error_property)
            
            # Seeing you want ascending
            if ascending is False:
                idx_sorted = idx_sorted[::-1]
                
            # Re-defining x label based on inputs
            x_labels = x_labels[idx_sorted]
        
        # Getting y values
        y = np.array(each_error_property.values)[idx_sorted]
        
        # Getting label
        label = error_storage.columns[idx]
        
        # Plotting horizontal plot
        ax.barh(x_ind + width * idx,
                y,
                width,
                color = color,
                label = label,
                edgecolor = 'k',
                linewidth = 0.5,
                )
    
    # Setting y ticks
    ax.set(yticks= x_ind + width*(len(error_for_property)-1)/2.0, 
           yticklabels=x_labels, 
           )
           # ylim=[2*width - 1, len(error_for_property)]) # 
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
    
#    # Adding legend if more than one error is plotted
#    if len(error_for_property) > 1:
#        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    
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
                                       colors = None,
                                       descriptor_dict = None,
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
    descriptor_key: str or list
        descriptor key to use. If list, then this will generate a bar plot 
        comparing between multiple descriptor keys. 
    fig_size_cm: tuple, optional
        figure size. The default values are stored within FIGURE_SIZES_DICT_CM.
    width: float, optional
        width of the distributions        
    ascending: logical, optional
        True if you want to sort models by ascending or decending (True/False).
        The default value is None.
    descriptor_dict: dict
        dictionary containing information about the descriptors. This way, we 
        could relate the key to the label. Note that each entry should have a 
        'label' within the dictionary. 
        
    Returns
    -------
    fig, ax: obj
        fig and axis object
    error_storage: dict
        dictionary containig error storage
    """
    # Storing errors
    error_storage_list = {}
    
    # Seeing if the descriptor key is a list
    if type(descriptor_key) is not list:
        # Making this a list
        descriptor_key = [descriptor_key]
    
    # Looping through descriptor keys
    for each_descriptor_key in descriptor_key:
        # Getting error storage
        error_storage = extract_model_errors(storage_descriptor_sets = storage_descriptor_sets,
                                             model_type_list = model_type_list,
                                             descriptor_key = each_descriptor_key,
                                             stat_var = stat_var,
                                             )
        
        # Seeing if dict exists
        output_descriptor_key = each_descriptor_key
        if descriptor_dict is not None:
            if each_descriptor_key in each_descriptor_key:
                output_descriptor_key = descriptor_dict[each_descriptor_key]['label']
        
        # Storing the errors
        error_storage_list[output_descriptor_key] = error_storage
        
    # Creating dataframe
    error_storage_list = pd.DataFrame(error_storage_list)
    
    # Plotting model comparisons
    fig, ax = plot_model_comparison(error_storage = error_storage_list,
                                    property_name = property_name,
                                    xlabel = stat_var,
                                    width = width,
                                    fig_size_cm = fig_size_cm,
                                    ascending = ascending,
                                    colors = colors,
                                    )
    
    return fig, ax, error_storage_list

# Function to plot specific models in sub plots
def plot_multiple_parity_for_property_as_subplots(plot_specific_models,
                                                  storage_descriptor_sets,
                                                  property_name = 'Experimental E(S1T1)',
                                                  descriptor_key = '2d_and_qm_descriptors',
                                                  figsize = plot_tools.cm2inch(*(24, 12)),
                                                  MAX_STRING_LENGTH = 20,
                                                  stats_desired = ['r2', 'rmse'],
                                                  property_label = None,
                                                  fig = None,
                                                  axs = None,
                                                  ax_idx = None):
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
    property_label: str, optional
        property label to output into the axis. Default value is None, which 
        would not output any property labels into axis titles
    fig, axs: obj
        figure and axis objects
    idx_model: [idx]
        index for the axis object
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
    if fig is None or axs is None:
        # Defining figure and axis
        fig, axs = plt.subplots(nrows = 1,
                                ncols = len(plot_specific_models),
                                sharey = False,
                                sharex = False,
                                figsize = figsize)
    
    # Looping through each model
    for idx_model, each_model in enumerate(plot_specific_models):
        # Getting axis
        try:
            ax = axs[idx_model]
        except TypeError:
            ax = axs
        
        # Changing axis if you inputted that
        if ax_idx is not None:
            ax = axs[ax_idx]
        
        # Getting results
        prediction_dict = storage_descriptor_sets[property_name][descriptor_key]['model_storage'][each_model]
        predict_df = prediction_dict['predict_df']
        stats_dict = prediction_dict['stats_dict']
        
        # Getting x and y label
        xlabel = 'Actual'
        ylabel = 'Predicted'
        if property_label is not None:
            xlabel = " ".join([xlabel, property_label])
            ylabel = " ".join([ylabel, property_label])
        
        # Plotting figure and axis
        fig, ax = plot_tools.plot_parity(predict_df = predict_df,
                                  stats_dict = stats_dict,
                                  xlabel = xlabel,
                                  ylabel = ylabel,
                                  fig = fig,
                                  ax = ax,
                                  want_extensive = False,
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
    
    return fig, axs

# Function to plot the importance features
def plot_importance_features(results_df = None,
                             results_df_sorted = None,
                             top_n = 3,
                             fig = None,
                             ax = None,
                             fig_size_cm = plot_tools.FIGURE_SIZES_DICT_CM['1_col'],
                             width = 0.2,
                             max_string_length = None,
                             return_sorted_df = False,
                             color = 'gray',
                             ):
    """
    This function plots the importance features. 
    
    Parameters
    ----------
    results_df: dataframe
        dataframe containing avg Shapley values, etc.
    results_df_sorted: dataframe, optional
        sorted dataframe. If this is already available, it will overwrite the results_df. 
        Default value is None.
    top_n: int, optional
        top n to plot. The default value is 3. 
    fig: obj, optional
        figure object. The default value is None.
    ax: obj, optional
        axis object. The default value is None.
    fig_size_cm: tuple
        figure size in centimeters. 
    width: int, optional
        bar plot width. The default value is 0.2.
    max_string_length: int, optional
        max string length for the features. The default value is None.
        If None, then we will use the entire string. Otherwise, we will 
        truncate string accordingly.
    return_sorted_df: logical, optional
        True if you want to return the sorted dataframe as well.
    color: str or array, optional
        color of the bars. Default value is 'gray'. 
    Returns
    -------
    fig: obj
        figure object. 
    ax: obj
        axis object. 
    results_df_sorted: df, optional
        dataframe of the sorted results.
    """
    # Function to sort results dataframe
    def sort_results_df(results_df):
        """
        This function simply sorts the results dataframe from largest to smallest 
        in terms of importance.
        """
        return results_df.sort_values(by = "Avg_Shap", ascending = False).reset_index(drop=True)

    
    # Sorting values
    if results_df_sorted is None:
        results_df_sorted = sort_results_df(results_df)
    
    # Getting values to plot
    bar_values = results_df_sorted['Avg_Shap_w_Sign'][:top_n]
    bar_errors = results_df_sorted['Std_Shap'][:top_n]
    feature_list = results_df_sorted['Feature'][:top_n]
    
    if max_string_length is not None:
        feature_list = [ each_feature[:max_string_length] for each_feature in feature_list]
    
    # Generating plot
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Plotting horizontal plots
    ax.barh(feature_list, 
            bar_values, 
            width, 
            xerr = bar_errors,
            align = 'center',
            color = color,
            linestyle = None,
            fill = True,
            capsize=3,
            alpha = 1,
            linewidth = 1,
            edgecolor = 'k',
            error_kw = {'linewidth': 1,
                        }                
            )
    
    # Reverse axis
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # Tight layout
    fig.tight_layout()
    
    # Drawing line at zero
    ax.axvline(x = 0, color = 'k')
    
    # Adding x label
    ax.set_xlabel("Importance")
    if return_sorted_df is True:
        return fig, ax, results_df_sorted
    else:
        return fig, ax
