#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_tools.py

This code contains all default plotting information. 

Created on: Tue Jun 15 12:52:38 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.

Global variables:
    FIGURE_SIZES_DICT_CM:
        Defalt figure sizes

Functions:
    create_horizontal_bar:
        creates horizontal bar plots
    set_mpl_defaults:
        sets matplotlib defaults
    cm2inch:
        converts centimeters to inches
    store_figure:
        function that sotres the figure
    create_fig_based_on_cm:
        creates figures based on centimer sizes
"""
# Importing tools
import numpy as np
import matplotlib.pyplot as plt

# Importing relative stats
from .core.stats import get_stats

## DEFINING DEFAULT FIGURE SIZES
FIGURE_SIZES_DICT_CM = {
        '1_col': (8.55, 8.55),
        '1_col_landscape': (8.55, 6.4125), # 4:3 ratio
        '1_col_portrait' : (6.4125, 8.55),
        '2_col': (17.1, 17.1), 
        '2_col_landscape': (17.1, 12), 
        }

### FUNCTION TO SET MPL DEFAULTS
def set_mpl_defaults():
    ''' 
    This function sets all default parameters 
    # https://matplotlib.org/tutorials/introductory/customizing.html
    '''
    import matplotlib as mpl
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    ## EDITING TICKS
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    
    ## FONT SIZE
    mpl.rcParams['legend.fontsize'] = 8
    
    ## CHANGING FONT
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"
    ## DEFINING FONT
    font = {'size'   : 10}
    mpl.rc('font', **font)
    return

# Function that converts centimeters to inches
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# Function to store the figure
def store_figure(fig, 
                 path, 
                 fig_extension = 'png', 
                 save_fig=False, 
                 dpi=300, 
                 bbox_inches = 'tight'):
    '''
    The purpose of this function is to store a figure.
    INPUTS:
        fig: [object]
            figure object
        path: [str]
            path to location you want to save the figure (without extension)
        fig_extension:
    OUTPUTS:
        void
    '''
    ## STORING FIGURE
    if save_fig is True:
        ## DEFINING FIGURE NAME
        fig_name =  path + '.' + fig_extension
        print("Printing figure: %s"%(fig_name) )
        fig.savefig( fig_name, 
                     format=fig_extension, 
                     dpi = dpi,    
                     bbox_inches = bbox_inches,
                     )
    return

# Function to create figure based on cm
def create_fig_based_on_cm(fig_size_cm = (16.8, 16.8)):
    ''' 
    The purpose of this function is to generate a figure based on centimeters 
    INPUTS:
        fig_size_cm: [tuple]
            figure size in centimeters 
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''
    ## FINDING FIGURE SIZE
    if fig_size_cm is not None:
        figsize=cm2inch( *fig_size_cm )
    ## CREATING FIGURE
    fig = plt.figure(figsize = figsize) 
    ax = fig.add_subplot(111)
    return fig, ax


# Function to create bar plot
def create_horizontal_bar(labels,
                          values,
                          width = 0.5,
                          color ='k',
                          xlabel = "Value",
                          fig_size_cm = FIGURE_SIZES_DICT_CM['1_col']):
    """
    This function generates a horizontal bar plot

    Parameters
    ----------
    labels : np.array
        Labels that you want to include
    values: np.array
        values that you want to plot as a horizontal bar plot
    width: float, optional
        width of the bar plot. Default is 0.5
    xlabel: str, optional
        label for the x axis. Default is "Value".
    fig_size_cm : tuple size 2, optional
        Figure size in centimeters. The default is plot_tools.FIGURE_SIZES_DICT_CM['1_col'].
        
    Returns
    -------
    fig, ax: obj
        figure and axis object

    """
    
    # Generating plot
    fig, ax = create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Adding bar plot
    ax.barh(labels, 
            values, 
            width, 
            color = color,
            capsize=2, )
    
    # Reverse axis
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # Setting label
    ax.set_xlabel(xlabel)
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax

# Function that plots the parity plot
def plot_parity(predict_df,
                fig_size_cm = FIGURE_SIZES_DICT_CM['1_col'],
                xlabel = "Actual",
                ylabel = "Predicted",
                stats_dict = None,
                fig = None,
                ax = None,
                want_extensive = False,
                ):
    """
    This function plots the parity given a prediction dataframe with 'y_pred' and 'y_act'

    Parameters
    ----------
    predict_df : dataframe
        Prediction dataframe containing 'y_act' and 'y_pred'
    fig_size_cm: tuple
        figure size in cm
    stats_dict: dict, optional
        statistics for the dataframe. The default value is None. If None, 
        this function will generate statistics.
    fig: obj, optional
        figure object. Default is None.
    ax: obj, optional
        axis object. Default is None.
    want_extensive: logical, optional
        True if you want extensive statistics box. Default is False
    Returns
    -------
    None.

    """
    # Generating plot
    if fig is None or ax is None:
        fig, ax = create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Defining x and y 
    x = predict_df['y_act']
    y = predict_df['y_pred']
    
    # Generating plot
    ax.scatter(x, y, color = 'k')
    
    # Adding labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Drawing y = x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    ax.plot(lims, lims, 'r-', alpha=0.5, zorder=0)
    ax.set_aspect('equal')
    
    # Generating statistics and including it in the bottom right corner
    if stats_dict is None:
        # Generating statistics
        stats_dict = get_stats(predict_df = predict_df)
    
    # Including into the plot
    if want_extensive is False:
        box_text = "R$^2$ = %.2f"%(stats_dict['r2'])
    else:
        box_text = "R$^2$ = %.2f\nRMSE = %.2f"%(stats_dict['r2'],
                                                stats_dict['rmse'],
                                                )
        
    # Adding text to axis
    ax.text(0.95, 0.05, box_text,
         horizontalalignment='right',
         verticalalignment='bottom',
         transform = ax.transAxes,
         bbox=dict(facecolor='none', edgecolor= 'none', pad=5.0))
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax
