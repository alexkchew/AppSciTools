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
import matplotlib.pyplot as plt

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
            color ='k',
            capsize=2, )
    
    # Reverse axis
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # Setting label
    ax.set_xlabel(xlabel)
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax
