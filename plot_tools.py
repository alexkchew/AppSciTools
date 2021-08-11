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
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Importing relative stats
from .core.stats import get_stats

# Defining statistics global var
STATS_NAME_CONVERSION_DICT = {
        'pearsonr': "Pearson's $r$",
        'r2' : "R$^2$",
        'rmse': 'RMSE'
        }

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
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['axes.linewidth'] = 1
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    ## EDITING TICKS
    matplotlib.rcParams['xtick.major.width'] = 1.0
    matplotlib.rcParams['ytick.major.width'] = 1.0

    ## FONT SIZE
    matplotlib.rcParams['legend.fontsize'] = 8

    ## CHANGING FONT
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    ## DEFINING FONT
    font = {'size': 10}
    matplotlib.rc('font', **font)
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
                          want_labels = False,
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
    want_labels: logical, optional
        True if you want lavbels
        
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
    
    # Adding labels
    if want_labels:
        _, xmax = plt.xlim()
        plt.xlim(0, xmax+300)
        for i, v in enumerate(values):
            ax.text(v + 10, i, str(v), color='black',  fontsize=10, ha='left', va='center')
    
    # Reverse axis
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # Setting label
    ax.set_xlabel(xlabel)
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax

# Function o create a box text based on states desired
def generate_box_text(stats_dict, stats_desired):
    """
    This function generates a box string based on the desired stats list.
    
    Parameters
    ----------
    stats_dict: dict
        dictionary of statistics
    stats_desired: list
        list of statistics
            
    Returns
    -------
    box_text: str
        box text
    """
    # Getting default box text
    box_text = ''
    # Looping through each stats desired
    for stat_idx, each_stat in enumerate(stats_desired):
        # Checking if the stat is within stat dict
        if each_stat in stats_dict:
            # Getting box text string
            if each_stat in STATS_NAME_CONVERSION_DICT:
                initial_string = STATS_NAME_CONVERSION_DICT[each_stat]
            else:
                initial_string = each_stat
            # Getting string
            input_str = "%s = %.2f"%(initial_string, stats_dict[each_stat])
            # Adding \n if we still need to add a new line
            if stat_idx != len(stats_desired) - 1:
                input_str += "\n"
            # Adding to box text
            box_text += input_str
    
    return box_text

# Function that plots the parity plot
def plot_parity(predict_df,
                fig_size_cm = FIGURE_SIZES_DICT_CM['1_col'],
                xlabel = "Actual",
                ylabel = "Predicted",
                stats_dict = None,
                fig = None,
                ax = None,
                want_extensive = False,
                stats_desired = None,
                show_outliers = False,
                df_act_key = 'y_act',
                df_pred_key = 'y_pred',
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
    stats_desired: list, optional
        list of stats desired. The default value is None.
    df_act_key: str, optional
        actual key for dataframe. The default value is 'y_act'.
    df_pred_key: str, optional
        prediction key for dataframe. The default value is 'y_pred'.
    show_outliers: logical, optional
        True if you want to show all outliers. This assumes that you already 
        run the `detect_outliers_from_df` function and that there is an 
        'outlier' key. 
    Returns
    -------
    fig, ax: obj
        figure and axis object

    """
    # Generating plot
    if fig is None or ax is None:
        fig, ax = create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # First check if show outlier is available
    if show_outliers is True:
        if 'outlier' not in predict_df:
            print("Warning, outlier is desired, but no 'outlier' key is available.")
            print("Make sure to run `detect_outliers_from_df` to compute all outliers.")
            print("Turning off show outliers to prevent errors.")
            show_outliers = False
    
    # Seeing if you want outliers
    if show_outliers is False:
        # Defining x and y 
        x = predict_df[df_act_key]
        y = predict_df[df_pred_key]
        
        # Generating plot
        ax.scatter(x, y, color = 'k')
        
    else:
        # Spltting between values with and without outliers
        for want_outliers, scatter_color in zip( [False, True],
                                                 ['k', 'r',]):
            # Plotting for each
            logical_for_outlier = predict_df['outlier'] == want_outliers
            x = predict_df[df_act_key][logical_for_outlier]
            y = predict_df[df_pred_key][logical_for_outlier]

            # Generating plot
            ax.scatter(x, y, color = scatter_color)
            
            # Adding labels if outliers are desired
            if want_outliers is True:
                # Identifying the labels
                labels = predict_df['label'][logical_for_outlier]
                
                texts_list = []
                
                # zip joins x and y coordinates in pairs
                for x_values,y_values,label_values in zip(x,y,labels):
                
                    label = "{:s}".format(label_values)
                    
                    # Annotating labels
                    text = plt.annotate(label, # this is the text
                                        (x_values,y_values), # these are the coordinates to position the label
                                        textcoords="offset points", # how to position the text
                                        xytext=(0,10), # distance from text to points (x,y)
                                        ha='center',
                                        color = scatter_color) # horizontal alignment can be left, right or center
                    # Storing
                    texts_list.append(text)
#                # Adjusting text
#                try:
#                    from adjustText import adjust_text
#                    adjust_text(texts_list,
#                                arrowprops=dict(arrowstyle='->', color=scatter_color)
#                                )
#                except ImportError:
#                    print("Skipping text adjustment labels")
#                    print("If desired, install adjustText:")
#                    print("--> pip install adjustText")
#                    pass
            
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
        stats_dict = get_stats(predict_df = predict_df,
                               df_act_key = df_act_key,
                               df_pred_key = df_pred_key)
    
    # Including into the plot
    if want_extensive is False:
        box_text = "R$^2$ = %.2f"%(stats_dict['r2'])
    else:
        box_text = "R$^2$ = %.2f\nRMSE = %.2f"%(stats_dict['r2'],
                                                stats_dict['rmse'],
                                                )
    # Getting stats desired
    if stats_desired is not None:
        # Changing box text
        box_text = generate_box_text(stats_dict, stats_desired)
    
    # Adding text to axis
    ax.text(0.95, 0.05, box_text,
         horizontalalignment='right',
         verticalalignment='bottom',
         transform = ax.transAxes,
         bbox=dict(facecolor='none', edgecolor= 'none', pad=5.0))
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax

# Function to plot histograms from csv
def plot_histogram(values,
                   n_bins = 20,
                   fig = None,
                   ax = None,
                   ylabel = 'N',
                   xlabel = 'Property',
                   want_total_box = True,
                   want_normal_dist = False,
                   normalize = False,
                   fig_size_cm = FIGURE_SIZES_DICT_CM['1_col'],
                   ):
    """
    This function plots the histogram as a bar plot. 
    
    Parameters
    ----------
    values: np.array
        values that you want ot plot a histogram for.
    n_bins: int
        number of bins ot plot
    fig: obj, optional
        figure object. The default value is None.
    ax: obj, optional
        axis object.  The default value is None.
    ylabel: str
        label of y
    want_total_box: logical, optional
        True if you want the total box in the upper right.
    want_normal_dist: logical, optional
        True if you want a normal distribution added onto the plot
    fig_size_cm: tuple
        figure size in cm
    normalize: logical, optional
        True if you want to normalize the density
        
    Returns
    -------
    
    fig: obj
        figure object
    ax: obj
        axis object
    
    """
    # Generating plot
    if fig is None or ax is None:
        fig, ax = create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Generating histogram
    num_hist, bins, patches = ax.hist(values, 
                                      bins = n_bins, 
                                      color = 'gray', 
                                      edgecolor = 'k',
                                      density = normalize)
    
    # Getting total points
    n_points = num_hist.sum()
    
    # Adding xlabel
    ax.set_xlabel(xlabel)
    # Adding y label
    ax.set_ylabel(ylabel)
    
    # Getting y limits
    y_lims = ax.get_ylim()
    
    # Adding labels
    for rect in ax.patches:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
            
    # Setting y limits to match y range
    y_range = y_lims[1] - y_lims[0]
    ax.set_ylim([y_lims[0], y_lims[1] + y_range * .3])
    
    # Seeing if you want normal distribution
    if want_normal_dist is True:
        from scipy.stats import norm
        
        # Getting mean and std
        (mu, sigma) = norm.fit(values)
        
        # Getting area
        widths = bins[1:] - bins[:-1]
        area = (widths * num_hist).sum()
        
        # Add a 'best fit' line
        best_fit_y = norm.pdf(bins, mu, sigma)
        ax.plot(bins,
                best_fit_y * area,                 
                color = 'k',
                linestyle = '--',)
        
    # Adding text to axis
    if want_total_box is True:
        box_text = "Total = %d"%(n_points)
        if want_normal_dist is True:
            box_text = '\n'.join([box_text,
                                  '$\mu$ = %.2f'%(mu),
                                  '$\sigma$ = %.2f'%(sigma),
                                  ])
        ax.text(0.95, 0.95, box_text,
             horizontalalignment='right',
             verticalalignment='top',
             transform = ax.transAxes,
             bbox=dict(facecolor='none', edgecolor= 'none', pad=5.0))
    
    return fig, ax