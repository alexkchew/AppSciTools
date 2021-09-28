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
    fig_extension_with_period = '.' + fig_extension
    ## STORING FIGURE
    if save_fig is True:
        ## DEFINING FIGURE NAME
        ext = os.path.splitext(path)[1]
        if ext != fig_extension_with_period:
            fig_name =  path + fig_extension_with_period
        else:
            fig_name = path
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
        xmin, xmax = plt.xlim()
        # Extending the min and max by 10%
        plt.xlim(0, xmax+ 0.30*(xmax - xmin))
        for i, v in enumerate(values):
            ax.text(v + 0.05*(xmax - xmin), i, str(v), color='black',  fontsize=10, ha='left', va='center')
    
    # Reverse axis
    ax.invert_yaxis()  # labels read top-to-bottom
    
    # Setting label
    ax.set_xlabel(xlabel)
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax

# Function to create a box text based on states desired
def generate_box_text(stats_dict, stats_desired,
                      prefix = ''):
    """
    This function generates a box string based on the desired stats list.
    
    Parameters
    ----------
    stats_dict: dict
        dictionary of statistics
    stats_desired: list
        list of statistics
    prefix: str, optional
        prefix for the box text. Default value is ''
            
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
            input_str = "%s%s = %.2f"%(prefix, initial_string, stats_dict[each_stat])
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
                color = 'k',
                deviations = None,
                test_set_colors = 'r',
                ):
    """
    This function plots the parity given a prediction dataframe with 'y_pred' and 'y_act'

    Parameters
    ----------
    predict_df : dataframe
        Prediction dataframe containing 'y_act' and 'y_pred'. If this dataframe 
        has a column called 'set', then it will divide the training and test 
        sets, then plot them accordingly. 
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
    color: str or list, optional
        color for the scatter plot. The default value is 'k' (or black).
    deviations: float, optional
        deviations from the y=x line. If None, this line won't be drawn. Otherwise, 
        we will draw a line above and below the y=x line. 
    test_set_colors: str, optional
        Test set colors. Default colors is 'k' (or black). 
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
    
    # getting tempty trianing and testing dataframe
    train_df, test_df = None, None
    
    # Getting the set of training and testing
    if 'set' in predict_df.columns:
        # Getting training and test set datafrmaes
        train_df = predict_df[predict_df['set'] == 'train'].copy()
        test_df = predict_df[predict_df['set'] == 'test'].copy()
        
        # Plotting the scatter for training and testing
        ax.scatter(train_df[df_act_key], 
                   train_df[df_pred_key],
                   color = color,
                   label = 'Train set',
                   zorder = 1)
        ax.scatter(test_df[df_act_key], 
                   test_df[df_pred_key],
                   color = test_set_colors,
                   label = 'Test set',
                   zorder = 2)        
        
    else:
        
        # Seeing if you want outliers
        if show_outliers is False:
            # Defining x and y 
            x = predict_df[df_act_key]
            y = predict_df[df_pred_key]
            
            # Generating plot
            ax.scatter(x, y, color = color)
            
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
            
    # Adding labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Drawing y = x line
    lims = np.array([
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ])
    
    ax.plot(lims, lims, 'r-', alpha=0.5, zorder=0)
    ax.set_aspect('equal')
    
    # Plotting the deviations
    if deviations is not None:
        deviations_dict = {
                'linestyle': '--',
                'alpha': 0.5,
                'zorder': 0,
                'color': 'r',
                }
        ax.plot(lims, lims + deviations, **deviations_dict)
        ax.plot(lims, lims - deviations, **deviations_dict)
    
    # Generating statistics and including it in the bottom right corner
    if stats_dict is None:
        # Generating statistics
        stats_dict = get_stats(predict_df = predict_df,
                               df_act_key = df_act_key,
                               df_pred_key = df_pred_key)
        
    if train_df is not None:
        # Generating stats for training set
        train_stats_dict = get_stats(predict_df = train_df,
                                     df_act_key = df_act_key,
                                     df_pred_key = df_pred_key)
        
    if test_df is not None:
        # Generating stats for training set
        test_stats_dict = get_stats(predict_df = test_df,
                                    df_act_key = df_act_key,
                                    df_pred_key = df_pred_key)
        
    
    # Including into the plot
    # Depreciated arguments
    if want_extensive is True:
        print("Warning, extensive argument is depreciated!")
        print("Use 'stats_desired' instead!")
        box_text = "R$^2$ = %.2f\nRMSE = %.2f"%(stats_dict['r2'],
                                                stats_dict['rmse'],
                                                )
    # Getting stats desired
    if stats_desired is not None:
        if train_df is not None or test_df is not None:
            overall_prefix = 'Overall '
        else:
            overall_prefix = ''
        # Changing box text
        box_text = generate_box_text(stats_dict, 
                                     stats_desired,
                                     prefix = overall_prefix)
        
        # Getting box text
        box_text_list = [box_text]
        
        # Getting training and testing information
        if train_df is not None:
            # Changing box text
            train_box_text = generate_box_text(train_stats_dict, 
                                               stats_desired,
                                               prefix = 'Train ')
            box_text_list.append(train_box_text)
        
        if test_df is not None:
            # Adding test box text
            test_box_text = generate_box_text(test_stats_dict, 
                                              stats_desired,
                                              prefix = 'Test ')
            box_text_list.append(test_box_text)
        # Creating box text string
        box_test_string = '\n'.join(box_text_list)
    else:
        box_test_string = None
    
    # Adding text to axis
    if box_test_string is not None:
        ax.text(0.95, 0.05, box_test_string,
             horizontalalignment='right',
             verticalalignment='bottom',
             transform = ax.transAxes,
             bbox=dict(facecolor='none', edgecolor= 'none', pad=5.0))
    
    # Adding legend
    if train_df is not None or test_df is not None:
        leg = ax.legend(loc='upper left')
        leg.get_frame().set_edgecolor('k')
#        leg.get_frame().set_linewidth(0.0)
        
    
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
                   text_loc = (0.95, 0.95),
                   text_dict = dict(
                       horizontalalignment='right',
                       verticalalignment='top',
                       bbox=dict(facecolor='none', edgecolor= 'none', pad=5.0)
                       ),
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
    text_loc: tuple, shape = 2, optional
        text location for the histogram box text. Default value is (0.95, 0.95)
    text_dict: dict, optional
        dictionary for text box
    
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
        ax.text(*text_loc, box_text, transform = ax.transAxes, **text_dict)
    
    return fig, ax

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

### FUNCTION TO GET CMAP
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    This function is useful to generate colors between red and purple without having to specify the specific colors
    USAGE:
        ## GENERATE CMAP
        cmap = get_cmap(  len(self_assembly_coord.gold_facet_groups) )
        ## SPECIFYING THE COLOR WITHIN A FOR LOOP
        for ...
            current_group_color = cmap(idx) # colors[idx]
            run plotting functions
    '''
    ## IMPORTING FUNCTIONS
    import matplotlib.pyplot as plt
    return plt.cm.get_cmap(name, n + 1)

