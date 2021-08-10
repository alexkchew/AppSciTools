#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpret_qspr_models.py

This script contains interpretability of the qspr models. We will use the Shapley 
values as a method of quantifying the feature importance. 

Created on: Tue Jul 13 10:32:42 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.

# Installing Shapley
$SCHRODINGER/internal/bin/python3 -m pip install shapley <-- doesn't get the right shap'
$SCHRODINGER/internal/bin/python3 -m pip install shap

Reference:
    For Random forest models
    https://www.kaggle.com/vikumsw/explaining-random-forest-model-with-shapely-values

    For general idea:
        https://christophm.github.io/interpretable-ml-book/shapley.html

"""
# Importing module
import os
import shap
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import copy
    
# From generate qspr models
from .qspr_models import main_generate_qspr_models_CV

# Importing plot tools
from . import plot_tools

# Setting defaults
plot_tools.set_mpl_defaults()

# Setting random seed
np.random.seed(0)

# Function to remove coefficients and delete X training dataframes for lasso
def remove_lasso_zero_coef(lasso_model,
                           X_train_df
                           ):
    """
    This function removes coefficients from lasso that has zero values. This 
    is intended to speed up LASSO analysis by removing non-important features. 
    
    By removing coefficients and training columns, the performance of the 
    models could dramatically improve! The times are:
        
        0.12 seconds for all nonzero coefficients removed
        3 minutes and 7 seconds for keeping all features.
    
    Hence, removing features result in 6.41% of the total time, significantly 
    speeding up Shapley calculations on lasso models. 
    
    Parameters
    ----------
    lasso_model: obj
        lasso object from scipy
    X_train_df: dataframe
        dataframe containing training data, which was used to train the 
        lasso model. 
    
    Returns
    -------
    lasso_model_nonzero: obj
        lasso model with the coefficients of zeros removed
    X_train_nonzero: dataframe
        X train dataframe with columns of coefficients removed. 
    """
    # Testing the removal of coefficients
    idx_nonzero_coef = np.where(lasso_model.coef_ != 0)[0]
    
    # Getting new X train
    X_train_nonzero = X_train_df.iloc[:,idx_nonzero_coef]
    
    # Changing the coefficients
    lasso_model_nonzero = copy.deepcopy(lasso_model)
    lasso_model_nonzero.coef_ = lasso_model.coef_[idx_nonzero_coef]
    
    return lasso_model_nonzero, X_train_nonzero

# Function to convert shapley to mean abs shapley
def compute_mean_abs_shap(shap_values):
    """
    This function computes the mean absolute values of the Shapley. It 
    tells you the average impact on the output magnitude. 
    
    Parameters
    -------
    shap_values: np.array
        shapley values with the same length as total instances and features. 
    
    Returns
    -------
    mean_abs_shap: np.array
        mean absolute shapley values: mean | SHAP value |
        This tells you the average impact of model on output magnitude.
    """
    # Getting shap values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    return mean_abs_shap

# Function to get explainer and shap values
def compute_shap_values(model,
                        X_train_df,
                        speed_up = True):
    """
    This function computes the shapley values for a model. It will search for the 
    model type and appropriate method will be done. 
    
    Parameters
    -------
    model: [obj]
        random forest model
    X_train_df: [dataframe]
        dataframe of the training data with the columns of descriptors
    speed_up: logical, optional
        True if you want to speed up the calculation by simplifying features. 
        For instance, LASSO models have zero coefficients. We could remove 
        these coefficients by togging this on. Default value is True.
    Returns
    -------
    explainer: obj
        explainer object
    shap_values: np.array
        array of shapley values
    X_training_to_use: dataframe
        X training used
    """
    # Defining model type
    model_type = str(type(model))
    
    # Defining available models
    available_models = ['linear_model', 'RandomForestRegressor']
    
    # Defining default model and X train to use
    model_to_use = model
    X_training_to_use = X_train_df
    
    # For LASSO model
    if 'linear_model' in model_type:
        if speed_up is True:            
            # Simplying model by removing coef
            model_to_use, X_training_to_use = remove_lasso_zero_coef(lasso_model = model,
                                                                     X_train_df = X_train_df,
                                                                     )
        
        explainer = shap.Explainer(model_to_use.predict, X_training_to_use)
        shap_values = explainer(X_training_to_use).values
    elif 'RandomForestRegressor' in model_type:
        
        # Editing lgbm to resolve issues
        # Error is noted in: https://github.com/slundberg/shap/issues/1042
#        if 'LGBMRegressor' in model_type:
#            model_to_use.booster_.params['objective'] = 'regression'
        explainer = shap.TreeExplainer(model_to_use)
        shap_values = explainer.shap_values(X_training_to_use)
    else:
        try:
            explainer = shap.Explainer(model_to_use.predict, X_training_to_use)
            shap_values = explainer(X_training_to_use).values
        except Exception:
            pass
            print("Error! Model type not found: %s"%(model_type))
            print("Available models for shapley values: %s"%(', '.join(available_models)))
        
    return explainer, shap_values, X_training_to_use

# Class function to analyze rf models
class interpret_models:
    """
    This function analyzes random forest models.
    
    Parameters
    -------
    
    model: [obj]
        random forest model
    X_train_df: [dataframe]
        dataframe of the training data with the columns of descriptors
    speed_up: logical, optional
        True if you want to speed up the calculation by simplifying features. 
        For instance, LASSO models have zero coefficients. We could remove 
        these coefficients by togging this on. Default value is True.
    
    Returns
    -------
    self.mean_abs_shap_df: dataframe
        contains the mean absolute shapley values after sorting by ascending values
    self.correlation_shap_to_descriptors: dataframe
        contains pearon's r correlation between Shapley values and descriptors. 
        It also contains the sign.
        If Pearon's r is N/A (which is true if your descriptor space do not vary), 
        then we will output a negative sign for that one. 
    """
    def __init__(self,
                 model,
                 X_train_df,
                 speed_up = True):
        
        # Storing inputs
        self.model = model
        self.X_train_df_orig = X_train_df
        self.speed_up = speed_up
        
        # Getting shapley values
        self.explainer, self.shap_values, self.X_train_df = compute_shap_values(model = self.model,
                                                                                X_train_df = self.X_train_df_orig,
                                                                                speed_up = speed_up)
        
        # Getting mean abs shapley values
        mean_abs_shap = compute_mean_abs_shap(shap_values = self.shap_values)

        
        # Getting dataframe
        self.mean_abs_shap_df = pd.DataFrame( np.array([self.X_train_df.columns, mean_abs_shap ]).T, columns = ['Feature', 'Mean Shap'] )
        
        # Turning off the sorting for now
#        # Sorting dataframe
#        self.mean_abs_shap_df = mean_abs_shap_df.sort_values(by = 'Mean Shap', ascending = False).reset_index(drop = True)
#        
        
        # Getting correlation of shap to descriptors 
        self.correlation_shap_to_descriptors = compute_pearsonsr_btn_shap_and_descriptors(X_train_df = self.X_train_df,
                                                                                     shap_values = self.shap_values
                                                                                     )
        
        return
    
    # Generating summary plot
    def plot_summary(self,
                     plot_type="bar",):
        """
        This function plots the summary plot for the shapley outputs.
        """
        
        # Adding summary plot
        shap.summary_plot(self.shap_values, 
                          self.X_train_df, 
                          plot_type=plot_type,
                          show=False)
        fig = plt.gcf()
        return fig
    
    # Getting shap versus descriptors
    def plot_shap_vs_descriptor(self,
                                descriptor_name = 'vdw surface area/Ang.^2',):
        """
        This function plots the shapley values versus descriptor space. 
        It will tell you what the correlatoin is between these two. 

        Parameters
        ----------
        descriptor_name : str, optional
            name of the descriptor to plot. The default is 'vdw surface area/Ang.^2'.
            For more, use self.X_train.columns

        Returns
        -------
        fig : obj
            figure object.
        ax : obj
            axis object.

        """
        # Plotting the correlations
        fig, ax = plot_shap_vs_descriptor(shap_values = self.shap_values,
                                          descriptor_name = descriptor_name,
                                          X_train_df = self.X_train_df,
                                          corr_df = self.correlation_shap_to_descriptors
                                          )
        return fig, ax

# Function to convert Pearson's R to signs
def add_pearsons_r_sign_to_df(correlation_shap_to_descriptors):
    """
    This function adds the sign to dataframe using the Pearson's r correlation. 
    
    If values are >= 0, then we give it a '+' sign. 
    Otherwise, it gets a negative sign. 

    Parameters
    ----------
    correlation_shap_to_descriptors: dataframe
        dataframe of pearson's r correlation versus feature. 

    Returns
    -------
    correlation_shap_to_descriptors: dataframe
        updated dataframe with the sign column

    """
    
    # Getting the sign
    pears_r = correlation_shap_to_descriptors['Pearsons_r_to_SHAP'].values
    
    # Seeing if any is positive or nan
    correlation_shap_to_descriptors['sign'] = np.where(pears_r > 0, 'positive', 'negative')
    
    return correlation_shap_to_descriptors

# Function to get snap correlatoin to descriptors
def compute_pearsonsr_btn_shap_and_descriptors(X_train_df,
                                               shap_values
                                               ):
    """
    Parameters
    ----------
    X_train_df : dataframe
        traikning data
    shap_values : np.array
        shapley values with the same shape as the training dataframe.

    Returns
    -------
    correlation_shap_to_descriptors: dataframe
        Pearson's correlation between Shapley and feature space. 

    """

    # Defining storage for it
    correlation_shap_to_descriptors = []
    
    # Getting sign using shapley values
    for idx, col_name in enumerate(X_train_df.columns):
        # Getting the Shapley values
        shap_v = shap_values[:, idx]
        
        # Getting the descriptor
        descriptor_values = X_train_df.values[:, idx]
        
        # Getting Pearson's r correlaton
        pear_r = pearsonr(shap_v, descriptor_values)[0]
    
        # Storing
        correlation_shap_to_descriptors.append({'Feature': col_name, 
                                                'Pearsons_r_to_SHAP': pear_r})
    
    # Creating correlation
    correlation_shap_to_descriptors = pd.DataFrame(correlation_shap_to_descriptors)
    
    # Adding sign
    add_pearsons_r_sign_to_df(correlation_shap_to_descriptors)
    
    return correlation_shap_to_descriptors

# Function to plot correlatoin for a descriptor
def plot_shap_vs_descriptor(shap_values,
                            X_train_df,
                            descriptor_name = 'vdw surface area/Ang.^2',
                            corr_df = None,
                            fig_size_cm = plot_tools.FIGURE_SIZES_DICT_CM['1_col']
                            ):
    """
    This function plots the shapley values versus the feature of interest. 
    It will show how the feature impacts the Shapley values. 

    Parameters
    ----------
    shap_values : np.array
        Shapley values with the array size n_instances and n_features.
    X_train_df : np.array
        Raw X_train data with the same shape and size as shap_values
    descriptor_name: str
        name of the descriptor that you want to show
    corr_df : dataframe, optional
        dataframe containing columns of 'Feature' and 'Pearsons_r_to_SHAP'.
        The default value for this is None, which will then generate Pearon's r
        correlation coefficient by itself. 
    fig_size_cm : tuple, optional
        figure size in cm. By default, we take a 1-col example.

    Returns
    -------
    None.

    """
    # Creating figure
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    # Adding labels
    ax.set_xlabel(descriptor_name)
    ax.set_ylabel("Shapley values")
    
    # Getting index
    index_of_feature = np.where(X_train_df.columns == descriptor_name)
    
    # Defining x and y
    x = X_train_df[descriptor_name].values
    y = shap_values[:,index_of_feature]
    
    # Getting pearsons r
    if corr_df is None:
        pear_r = pearsonr(y,x)[0]
    else:
        pear_r = corr_df[corr_df['Feature'] == descriptor_name]['Pearsons_r_to_SHAP']
    
    # Adding box text
    box_text = "Pearson's $r$: %.2f"%(pear_r)
    
    # Plotting
    ax.scatter(x,y,color = 'k', label = box_text)
    
    ax.legend()
    
# =============================================================================
#         # Adding text to axis
#         ax.text(0.95, 0.05, box_text,
#              horizontalalignment='right',
#              verticalalignment='bottom',
#              transform = ax.transAxes,
#              bbox=dict(facecolor='none', edgecolor= 'none', pad=5.0))
# =============================================================================
    
    
    return fig, ax

# Function to generate multiple interpretations
def interpret_multiple_models(model_list,
                              X_train_df_list,
                              speed_up = True,
                              ):
    """
    This function interprets multiple models and outputs them into a list. 
    
    Parameters
    ----------
    model_list: list
        list of model interpretations
    X_train_df_list: list
        list of X training dataframes
    speed_up: logical, optional
        True if you want to speed up the calculation by simplifying features. 
        For instance, LASSO models have zero coefficients. We could remove 
        these coefficients by togging this on. Default value is True.
        
    Returns
    -------
    store_dfs: dataframe
        dataframe storing all information of mean abs shap and sign dataframes.
    """
    # Storing each of them
    store_dfs = []
    
    # Looping through the models
    for idx, model in enumerate(model_list):
        # Getting dataframe
        X_train_df = X_train_df_list[idx]
        # Getting interpret rf
        interpretation = interpret_models(model = model,
                                          X_train_df = X_train_df,
                                          speed_up = speed_up)
        # Storing outputs
        output_dict = {
                'mean_abs_shap_df': interpretation.mean_abs_shap_df,
                'sign_df': interpretation.correlation_shap_to_descriptors}
        # Appending
        store_dfs.append(output_dict)
    return store_dfs



# Function to rapidly generate X_train_df list
def generate_X_train_df_list(descriptor_dict_output):
    """
    This function rapidly generates the training dataframe for a given fold. 
    
    Parameters
    ----------
    descriptor_dict_output: dict
        dictionary of the model that you are looking at.
    Returns
    -------
    X_train_df_list: list
        list of training dataframes
    """
    # Getting X train list
    X_train_df_list = []
    
    # Getting Dataframe
    X_df = descriptor_dict_output['X_df']
    
    # Going through the index
    for idx in range(len(descriptor_dict_output['fold_list'])):
        X_train = descriptor_dict_output['fold_list'][idx]['X_train']
        X_train_df = pd.DataFrame(X_train, columns = X_df.columns)
        
        # Storing
        X_train_df_list.append(X_train_df)
    return X_train_df_list

# Function to combine all dataframes
def combine_shap_dfs(store_dfs):
    """
    This function combines multiple dataframes, such as the ones from the 
    shapley dataframe. It will iteratively loop through each dataframe and 
    store any new information other than the default column of "feature". 
    
    Parameters
    ---------- 
    store_dfs: list
        list of dataframes containing shapley parameters
        
    Returns
    -------
    combined_df_dict: [dict]
        dictionary containing combined information from the dataframes.
    """
    
    # Defining default feature column
    default_col = 'Feature'
    merged_df_args = dict(left_on = default_col, right_on = default_col, how = 'inner')
    
    # Defining combined dict
    combined_df_dict = {}
    
    # Loop for each type
    for each_df_type in store_dfs[0].keys():
        # Getting list of dataframes
        list_of_dfs = [store_dfs[idx][each_df_type] for idx in range(len(store_dfs))]
        
        # Generating single dataframe
        for df_idx, df_info in enumerate(list_of_dfs):
            # Relabelling the dataframe
            suffix_str = '_%d'%(df_idx)
            df_info = df_info.add_suffix(suffix_str)
            df_info = df_info.rename(index=str, columns={'%s%s'%(default_col, suffix_str):default_col})
            
            # If first iteration, we say that is the merged one.
            if df_idx == 0:
                merged_df = df_info.copy()
            else:
                # Begin attaching dataframes on. 
                # Start by adding suffix and renaming feature
                merged_df = merged_df.merge(df_info, **merged_df_args) #  suffixes = ('_1','_2'),
        
        # After merging, store it
        combined_df_dict[each_df_type] = merged_df.copy()
    
    return combined_df_dict

# Function to summarize combined dataframes
def summarize_shap_df(combined_df_dict):
    """
    This function will compute the mean and standard deviation of Shapley 
    values. In addition, it will take the sign and convert it to a final 
    value. These will be stored to a single dataframe that contains 
    the model results. 
    
    Parameters
    ----------         
    combined_shap_dfs: [dict]
        dictionary containing multiple dataframes for Shapley values.
    
    Returns
    -------
    results_df: [dataframe]
        contains the results for average Shapley + std Shapley. 
    """
    
    # Defining default feature column
    default_col = 'Feature'
    merged_df_args = dict(left_on = default_col, right_on = default_col, how = 'inner')
    
    # Getting merged dataframe
    combined_df_merged = combined_df_dict['mean_abs_shap_df'].merge(combined_df_dict['sign_df'], **merged_df_args)
    
    # Getting only mean cols
    mean_cols = [each_col for each_col in combined_df_merged.columns if each_col.startswith("Mean Shap")]
    
    # Getting only sign cols
    sign_cols = [each_col for each_col in combined_df_merged.columns if each_col.startswith("sign_")]
    
    # Generating a Features dataframe
    results_df = combined_df_merged[['Feature']].copy()
    
    # Getting mean and std of Shapley values
    results_df['Avg_Shap'] = combined_df_merged[mean_cols].mean(axis = 1)
    results_df['Std_Shap'] = combined_df_merged[mean_cols].std(axis = 1)
    
    # Getting the sign of the value
    results_df['Mode_sign'] = combined_df_merged[sign_cols].mode(axis=1)
    
    # Getting +1 for positive or -1 for negative
    sign_array = np.where(results_df['Mode_sign'] == 'positive', 1, -1)
    
    # Adding with sign
    results_df['Avg_Shap_w_Sign'] = results_df['Avg_Shap'] * sign_array
    
    return results_df

# Getting shapley dataframe
def main_compute_shap_df(X_train_df_list,
                         model_list,
                         speed_up = True,
                         ):
    """
    This function computes Shapley dataframe for multiple models and 
    training dataframes.
    
    Parameters
    ----------
    X_train_df_list: list
        X training dataframe list
    model_list: list
        list of the models
    speed_up: logical, optional
        True if you want to speed up the calculation by simplifying features.
        Default value is True.
        
    Returns
    -------
    combined_df_dict: datafrane
        dataframe with all combined values
    results_df: dataframe
        dataframe summarizing the mean and std of each list
    
    """
    
    # Getting dataframes for multiple models
    store_dfs = interpret_multiple_models(model_list = model_list,
                                          X_train_df_list = X_train_df_list,
                                          speed_up = speed_up,
                                          )
    
    # Getting combined dataframes
    combined_df_dict = combine_shap_dfs(store_dfs)    
    # Getting results dataframe
    results_df = summarize_shap_df(combined_df_dict = combined_df_dict)
    
    return combined_df_dict, results_df

# Function to get interpretation of descriptor sets
def compute_interpretation_of_descriptor_sets(storage_descriptor_sets,
                                              model_type_list = None):
    """
    This function iteratively loops through the available descriptor set and 
    computes intepretation for each of them. 
    
    Parameters
    ----------
    storage_descriptor_sets: [dict]
        dictionary containing properties, models, etc.
    model_type_list (list):
        list of models that you want to interpret. If None, then we will 
        iterate across all models.
    
    Returns
    -------
    storage_interpretation_descriptors: [dict]
        dictionary containing all interpretation results for each property, models, etc.
    
    """
    # Start with empty dictionary
    storage_interpretation_descriptors = {}
    
    # Counting total
    total_calc = 0
    n_calcs = len(storage_descriptor_sets)
    
    # Looping through each property
    for idx_property, each_property in enumerate(storage_descriptor_sets):
        
        if idx_property == 0:
            n_calcs *= len(storage_descriptor_sets[each_property])
        
        # Creating empty dict
        storage_interpretation_descriptors[each_property] = {}
        
        # Looping through each dataset
        for idx_dataset, each_dataset in enumerate(storage_descriptor_sets[each_property]):
            
            # Creating empty dict
            storage_interpretation_descriptors[each_property][each_dataset] = {}
            
            # Getting X_df
            descriptor_dict_output = storage_descriptor_sets[each_property][each_dataset]
            
            # Getting X train as a list
            X_train_df_list = generate_X_train_df_list(descriptor_dict_output)
            
            # Getting model list
            if model_type_list is None:
                model_list = descriptor_dict_output['model_storage']
            else:
                model_list = model_type_list
            
            # Adding to number of calculations
            if idx_property == 0:
                n_calcs *= len(model_list)
            
            
            # Looping through each model
            for idx_model, each_model in enumerate(model_list):
            
                # Printing
                print("Interpretation calculation for (%d of %d):"%(total_calc, n_calcs))
                print("--> Property: %s"%(each_property))
                print("--> Dataset: %s"%(each_dataset))
                print("--> Model: %s"%(each_model))
                
                # Getting model list
                model_list = descriptor_dict_output['model_storage'][each_model]['model_list']
                
                # Getting dataframe
                combined_df_dict, results_df = main_compute_shap_df(X_train_df_list = X_train_df_list,
                                                                    model_list = model_list,
                                                                    speed_up = True,
                                                                    )
                
                # Adding to dict
                storage_interpretation_descriptors[each_property][each_dataset][each_model] = dict(
                        combined_df_dict = combined_df_dict,
                        results_df = results_df,
                        )
                
                # Adding to the total
                total_calc += 1
                
    return storage_interpretation_descriptors


#%% Main tool
if __name__ == '__main__':
    
    # Running analysis of the models
    # Defining desired descriptor list
    descriptor_keys_to_use = [
        # 'all_descriptors_10PC',
        'all_descriptors_10PC_with_matminer',
        ]
    
    # Defining property list
    output_property_list = OUTPUT_PROPERTY_LIST
    
    # Defining model type list
    model_type_list = ['lasso', 'RF'] # , 'lightgbm' , 'lightgbm'
    
    # Getting descriptor sets
    storage_descriptor_sets = main_generate_qspr_models_CV(descriptor_keys_to_use = descriptor_keys_to_use,
                                                           output_property_list = OUTPUT_PROPERTY_LIST,
                                                           model_type_list = model_type_list,
                                                           want_normalize = True)
    
    #%% Interpretability of random forest model. 
    
    # Defining current dict
    descriptor_dict_output = storage_descriptor_sets['Final Energy']['all_descriptors_10PC_with_matminer']
    
    # Getting X train
    X_df = descriptor_dict_output['X_df']
    X_train = descriptor_dict_output['fold_list'][0]['X_train']
    
    # Getting dataframe
    X_train_df = pd.DataFrame(X_train, columns = X_df.columns)
    
    # Getting single model
    rf_model = descriptor_dict_output['model_storage']['RF']['model_list'][0]
    

    # Getting interpret rf
    interpret_rf = interpret_models(model = rf_model,
                                       X_train_df = X_train_df)
    

    # Plotting
    fig = interpret_rf.plot_summary(plot_type= 'bar')
    
    # Storing image
    plot_tools.store_figure(fig = fig,
                            path = os.path.join(OUTPUT_DIR,
                                                "RF-SUMMARY"),
                            save_fig = True,    
                            )
    
    
    # Plotting the signs
    descriptor_name = 'vdw surface area/Ang.^2'
    fig, ax = interpret_rf.plot_shap_vs_descriptor(descriptor_name = descriptor_name,)
    
    # Storing image
    plot_tools.store_figure(fig = fig,
                            path = os.path.join(OUTPUT_DIR,
                                                "RF-%s"%('descriptor')),
                            save_fig = True,    
                            )
    

    #%% Interpretability of linear model. 

    
    # Getting single model
    lasso_model = descriptor_dict_output['model_storage']['lasso']['model_list'][0]
    
    # Getting lasso model that is nonzero
    lasso_model_nonzero, X_train_nonzero = remove_lasso_zero_coef(lasso_model = lasso_model,
                                                                  X_train_df = X_train_df
                                                                  )
    
    # Running interpret model code
    interpret_nonzero = interpret_models(model = lasso_model_nonzero,
                                 X_train_df = X_train_nonzero,
                                 )
    
    #%%
    # Running interpret model code
    interpret = interpret_models(model = lasso_model,
                                 X_train_df = X_train_df,
                                 speed_up = True,
                                 )
    

    #%% Plotting for linear model
    # Plotting
    fig = interpret.plot_summary(plot_type= 'bar')

    # Plotting the signs
    fig, ax = interpret.plot_shap_vs_descriptor(descriptor_name = 'vdw surface area/Ang.^2',)
    
    

    #%% Looping through multiple models (Random Forest or Lasso)
    
            
    # Getting X train
    X_train_df_list = generate_X_train_df_list(descriptor_dict_output)
    model_list = descriptor_dict_output['model_storage']['lasso']['model_list']
    
    # Getting dataframe
    combined_df_dict, results_df = main_compute_shap_df(X_train_df_list = X_train_df_list,
                                                        model_list = model_list,
                                                        speed_up = True,
                                                        )
    
    
    #%% Method to plot the importance features
    
    # Plotting importance features
    fig, ax = plot_importance_features(results_df = results_df,
                                 top_n = 5,
                                 fig = None,
                                 ax = None,
                                 fig_size_cm = plot_tools.FIGURE_SIZES_DICT_CM['1_col'],
                                 width = 0.5,
                                 max_string_length = 15,
                                 )
    
    
