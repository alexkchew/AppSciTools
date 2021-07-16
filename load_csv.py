#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_csv.py

This script controlls all load csv information.

Created on: Fri Jul 16 15:54:43 2021
Author: Alex K. Chew (alex.chew@schrodinger.com)

Copyright Schrodinger, LLC. All rights reserved.
"""


# Loading modules
import os
import pandas as pd
import glob

# Importing filtration tools
from .filtration import filter_by_variance_threshold

# Defining default columns
DEFAULT_INDEX_COLS = ["Title", "Entry Name"]

# Loading experimental data
def load_property_data(csv_data_path,
                       keep_list = []):
    """
    This function loads property data from spreadsheet
    Parameters
    ----------
    csv_data_path: [str]
        path to csv file
    keep_list: [list, default = []]
        list of columns to keep. If None, the entire dataframe is outputted.
    
    Returns
    -------
    csv_data: [df]
        dataframe containing csv information with the keep list

    """
    # Loading dataframe
    csv_data = pd.read_csv(csv_data_path)
    
    # Checking if list is empty
    if len(keep_list) == 0:
        return csv_data
    else:
        return csv_data[keep_list]

# Function to load descriptor data
def load_descriptor_data(csv_path,
                         clean_data = True,
                         default_index_cols = DEFAULT_INDEX_COLS):
    """
    This function loads the descriptor information. Note that all:
        - non-numerical descriptors are removed automatically. 
        - missing NaN columns are removed automatically

    Parameters
    ----------
    csv_path : str
        Path to csv file
    clean_data: logical, default = True
        True if you want to clean the data by removing non-numerical descriptors / NaN columns

    Returns
    -------
    output_df : str
        dataframe containing csv file

    """
    # Loading csv file
    csv_df = pd.read_csv(csv_path)
    
    # Checking if you want to clean the dataframe
    if clean_data is True:
        
        # Cleaning the dataframe
        csv_df_nonan = csv_df.dropna(axis=1) # Removes NaN values
        csv_df_nums = csv_df_nonan.select_dtypes(['number']) # Numbers only stored
        
        # Removing cols with low variance
        output_df = filter_by_variance_threshold(X_df = csv_df_nums)
        
        # Adding back the index cols to the beginning
        for each_col in default_index_cols[::-1]: # Reverse order
            if each_col in csv_df:
                output_df.insert (0, each_col, csv_df[each_col])
        return output_df
    else:
        return csv_df
    
# Function to load multiple descriptor datas
def load_multiple_descriptor_data(default_csv_paths,
                                  descriptor_list = ["2d_descriptors",
                                                     "3d_descriptors",],
                                 ):
    """
    This function loads multiple descriptor data given a descriptor list.

    Parameters
    ----------
    default_csv_paths: dict
        dictionary of csv paths
    descriptor_list : list
        list of descriptors to load from dictionary

    Returns
    -------
    descriptor_df_dict: dict
        dictionary containing descritpors

    """
    # Loading all descriptor files
    descriptor_df_dict = { each_descriptor_key: load_descriptor_data(default_csv_paths[each_descriptor_key]) 
                                                              for each_descriptor_key in descriptor_list }
    
    return descriptor_df_dict

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