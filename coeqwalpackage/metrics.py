"""IMPORTS"""
import os
import sys
import importlib
import datetime as dt
import time
from pathlib import Path
from contextlib import redirect_stdout
import calendar

# Import data manipulation libraries
import numpy as np
import pandas as pd

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


"""READ FUNCTIONS"""

def read_in_df(df_path, names_path):
    """returns a df, dss_names in correct format
        df_path = path to extracted and converted data csv (ex. '../output/convert/convert_EDA_data_10_01_24.csv')
        names_path = path to extracted dss names csv (ex. '../data/metrics/EDA_10_01_24_dss_names.csv') 
    """
    df = pd.read_csv(df_path, header=[0, 1, 2, 3, 4, 5, 6], index_col=0, parse_dates=True)
    dss_names = pd.read_csv(names_path)["0"].tolist()
    return df, dss_names


"""SUBSET FUNCTIONS"""

def add_water_year_column(df):
    df_copy = df.copy().sort_index()
    df_copy['Date'] = pd.to_datetime(df_copy.index)
    df_copy.loc[:, 'Year'] = df_copy['Date'].dt.year
    df_copy.loc[:, 'Month'] = df_copy['Date'].dt.month
    df_copy.loc[:, 'WaterYear'] = np.where(df_copy['Month'] >= 10, df_copy['Year'] + 1, df_copy['Year'])
    return df_copy.drop(["Date", "Year", "Month"], axis=1)

def create_subset_var(df, varname):
    """ 
    Filters df to return columns that contain the string varname
    :param df: Dataframe to filter
    :param varname: variable of interest, e.g. S_SHSTA
    """
    filtered_columns = df.columns.get_level_values(1).str.contains(varname)
    return df.loc[:, filtered_columns]

def create_subset_unit(df, varname, units):
    """ 
    Filters df to return columns that contain the string varname and units
    :param df: Dataframe to filter
    :param varname: variable of interest, e.g. S_SHSTA
    :param units: units of interest
    """
    var_filter = df.columns.get_level_values(1).str.contains(varname)
    unit_filter = df.columns.get_level_values(6).str.contains(units)
    filtered_columns = var_filter & unit_filter
    return df.loc[:, filtered_columns]


"""MEAN, SD, IQR FUNCTIONS"""

def compute_annual_means(df, var, study_lst = None, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units)
    if study_lst is not None:
        subset_df = subset_df.iloc[:, study_lst]
    
    subset_df = add_water_year_column(subset_df)
    
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
        
    annual_sum = subset_df.groupby('WaterYear').sum()
    return annual_sum

def compute_mean(df, variable_list, study_lst, units, months = None):
    df = compute_annual_means(df, variable_list, study_lst, units, months)
    num_years = len(df)
    return (df.sum() / num_years).iloc[-1]

def compute_sd(df, variable_list, units, varname, months = None):
    subset_df = create_subset_unit(df, variable_list, units)
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]

    standard_deviation = subset_df.std().to_frame(name=varname).reset_index(drop=True)
    return standard_deviation

def compute_iqr_value(df, iqr_value, variable, units, varname, study_list, months=None, annual=True):
    if annual:
        subset_df = compute_annual_means(create_subset_unit(df, variable, units), variable, study_list, units, months)
    else:
        subset_df = create_subset_unit(df, variable, units)
        if months is not None:
            subset_df = subset_df[subset_df.index.month.isin(months)]
    iqr_values = subset_df.apply(lambda x: x.quantile(iqr_value), axis=0)
    iqr_df = pd.DataFrame(iqr_values, columns=['IQR']).reset_index()[["IQR"]].rename(columns = {"IQR": varname})
    return iqr_df

def calculate_monthly_average(flow_data):
    flow_data = flow_data.reset_index()
    flow_data['Date'] = pd.to_datetime(flow_data.iloc[:, 0])

    flow_data.loc[:, 'Month'] = flow_data['Date'].dt.strftime('%m')
    flow_data.loc[:, 'Year'] = flow_data['Date'].dt.strftime('%Y')

    flow_values = flow_data.iloc[:, 1:]
    monthly_avg = flow_values.groupby(flow_data['Month']).mean().reset_index()

    monthly_avg.rename(columns={'Month': 'Month'}, inplace=True)
    return monthly_avg


"""EXCEEDANCE FUNCTIONS"""

def count_exceedance_days(data, threshold):
    """
    Count the number of days in the data that exceed a given threshold
    """
    exceedance_counts = pd.DataFrame(np.nan, index=[0], columns=data.columns)

    for col in data.columns:
        exceedance_counts.loc[0, col] = (data[col] > threshold).sum()
    return exceedance_counts

def calculate_flow_sum_per_year(flow_data):
    """
    Calculate the annual total of the given data per year
    :NOTE: This was translated from Abhinav's code and is only used in the exceedance_metric function
    """
    flow_data = add_water_year_column(flow_data)
    flow_sum_per_year = flow_data.groupby('WaterYear').sum(numeric_only=True).reset_index()

    return flow_sum_per_year

def calculate_exceedance_probabilities(df):
    exceedance_df = pd.DataFrame(index=df.index)

    for column in df.columns:
        sorted_values = df[column].sort_values(ascending=False)
        exceedance_probs = (sorted_values.rank(method='first', ascending=False)) / (1 + len(sorted_values))
        exceedance_df[column] = exceedance_probs.sort_index()

    return exceedance_df

def exceedance_probability(df, var, threshold, month, vartitle):
    var_df = create_subset_var(df, var)
    var_month_df = var_df[var_df.index.month.isin([month])]
    result_df = count_exceedance_days(var_month_df, threshold) / len(var_month_df) * 100
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    return reshaped_df

def exceedance_metric(df, var, exceedance_percent, vartitle, unit):
    var_df = create_subset_unit(df, var, unit)
    annual_flows = calculate_flow_sum_per_year(var_df).iloc[:, 1:]
    exceedance_probs = calculate_exceedance_probabilities(annual_flows)

    annual_flows_sorted = annual_flows.apply(np.sort, axis=0)[::-1]
    exceedance_prob_baseline = exceedance_probs.apply(np.sort, axis=0).iloc[:, 0].to_frame()
    exceedance_prob_baseline.columns = ["Exceedance Sorted"]

    exceeding_index = exceedance_prob_baseline[exceedance_prob_baseline['Exceedance Sorted'] >= exceedance_percent].index[0]
    baseline_threshold = annual_flows_sorted.iloc[len(annual_flows_sorted) - exceeding_index - 1, 0]

    result_df = count_exceedance_days(annual_flows, baseline_threshold) / len(annual_flows) * 100
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]

    return reshaped_df