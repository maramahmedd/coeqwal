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

"""CONVERT UNITS"""

def load_metadata_df(extract_path, all_data, metadata_file,):
    metadata_df = pd.read_excel(extract_path + metadata_file,
                                engine='openpyxl',
                                skiprows=7,
                                usecols="B:K",
                                nrows=170)
    metadata_df.columns=[
    'Pathnames',
    'Part A',
    'Part B',
    'Part C',
    'UNITS',
    'Part F',
    'Empty1',
    'Col',
    'Empty2',
    'Description'
    ]

    metadata_df.drop(['Empty1', 'Empty2'], axis=1, inplace=True)
    df = pd.read_csv(extract_path + all_data, header=[0, 1, 2, 3, 4, 5, 6], index_col=0, parse_dates=True)
    return metadata_df, df

def convert_cfs_to_taf(df, metadata_df):
    """
    Convert columns from CFS to TAF based on metadata unit specifications.
    
    Parameters:
    - df (pd.DataFrame): The main data DataFrame with multi-level columns.
    - metadata_df (pd.DataFrame): The metadata DataFrame containing 'Part B' and 'UNITS'.
    
    Returns:
    - pd.DataFrame: The DataFrame with converted units where applicable.
    """
    units_mapping = metadata_df.set_index('Part B')['UNITS'].dropna().to_dict()

    print("\nUnits Mapping (First 10):")
    for key, value in list(units_mapping.items())[:10]:
        print(f"{key}: {value}")

    date_column = df.index
    months = date_column.strftime('%m')
    years = date_column.strftime('%Y')

    days_in_month = np.zeros(len(df))

    # Compute the number of days in each month, considering leap years for February
    for i in range(len(months)):
        if months[i] in {"01", "03", "05", "07", "08", "10", "12"}:
            days_in_month[i] = 31
        elif months[i] == "02":
            year = int(years[i])
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month[i] = 29
            else:
                days_in_month[i] = 28
        elif months[i] in {"04", "06", "09", "11"}:
            days_in_month[i] = 30

    columns_to_convert = []
    columns_to_skip = []

    for col in df.columns:
        data_part_b = col[1] 
        data_unit = col[6]     


        matched_part_b = None
        for meta_part_b in units_mapping.keys():
            if meta_part_b in data_part_b:
                matched_part_b = meta_part_b
                break  

        if matched_part_b:
            desired_unit = units_mapping.get(matched_part_b, data_unit)
            if data_unit != desired_unit:
                columns_to_convert.append((col, desired_unit))
            else:
                columns_to_skip.append(col)
        else:
            columns_to_skip.append(col)

    print("\nColumns to Convert:")
    for col, desired_unit in columns_to_convert:
        print(f"{col}: Data Unit = {col[6]}, Desired Unit = {desired_unit}")

    print("\nColumns to Skip:")
    for col in columns_to_skip:
        print(f"{col}: Data Unit = {col[6]}, Desired Unit = {units_mapping.get(col[1], 'No Unit Information')}")

    for col, desired_unit in columns_to_convert:
        data_unit = col[6]
        part_b = col[1]

        if data_unit == 'CFS' and desired_unit == 'TAF':
            print(f"\nConverting column: {col} from {data_unit} to '{desired_unit}'")

            new_values = df[col].values * 0.001984 * days_in_month

            new_col = list(col)
            new_col[6] = 'TAF'  
            new_col = tuple(new_col)

            df[new_col] = new_values

            print(f"Updated column units to 'TAF' for {new_col}")
        else:
            print(f"No defined conversion rule for {data_unit} to {desired_unit}. Skipping.")

    return df


"""SUBSET AND TRANSFORMATION FUNCTIONS"""

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

def create_subset_list(df, var_names):
    """ 
    Filters df to return columns that contain any of the strings in var_names.
    :param df: Dataframe to filter.
    :param var_names: List of variables of interest, e.g. ['S_SHSTA', 'S_OROVL'].
    """
    filtered_columns = df.columns.get_level_values(1).str.contains('|'.join(var_names))
    return df.loc[:, filtered_columns]

"""FORMATTING HELPER FUNCTIONS"""
def set_index(df, dss_names):
    scenario_names = []
    for i in range(len(dss_names)):
        scenario_names.append(dss_names[i][:5])
    df.index = scenario_names
    return df

"""MEAN, SD, IQR, SUM FUNCTIONS"""

def compute_annual_means(df, var, study_lst = None, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units)
    if study_lst is not None:
        subset_df = subset_df.iloc[:, study_lst]
    
    subset_df = add_water_year_column(subset_df)
    
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
        
    annual_mean = subset_df.groupby('WaterYear').mean()
    return annual_mean

def compute_mean(df, variable_list, study_lst, units="TAF", months = None):
    df = compute_annual_means(df, variable_list, study_lst, units, months)
    len_nonnull_yrs = df.dropna().shape[0]
    return (df.sum() / len_nonnull_yrs).iloc[-1]

def compute_sd(df, variable_list, varname, months = None, units="TAF"):
    subset_df = create_subset_unit(df, variable_list, units)
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]

    standard_deviation = subset_df.std().to_frame(name=varname).reset_index(drop=True)
    return standard_deviation

def compute_iqr(df, variable, units, varname, upper_quantile=0.75, lower_quantile=0.25, months=None):
    subset_df = create_subset_unit(df, variable, units)
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
    iqr_values = subset_df.apply(lambda x: x.quantile(upper_quantile) - x.quantile(lower_quantile), axis=0)
    iqr_df = pd.DataFrame(iqr_values, columns=['IQR']).reset_index()[["IQR"]].rename(columns = {"IQR": varname})

    return iqr_df

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

def compute_annual_sums(df, var, study_lst = None, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units).iloc[:, study_lst]
    subset_df = add_water_year_column(subset_df)
    
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
        
    annual_sum = subset_df.groupby('WaterYear').sum()

    return annual_sum

def compute_sum(df, variable_list, study_lst, units, months = None):
    df = compute_annual_sums(df, variable_list, study_lst, units, months)
    return (df.sum()).iloc[-1]


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
        sorted_values = df[column].dropna().sort_values(ascending=False)
        exceedance_probs = sorted_values.rank(method='first', ascending=False) / (1 + len(sorted_values))
        exceedance_df[column] = exceedance_probs.reindex(df.index)

    new_columns = pd.MultiIndex.from_tuples([
        col if isinstance(col, tuple) else (col,) 
        for col in exceedance_df.columns
    ])
    exceedance_df.columns = new_columns
    return exceedance_df

def exceedance_probability(df, var, threshold, month, vartitle):
    # Subset data for the specific variable
    var_df = create_subset_var(df, var)
    # Filter by the specified month and drop NaNs --> only valid values are used in calculating the exceedance probability
    var_month_df = var_df[var_df.index.month.isin([month])].dropna()
    # Count how often the values exceed the threshold and calculate the percentage
    result_df = count_exceedance_days(var_month_df, threshold) / len(var_month_df) * 100
    # Reshape the result to match the expected output format
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    return reshaped_df

def exceedance_metric(df, var, exceedance_percent, vartitle, unit):
    # Extract data for a specific variable in the desired units
    var_df = create_subset_unit(df, var, unit)
    # Drop NaNs and calculate annual flow sums
    annual_flows = calculate_flow_sum_per_year(var_df).iloc[:, 1:].dropna()
    # Calculate exceedance probabilities for valid data only
    exceedance_probs = calculate_exceedance_probabilities(annual_flows)
    # Sort annual flows and exceedance probabilities for thresholding
    annual_flows_sorted = annual_flows.apply(np.sort, axis=0)[::-1]
    """exceedance_prob_baseline = exceedance_probs.apply(np.sort, axis=0).iloc[:, 0].to_frame()
    exceedance_prob_baseline.columns = [“Exceedance Sorted”]"""
    exceedance_prob_baseline = exceedance_probs.apply(np.sort, axis=0)
    if not exceedance_prob_baseline.empty:
        exceedance_prob_baseline = exceedance_prob_baseline.iloc[:, 0].to_frame()
        exceedance_prob_baseline.columns = ["Exceedance Sorted"]
    else:
        raise ValueError("No data available for exceedance probability calculation")
    # Find the index where exceedance probability meets or exceeds the given percentage
    #exceeding_index = exceedance_prob_baseline[exceedance_prob_baseline[‘Exceedance Sorted’] >= exceedance_percent].index[0]
    if 'Exceedance Sorted' not in exceedance_prob_baseline.columns:
        raise KeyError("Column 'Exceedance Sorted' not found in DataFrame")
    filtered_indices = exceedance_prob_baseline.loc[exceedance_prob_baseline['Exceedance Sorted'] >= exceedance_percent].index
    if len(filtered_indices) == 0:
        raise ValueError("No values found meeting the exceedance criteria")
    exceeding_index = filtered_indices[0]
    baseline_threshold = annual_flows_sorted.iloc[len(annual_flows_sorted) - exceeding_index - 1, 0]
    # Count exceedance days, ignoring NaNs
    result_df = count_exceedance_days(annual_flows, baseline_threshold).dropna() / len(annual_flows) * 100
    # Reshape the result for output format
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    return reshaped_df

"""CUSTOM FUNCTIONS using dss_names"""

# Annual Avg (using dss_names)
def ann_avg(df, dss_names, var_name, units="TAF"):
    metrics = []
    for study_index in np.arange(0, len(dss_names)):
        metric_value = compute_mean(df, var_name, [study_index], units, months=None)
        metrics.append(metric_value)

    ann_avg_delta_df = pd.DataFrame(metrics, columns=['Ann_Avg_' + var_name + units])
    ann_avg_delta_df = set_index(ann_avg_delta_df, dss_names)
    return ann_avg_delta_df

# Annual X Percentile outflow of a Delta or X Percentile Resevoir Storage
def ann_percentile(df, dss_names, pct, var_name, df_title, units="TAF"):
    study_list = np.arange(0, len(dss_names))
    iqr_df = compute_iqr_value(df, pct, var_name, units, df_title, study_list, months=None, annual=True)
    iqr_df = set_index(iqr_df, dss_names)
    return iqr_df

# 1 Month Avg using dss_names
def mnth_avg(df, dss_names, var_name, mnth_num, units="TAF"):
    metrics = []
    for study_index in np.arange(0, len(dss_names)):
        metric_value = compute_mean(df, var_name, [study_index], units, months=[mnth_num])
        metrics.append(metric_value)

    mnth_str = calendar.month_abbr[mnth_num]
    mnth_avg_df = pd.DataFrame(metrics, columns=[mnth_str + '_Avg_' + var_name + units])
    mnth_avg_df = set_index(mnth_avg_df, dss_names)
    return mnth_avg_df

# All Months Avg Resevoir Storage or Avg Delta Outflow
def moy_avgs(df, var_name, dss_names, units="TAF"):
    """
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study. 
    """
    var_df = create_subset_var(df, varname=var_name)
    
    all_months_avg = {}
    for mnth_num in range(1, 13):
        metrics = []

        for study_index in np.arange(0, len(dss_names)):
            metric_val = compute_mean(var_df, var_name, [study_index], units, months=[mnth_num])
            metrics.append(metric_val)

        mnth_str = calendar.month_abbr[mnth_num]
        all_months_avg[mnth_str] = np.mean(metrics)
    
    moy_df = pd.DataFrame(list(all_months_avg.items()), columns=['Month', f'moy_Avg_{var_name}_{units}'])
    return moy_df

# Monthly X Percentile Resevoir Storage or X Percentile Delta Outflow
def mnth_percentile(df, dss_names, pct, var_name, df_title, mnth_num, units="TAF"):
    study_list = np.arange(0, len(dss_names))
    mnth_str = calendar.month_abbr[mnth_num]
    df_title = mnth_str + "_" + df_title + "_" + units
    iqr_df = compute_iqr_value(df, pct, var_name, units, df_title, study_list, months = [mnth_num], annual = True)
    iqr_df = set_index(iqr_df, dss_names)
    return iqr_df

def annual_totals(df, var_name, units):
    """
    Plots a time-series graph of annual totals for a given MultiIndex Dataframe that 
    follows calsim conventions
    
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study. 
    """
    df = create_subset_unit(df, var_name, units)
    
    annualized_df = pd.DataFrame()
    var = '_'.join(df.columns[0][1].split('_')[:-1])
    studies = [col[1].split('_')[-1] for col in df.columns]
        
    i=0
    for study in studies:
        study_cols = [col for col in df.columns if col[1].endswith(study)]
        for col in study_cols:
            with redirect_stdout(open(os.devnull, 'w')):
                temp_df = df.loc[:,[df.columns[i]]]
                temp_df["Year"] = df.index.year
                df_ann = temp_df.groupby("Year").sum()
                annualized_df = pd.concat([annualized_df, df_ann], axis=1)
                i+=1
                
    return annualized_df 


"""Calculate Frequency Hitting Floodzone/Deadpool Levels"""

def frequency_hitting_level(df, var_res, var_fldzn, units, vartitle, floodzone = True, months = None):
    """
    Calculate the frequency of hitting the floodzone or deadpool levels
    Use floodzone = True to calculate probability hitting floodzone, and False to calculate hitting deadpool levels
    """
    subset_df_res = create_subset_unit(df, var_res, units)
    subset_df_floodzone = create_subset_unit(df, var_fldzn, units)

    if months is not None:
        subset_df_res = subset_df_res[subset_df_res.index.month.isin(months)]
        subset_df_floodzone = subset_df_floodzone[subset_df_floodzone.index.month.isin(months)]

    multiindex_columns = subset_df_res.columns

    subset_df_res_comp_values = subset_df_res.values - subset_df_floodzone.values
    if floodzone:
        subset_df_res_comp_values += 0.000001

    subset_df_res_comp = pd.DataFrame(subset_df_res_comp_values, index=subset_df_res.index, columns=multiindex_columns)

    exceedance_days = count_exceedance_days(subset_df_res_comp, 0)
    exceedance_days = exceedance_days / len(subset_df_res_comp) * 100
    if not floodzone:
        exceedance_days = 100 - exceedance_days


    exceedance_days = exceedance_days.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]

    return exceedance_days

"""OLD VERSIONS OF FUNCTIONS"""

"""def exceedance_metric(df, var, exceedance_percent, vartitle, unit):
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

    return reshaped_df"""

"""def exceedance_probability(df, var, threshold, month, vartitle):
    var_df = create_subset_var(df, var)
    var_month_df = var_df[var_df.index.month.isin([month])]
    result_df = count_exceedance_days(var_month_df, threshold) / len(var_month_df) * 100
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    return reshaped_df"""

"""def calculate_exceedance_probabilities(df):
    exceedance_df = pd.DataFrame(index=df.index)

    for column in df.columns:
       sorted_values = df[column].sort_values(ascending=False)
       exceedance_probs = (sorted_values.rank(method='first', ascending=False)) / (1 + len(sorted_values))
       exceedance_df[column] = exceedance_probs.sort_index()

    return exceedance_df"""

"""def compute_annual_means(df, var, study_lst = None, units = "TAF", months = None):
    subset_df = create_subset_unit(df, var, units)
    if study_lst is not None:
        subset_df = subset_df.iloc[:, study_lst]
    
    subset_df = add_water_year_column(subset_df)
    
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
        
    annual_mean = subset_df.groupby('WaterYear').mean()
    return annual_mean"""
