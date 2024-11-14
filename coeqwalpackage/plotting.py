# Import standard libraries
import os
import sys
import importlib
import datetime as dt
import time
from pathlib import Path
from contextlib import redirect_stdout

# Import data manipulation libraries
import numpy as np
import pandas as pd

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules - NEED WINDOWS OS
import AuxFunctions as af, cs3, csPlots, cs_util as util, dss3_functions_reference as dss

"""PLOTTING FUNCTIONS"""

def plot_ts(df, pTitle = 'Time Series', xLab = 'Date', lTitle = 'Studies', fTitle = 'mon_tot', pSave = True, fPath = fPath):
    """
    Plots a time-series graph for a given MultiIndex dataframe (follows calsim conventions)
    
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study.
    """
    
    var = '_'.join(df.columns[0][1].split('_')[:-1])
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df.shape[1])]
    colors[-1] = [0,0,0,1]

    count = 0
    
    plt.figure(figsize=(14, 8))
    
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size # Change it to font size you want
    default_line_width = plt.rcParams['lines.linewidth']  
    scaled_line_width = 1.5 * default_line_width
    
    studies = [col[1].split('_')[-1] for col in df.columns]

    for study in studies:
        study_cols = [col for col in df.columns if col[1].endswith(study)]
        for col in study_cols:
            sns.lineplot(data=df, x=df.index, y=col, label=f'{study}', color = colors[count], linewidth=scaled_line_width)
            count+=1
            
    plt.title(var + ' ' + pTitle, fontsize=scaled_font_size*2)
    plt.xlabel(xLab, fontsize=scaled_font_size*1.5)
    plt.ylabel(var+"\nUnits: " + df.columns[0][6], fontsize=scaled_font_size*1.5)

    plt.legend(title=lTitle, title_fontsize = scaled_font_size*1.5, fontsize=scaled_font_size*1.25, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xticks(rotation=45, fontsize=scaled_font_size)  
    plt.yticks(fontsize=scaled_font_size)  
    plt.tight_layout()  
     
    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format = 'png', bbox_inches='tight', dpi=600, transparent=False)
        
    plt.show()
   

def plot_annual_totals(df, xLab = 'Date', pTitle = 'Annual Totals', lTitle = 'Studies', fTitle = 'ann_tot', pSave = True, fPath = fPath):
    """
    Plots a time-series graph of annual totals for a given MultiIndex Dataframe that 
    follows calsim conventions
    
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study. 
    """
    
    annualized_df = pd.DataFrame()
    var = '_'.join(df.columns[0][1].split('_')[:-1])
    studies = [col[1].split('_')[-1] for col in df.columns]
        
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df.shape[1])]
    colors[-1] = [0,0,0,1]
        
    i=0

    plt.figure(figsize=(14, 8))
        
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size # Change it to font size you want
    default_line_width = plt.rcParams['lines.linewidth']  
    scaled_line_width = 1.5 * default_line_width
    
    for study in studies:
        study_cols = [col for col in df.columns if col[1].endswith(study)]
        for col in study_cols:
            with redirect_stdout(open(os.devnull, 'w')):
                df_ann = csPlots.annualize(df.loc[:, [df.columns[i]]])
                annualized_df = pd.concat([annualized_df, df_ann], axis=1)
                annualized_col_name = df_ann.columns[0]
                sns.lineplot(data = df_ann, x=df_ann.index, y=annualized_col_name, label=f'{study}', color = colors[i],
                            linewidth = scaled_line_width)
                i+=1
                    

    plt.title(var + ' ' + pTitle, fontsize=scaled_font_size*2)
    plt.xlabel(xLab, fontsize=scaled_font_size*1.5)
    plt.ylabel(var+"\nUnits: " + df.columns[0][6], fontsize=scaled_font_size*1.5)

    plt.legend(title=lTitle, title_fontsize = scaled_font_size*1.5, fontsize=scaled_font_size*1.25, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xticks(rotation=45, fontsize=scaled_font_size)  
    plt.yticks(fontsize=scaled_font_size)  
    plt.tight_layout()  
        
    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format = 'png', bbox_inches='tight', dpi=600, transparent=False)
        
    plt.show()
    return annualized_df 

def plot_exceedance(df, month = "All Months", xLab = 'Probability', pTitle = 'Exceedance Probability', lTitle = 'Studies', fTitle = 'exceed', pSave = True, fPath = fPath):
    """
    Plots an exceedance graph for a given MultiIndex Dataframe that follows calsim conventions
  
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study. 
    """
    pTitle = pTitle + " " + month
    fTitle = fTitle + " " + month
    
    var = '_'.join(df.columns[0][1].split('_')[:-1])
    studies = [col[1].split('_')[-1] for col in df.columns]
    i=0
    
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df.shape[1])]
    colors[-1] = [0,0,0,1]

    plt.figure(figsize=(14, 8))
            
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size # Change it to font size you want
    default_line_width = plt.rcParams['lines.linewidth']  
    scaled_line_width = 1.5 * default_line_width

    for study in studies:
        study_cols = [col for col in df.columns if col[1].endswith(study)]
        for col in study_cols:
            df_ex = csPlots.single_exceed(df, df.columns[i])
            ex_col_name = df_ex.columns[0]
            sns.lineplot(data = df_ex, x=df_ex.index, y=ex_col_name, label=f'{study}', color = colors[i], linewidth = scaled_line_width)
            i+=1

    plt.title(var + ' ' + pTitle, fontsize=scaled_font_size*2)
    plt.xlabel(xLab, fontsize=scaled_font_size*1.5)
    plt.ylabel(var+"\nUnits: " + df.columns[0][6], fontsize=scaled_font_size*1.5)
    plt.legend(title=lTitle, title_fontsize = scaled_font_size*1.5, fontsize=scaled_font_size*1.25, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xticks(rotation=45, fontsize=scaled_font_size)  
    plt.yticks(fontsize=scaled_font_size)  
    plt.tight_layout()  
    
    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format = 'png', bbox_inches='tight', dpi=600, transparent=False)
        
    plt.show()

def plot_moy_averages(df, xLab = 'Month of Year', pTitle = 'Month of Year Average Totals', lTitle = 'Studies', fTitle = 'moy_avg', fPath = fPath):
    """
    Plots a time-series graph of month of year averages of a study for a given MultiIndex Dataframe that follows calsim conventions. Calculates mean for 12 months across all study years and uses the plot_ts function to produce a graph.
    
    The function assumes the DataFrame columns follow a specific naming
    convention where the last part of the name indicates the study. 
    """
    df_copy = df.copy()
    df_copy["Month"] = df.index.month
    df_moy = df_copy.groupby('Month').mean()
    plot_ts(df_moy, pTitle = pTitle, xLab = xLab, lTitle = lTitle, fTitle = fTitle, fPath = fPath)


"""DIFFERENCE FROM BASELINE"""

def get_difference_from_baseline(df):
    """
    Calculates the difference from baseline for a given variable
    Assumptions: baseline column on first column, df only contains single variable
    """
    df_diff = df.copy()
    baseline_column = df_diff.iloc[:, 0]
    
    for i in range(1, df_diff.shape[1]):
        df_diff.iloc[:, i] = df_diff.iloc[:, i].sub(baseline_column)
    df_diff = df_diff.iloc[:, 1:]

    return df_diff

def difference_from_baseline(df, plot_type, pTitle = 'Difference from Baseline ', xLab = 'Date', lTitle = 'Studies', fTitle = "___", pSave = True, fPath = fPath):
    """
    Plots the difference from baseline of a single variable with a specific plot type
    plot_type parameter inputs: plot_ts, plot_exceedance, plot_moy_averages, plot_annual_totals
    """
    pTitle += plot_type.__name__
    diff_df = get_difference_from_baseline(df)
    plot_type(diff_df, pTitle = pTitle, fTitle = fTitle, fPath = fPath)

"""Looping Through All Variables to Create Plots???????include these?????"""

def slice_with_baseline(df, var, study_lst):
    """
    Creates a subset of df based on varname and slices it according to the provided range.
    """
    subset_df = create_subset(df, var)
    df_baseline = subset_df.iloc[:,[0]]
    df_rest = subset_df.iloc[:, study_lst]
    return pd.concat([df_baseline, df_rest], axis = 1)