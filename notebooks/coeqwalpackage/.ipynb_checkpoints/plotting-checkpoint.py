# import metrics library
from metrics import *

# Import visualization libraries
from matplotlib import colormaps, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules - NEED WINDOWS OS
#import AuxFunctions as af, cs3, csPlots, cs_util as util, dss3_functions_reference as dss

"""PLOTTING FUNCTIONS"""

def plot_ts(
        df,
        varname,
        units='TAF',
        pTitle='Time Series',
        xLab='Date',
        lTitle='Studies',
        fTitle='mon_tot',
        pSave=True,
        fPath='fPath',
        study_list=None,
        start_date=None,
        end_date=None,
        scenario_styles=None
):
    """
    Plots a time-series graph for a given MultiIndex DataFrame (CalSim style).
    Allows optional subsetting by study_list and date range, plus scenario styles.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with columns like (PartA, PartB, ..., Units).
    varname : str
        Variable name/pattern to filter, e.g. "S_SHSTA_".
    units : str
        Units to filter, e.g. "TAF", "CFS", etc.
    pTitle : str
        Plot title.
    xLab : str
        X-axis label.
    lTitle : str
        Legend title.
    fTitle : str
        Filename prefix for saving.
    pSave : bool
        Whether to save the figure as PNG.
    fPath : str
        Directory to save the PNG.
    study_list : list of int, optional
        e.g. [2,11] => subselect columns ending in "_s0002" or "_s0011".
    start_date : str or datetime, optional
        e.g. '1975-10-31'. Subset rows after this date.
    end_date : str or datetime, optional
        e.g. '1978-09-30'. Subset rows before this date.
    scenario_styles : dict, optional
        e.g. {2: {'color':'black', 'linestyle':'-', 'label':'Baseline'}}.

    Returns
    -------
    None
    """
    df_plot = create_subset_unit(df, varname, units)

    if start_date is not None:
        df_plot = df_plot.loc[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot.loc[df_plot.index <= pd.to_datetime(end_date)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_ts] WARNING: No data to plot after subsetting!")
        return

    var = '_'.join(df_plot.columns[0][1].split('_')[:-1])
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df_plot.shape[1])]
    if len(colors) > 0:
        colors[-1] = (0, 0, 0, 1)

    plt.figure(figsize=(14, 8))
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size
    scaled_line_width = 1.5 * plt.rcParams['lines.linewidth']

    studies = [col[1].split('_')[-1] for col in df_plot.columns]
    scenario_labeled = set()
    count = 0

    for col, study in zip(df_plot.columns, studies):
        numeric_study = int(study.replace('s', ''))
        if scenario_styles and numeric_study in scenario_styles:
            style_dict = scenario_styles[numeric_study]
            this_color = style_dict.get('color', colors[count])
            this_linestyle = style_dict.get('linestyle', '-')
            this_label = style_dict.get('label', study) if study not in scenario_labeled else None
            this_lw = style_dict.get('linewidth', scaled_line_width)
        else:
            this_color = colors[count]
            this_linestyle = '-'
            this_label = study if study not in scenario_labeled else None
            this_lw = scaled_line_width

        sns.lineplot(
            data=df_plot,
            x=df_plot.index,
            y=col,
            label=this_label,
            color=this_color,
            linewidth=this_lw,
            linestyle=this_linestyle
        )

        if this_label is not None:
            scenario_labeled.add(study)

        count += 1

    plt.title(var + ' ' + pTitle, fontsize=scaled_font_size * 2)
    plt.xlabel(xLab, fontsize=scaled_font_size * 1.5)
    first_col_units = df_plot.columns[0][6]
    plt.ylabel(var + "\nUnits: " + str(first_col_units), fontsize=scaled_font_size * 1.5)

    plt.legend(
        title=lTitle,
        title_fontsize=scaled_font_size * 1.5,
        fontsize=scaled_font_size * 1.25,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )
    plt.xticks(rotation=45, fontsize=scaled_font_size)
    plt.yticks(fontsize=scaled_font_size)
    plt.tight_layout()

    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format='png', bbox_inches='tight', dpi=600, transparent=False)

    plt.show()


def plot_annual_totals(
        df,
        varname,
        units='TAF',
        xLab='Date',
        pTitle='Annual Totals',
        lTitle='Studies',
        fTitle='ann_tot',
        pSave=True,
        fPath='fPath',
        study_list=None,
        start_date=None,
        end_date=None,
        scenario_styles=None,
        months=None
):
    """
    Plots a time-series graph of annual totals for a given MultiIndex DataFrame
    that follows CalSim conventions, using annualize_ts for each column.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with columns shaped like (PartA, PartB, ..., Units).
        The index must be a DatetimeIndex.
    varname : str
        Variable name/pattern to filter, e.g. "DEL_SOD_AG_".
    units : str
        Units to filter, e.g. "TAF", "CFS", etc.
    xLab : str
        X-axis label.
    pTitle : str
        Plot title.
    lTitle : str
        Legend title.
    fTitle : str
        Filename prefix for saving.
    pSave : bool
        Whether to save the figure as PNG.
    fPath : str
        Directory to save the PNG.
    study_list : list of int, optional
        e.g. [2,11] => columns ending in "_s0002" or "_s0011".
    start_date : str or datetime, optional
        Subset rows after this date (inclusive).
    end_date : str or datetime, optional
        Subset rows before this date (inclusive).
    scenario_styles : dict, optional
    months : list of int, optional
        Subset the data to these months (1=Jan, ..., 12=Dec).

    Returns
    -------
    pd.DataFrame
        The concatenated annual DataFrame for all columns (after summation).
    """
    df_plot = create_subset_unit(df, varname, units)

    if start_date is not None:
        df_plot = df_plot.loc[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot.loc[df_plot.index <= pd.to_datetime(end_date)]

    if months is not None:
        df_plot = df_plot[df_plot.index.month.isin(months)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_annual_totals] WARNING: No data after subsetting!")
        return pd.DataFrame()

    var = '_'.join(df_plot.columns[0][1].split('_')[:-1])
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df_plot.shape[1])]
    if len(colors) > 0:
        colors[-1] = (0, 0, 0, 1)

    plt.figure(figsize=(14, 8))
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size
    scaled_line_width = 1.5 * plt.rcParams['lines.linewidth']

    annualized_df = pd.DataFrame()

    studies = [col[1].split('_')[-1] for col in df_plot.columns]
    scenario_labeled = set()
    count = 0

    for col, study in zip(df_plot.columns, studies):
        numeric_study = int(study.replace('s',''))
        if scenario_styles and numeric_study in scenario_styles:
            style_dict = scenario_styles[numeric_study]
            this_color = style_dict.get('color', colors[count])
            this_linestyle = style_dict.get('linestyle', '-')
            this_label = style_dict.get('label', study) if (study not in scenario_labeled) else None
            this_lw = style_dict.get('linewidth', scaled_line_width)
        else:
            this_color = colors[count]
            this_linestyle = '-'
            this_label = study if study not in scenario_labeled else None
            this_lw = scaled_line_width

        from contextlib import redirect_stdout
        with redirect_stdout(open(os.devnull, 'w')):
            single_col_df = df_plot[[col]]
            df_ann = annualize_ts(single_col_df, freq='YS-OCT')

        annualized_df = pd.concat([annualized_df, df_ann], axis=1)

        sns.lineplot(
            data=df_ann,
            x=df_ann.index,
            y=df_ann.columns[0],
            label=this_label,
            color=this_color,
            linewidth=this_lw,
            linestyle=this_linestyle,
            drawstyle='steps-post'
        )

        if this_label is not None:
            scenario_labeled.add(study)
        count += 1

    plt.title(f"{var} {pTitle}", fontsize=scaled_font_size * 2)
    plt.xlabel(xLab, fontsize=scaled_font_size * 1.5)
    first_col_units = df_plot.columns[0][6]
    plt.ylabel(var + "\nUnits: " + str(first_col_units), fontsize=scaled_font_size * 1.5)

    plt.legend(
        title=lTitle,
        title_fontsize=scaled_font_size * 1.5,
        fontsize=scaled_font_size * 1.25,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )
    plt.xticks(rotation=45, fontsize=scaled_font_size)
    plt.yticks(fontsize=scaled_font_size)
    plt.tight_layout()

    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format='png', bbox_inches='tight', dpi=600, transparent=False)

    plt.show()
    return annualized_df

def plot_exceedance(
        df,
        varname,
        units='TAF',
        xLab='Probability',
        pTitle='Exceedance Probability',
        lTitle='Studies',
        fTitle='exceed',
        pSave=True,
        fPath='fPath',
        study_list=None,
        scenario_styles=None,
        months=None
):
    """
    Plots an exceedance graph for a given MultiIndex DataFrame (follows CalSim conventions)
    using single_exceed_alternative() for each column.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with columns shaped like (PartA, PartB, ..., Units).
        Index must be a DatetimeIndex if you're subsetting by month.
    varname : str
        Variable name/pattern to filter.
    units : str
        Units to filter.
    xLab : str
        X-axis label.
    pTitle : str
        Base for the plot title.
    lTitle : str
        Legend title.
    fTitle : str
        Filename prefix for saving.
    pSave : bool
        Whether to save the figure as PNG.
    fPath : str
        Directory to save the PNG.
    study_list : list of int, optional
        e.g. [2,11] => columns ending in "_s0002" or "_s0011".
    scenario_styles : dict, optional
    months : list of int, optional
        Subset the data to these months (1=Jan, 2=Feb, ..., 12=Dec).

    Returns
    -------
    None
    """

    df_plot = create_subset_unit(df, varname, units)

    if months is not None:
        df_plot = df_plot[df_plot.index.month.isin(months)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_exceedance] WARNING: No data to plot after subsetting!")
        return

    var = '_'.join(df_plot.columns[0][1].split('_')[:-1])
    studies = [col[1].split('_')[-1] for col in df_plot.columns]

    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df_plot.shape[1])]
    if len(colors) > 0:
        colors[-1] = (0, 0, 0, 1)

    plt.figure(figsize=(14, 8))
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size
    scaled_line_width = 1.5 * plt.rcParams['lines.linewidth']

    scenario_labeled = set()

    for i, (col, study) in enumerate(zip(df_plot.columns, studies)):
        numeric_study = int(study.replace('s',''))
        if scenario_styles and numeric_study in scenario_styles:
            style_dict = scenario_styles[numeric_study]
            this_color = style_dict.get('color', colors[i])
            this_linestyle = style_dict.get('linestyle', '-')
            this_label = style_dict.get('label', study) if (study not in scenario_labeled) else None
            this_lw = style_dict.get('linewidth', scaled_line_width)
        else:
            this_color = colors[i]
            this_linestyle = '-'
            this_label = study if study not in scenario_labeled else None
            this_lw = scaled_line_width

        df_ex = single_exceed_alternative(df_plot[col])
        ex_col_name = df_ex.columns[0]

        sns.lineplot(
            data=df_ex,
            x=df_ex.index,
            y=ex_col_name,
            label=this_label,
            color=this_color,
            linewidth=this_lw,
            linestyle=this_linestyle
        )

        if this_label:
            scenario_labeled.add(study)

    plt.title(f"{var} {pTitle}", fontsize=scaled_font_size * 2)
    plt.xlabel(xLab, fontsize=scaled_font_size * 1.5)
    first_col_units = df_plot.columns[0][6]
    plt.ylabel(f"{var}\nUnits: {first_col_units}", fontsize=scaled_font_size * 1.5)

    plt.legend(
        title=lTitle,
        title_fontsize=scaled_font_size * 1.5,
        fontsize=scaled_font_size * 1.25,
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )
    plt.xticks(rotation=45, fontsize=scaled_font_size)
    plt.yticks(fontsize=scaled_font_size)
    plt.tight_layout()

    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format='png', bbox_inches='tight', dpi=600, transparent=False)

    plt.show()

def single_exceed_alternative(series):
    """
    Compute exceedance probabilities for a given Pandas Series.

    Returns a DataFrame with:
      - index = exceedance probability (0..1)
      - one column with the sorted (descending) data
    """
    s_clean = series.dropna()
    s_sorted = s_clean.sort_values(ascending=False)
    n = len(s_sorted)
    ranks = np.arange(1, n + 1)
    exceed_probs = ranks / (n + 1.0)
    out_df = pd.DataFrame(data=s_sorted.values, index=exceed_probs, columns=[series.name])
    return out_df

def annualize_exceedance_plot(
        df,
        varname,
        units="TAF",
        freq="YS-OCT",
        pTitle='Annual Exceedance',
        xLab='Probability',
        lTitle='Studies',
        fTitle='annual_exceed',
        pSave=True,
        fPath='fPath',
        study_list=None,
        scenario_styles=None,
        months=None
):
    """
    Subset df to varname in given units, optionally subset to specific months,
    annualize by water year (or chosen freq), then pass the annual data
    into plot_exceedance.
    """

    df_subset = create_subset_unit(df, varname, units)

    if months is not None:
        df_subset = df_subset[df_subset.index.month.isin(months)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        keep_cols = []
        for col in df_subset.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                keep_cols.append(col)
        df_subset = df_subset[keep_cols]

    if df_subset.empty:
        print("[annualize_exceedance_plot] WARNING: No data after subsetting!")
        return pd.DataFrame()

    annual_cols = []
    for col in df_subset.columns:
        one_col = df_subset[[col]]
        ann_col = one_col.resample(freq).sum(min_count=1)
        annual_cols.append(ann_col)

    annual_df = pd.concat(annual_cols, axis=1)

    plot_exceedance(
        annual_df,
        varname=varname,
        units=units,
        xLab=xLab,
        pTitle=pTitle,
        lTitle=lTitle,
        fTitle=fTitle,
        pSave=pSave,
        fPath=fPath,
        study_list=None,
        scenario_styles=scenario_styles,
        months=None
    )

    return annual_df

def annualize_ts(df_col, freq='YS-OCT'):
    """
    Aggregate a single-column DataFrame to annual totals by water year,
    assuming the column is already in volumetric units (e.g. TAF).

    Parameters
    ----------
    df_col : pd.DataFrame
        Exactly ONE column with a DateTimeIndex.
    freq : str
        Defaults to 'YS-OCT', meaning the water year starts in October.

    Returns
    -------
    pd.DataFrame (one column) with annual sums, indexed by Water Year start date.
    """
    if df_col.shape[1] != 1:
        raise ValueError("annualize_ts expects exactly ONE column in df_col")
    df_annual = df_col.resample(freq).sum(min_count=1)
    return df_annual

def plot_moy_averages( 
        df,
        varname,
        units='TAF',
        xLab='Month of Year',
        pTitle='Month of Year Average Totals',
        lTitle='Studies',
        fTitle='moy_avg',
        pSave=True,
        fPath='fPath',
        # OPTIONAL
        study_list=None,
        scenario_styles=None,
):
    """
    Plots a time-series graph of month-of-year averages (1..12) for each column,
    then calls plot_ts on the result. No date subsetting.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with columns shaped like (PartA, PartB, ..., Units).
    varname : str
        Variable name/pattern to filter.
    units : str
        Units to filter.
    xLab : str
        X-axis label.
    pTitle : str
        Plot title.
    lTitle : str
        Legend title.
    fTitle : str
        Filename prefix for saving.
    pSave : bool
        Whether to save plot.
    fPath : str
        Directory to save.
    study_list : list of int, optional
        e.g. [2,11]
    scenario_styles : dict, optional
        e.g. {2: {'color':'black','linestyle':'-','label':'Baseline'}}

    Returns
    -------
    None
    """
    df_plot = create_subset_unit(df, varname, units)

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_moy_averages] WARNING: No data after subsetting!")
        return

    df_copy = df_plot.copy()
    df_copy["Month"] = df_copy.index.month
    df_moy = df_copy.groupby("Month").mean(numeric_only=True)

    plot_ts(
        df_moy,
        varname=varname,
        units=units,
        pTitle=pTitle,
        xLab=xLab,
        lTitle=lTitle,
        fTitle=fTitle,
        pSave=pSave,
        fPath=fPath,
        study_list=None,
        start_date=None,
        end_date=None,
        scenario_styles=scenario_styles
    )

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

def difference_from_baseline(df, plot_type, pTitle = 'Difference from Baseline ', xLab = 'Date', lTitle = 'Studies', fTitle = "___", pSave = True, fPath = 'fPath'):
    """
    Plots the difference from baseline of a single variable with a specific plot type
    plot_type parameter inputs: plot_ts, plot_exceedance, plot_moy_averages, plot_annual_totals
    """
    pTitle += plot_type.__name__
    diff_df = get_difference_from_baseline(df)
    plot_type(diff_df, pTitle = pTitle, fTitle = fTitle, fPath = 'fPath')

"""Looping Through All Variables to Create Plots???????include these?????"""

def slice_with_baseline(df, var, study_lst):
    """
    Creates a subset of df based on varname and slices it according to the provided range.
    """
    subset_df = create_subset_var(df, var)
    df_baseline = subset_df.iloc[:,[0]]
    df_rest = subset_df.iloc[:, study_lst]
    return pd.concat([df_baseline, df_rest], axis = 1)


"""PARALLEL PLOTS"""
#source: https://reedgroup.github.io/FigureLibrary/ParallelCoordinatesPlots.html

figsize = (18,6)
fontsize = 14
main_data_dir = "../output/metrics/"
data_dir_knobs = "../data/parallelplots/"
fig_dir = '../output/parallelplots/'

"""Functions for flexible parallel coordinates plots"""
def reorganize_objs(objs, columns_axes, ideal_direction, minmaxs):
    ### function normalize data based on direction of preference and whether each objective is minimized or maximized
    ### -> output dataframe will have values ranging from 0 (which maps to bottom of figure) to 1 (which maps to top)
    ### if min/max directions not given for each axis, assume all should be maximized
    if minmaxs is None:
        minmaxs = ['max']*len(columns_axes)

    ### get subset of dataframe columns that will be shown as parallel axes
    objs_reorg = objs[columns_axes]

    ### reorganize & normalize data to go from 0 (bottom of figure) to 1 (top of figure), 
    ### based on direction of preference for figure and individual axes
    if ideal_direction == 'bottom':
        tops = objs_reorg.min(axis=0)
        bottoms = objs_reorg.max(axis=0)
        for i, minmax in enumerate(minmaxs):
            if minmax == 'max':
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i]) / \
                                        (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i].min(axis=0))
            else:
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, -1] = (objs_reorg.iloc[:, -1] - objs_reorg.iloc[:, -1].min(axis=0)) / \
                                         (objs_reorg.iloc[:, -1].max(axis=0) - objs_reorg.iloc[:, -1].min(axis=0))
    elif ideal_direction == 'top':
        tops = objs_reorg.max(axis=0)
        bottoms = objs_reorg.min(axis=0)
        for i, minmax in enumerate(minmaxs):
            if minmax == 'max':
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i] - objs_reorg.iloc[:, i].min(axis=0)) / \
                                        (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i].min(axis=0))
            else:
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i]) / \
                                        (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i].min(axis=0))

    return objs_reorg, tops, bottoms

def get_color(value, color_by_continuous, color_palette_continuous, color_by_categorical, color_dict_categorical):
# get color based on continuous color map or categorical map
    if color_by_continuous is not None:
        color = plt.get_cmap(color_palette_continuous)(value)
    elif color_by_categorical is not None:
        color = color_dict_categorical[value]
    return color

def get_zorder(norm_value, zorder_num_classes, zorder_direction):
# Get zorder value for ordering lines on plot.
# Works by binning a given axis' values and mapping to discrete classes.
    xgrid = np.arange(0, 1.001, 1/zorder_num_classes)
    if zorder_direction == 'ascending':
        return 4 + np.sum(norm_value > xgrid)
    elif zorder_direction == 'descending':
        return 4 + np.sum(norm_value < xgrid)
    

def custom_parallel_coordinates(objs, columns_axes=None, axis_labels=None, ideal_direction='top', minmaxs=None, color_by_continuous=None, color_palette_continuous=None, color_by_categorical=None, color_palette_categorical=None, colorbar_ticks_continuous=None, color_dict_categorical=None, zorder_by=None, zorder_num_classes=10, zorder_direction='ascending', alpha_base=0.8, brushing_dict=None, alpha_brush=0.05, lw_base=1.5, fontsize=14, figsize=(11,6), save_fig_filename=None):
    # customizable parallel coordinates plot
    # COLUMN AXES???
    """
    Parameters:
    objs (DataFrame): The DataFrame containing the data to plot.
    columns_axes (list, optional): List of column names to use as axes. Defaults to all columns.
    axis_labels (list, optional): List of axis labels. Defaults to the same as columns_axes.
    ideal_direction (str, optional): Direction of preference for objective values. Can be 'top' or 'bottom'. Defaults to 'top'.
    minmaxs (list, optional): List of 'max' or 'min' for each column, indicating if higher or lower values are better.
    color_by_continuous (str, optional): Column name to color lines by continuous values. Cannot be used with color_by_categorical.
    color_palette_continuous (Colormap, optional): Colormap to use for continuous coloring.
    color_by_categorical (str, optional): Column name to color lines by categorical values. Cannot be used with color_by_continuous.
    color_palette_categorical (list, optional): List of colors to use for categorical coloring.
    colorbar_ticks_continuous (list, optional): List of tick values for the continuous colorbar.
    color_dict_categorical (dict, optional): Dictionary mapping categorical values to colors.
    zorder_by (str, optional): Column name to determine the z-order (layer) of the lines.
    zorder_num_classes (int, optional): Number of classes for z-ordering. Defaults to 10.
    zorder_direction (str, optional): Direction for z-ordering. Can be 'ascending' or 'descending'. Defaults to 'ascending'.
    alpha_base (float, optional): Base alpha (transparency) value for the lines. Defaults to 0.8.
    brushing_dict (dict, optional): Dictionary specifying brushing criteria in the form {column_index: (threshold, operator)}.
    alpha_brush (float, optional): Alpha value for lines that do not meet brushing criteria. Defaults to 0.05.
    lw_base (float, optional): Baseline width for the lines. Defaults to 1.5.
    fontsize (int, optional): Font size for the labels and annotations. Defaults to 14.
    figsize (tuple, optional): Size of the figure in inches. Defaults to (11, 6).
    save_fig_filename (str, optional): Filename to save the figure. If None, the figure is not saved. Defaults to None.
    """
    ### verify that all inputs take supported values
    assert ideal_direction in ['top','bottom']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max','min']
    assert color_by_continuous is None or color_by_categorical is None
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        #axis_labels = columns_axes ## <--- column_axes is not defined? Ask about this (Axis labels have to be defined)
        axis_labels = columns_axes

    ### create figure
    fig,ax = plt.subplots(1,1,figsize=figsize, gridspec_kw={'hspace':0.1, 'wspace':0.1})

    ### reorganize & normalize objective data
    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    ### apply any brushing criteria
    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        ### iteratively apply all brushing criteria to get satisficing set of solutions
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] >= threshold)

            ### add rectangle patch to plot to represent brushing
            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            if ideal_direction == 'top' and minmaxs[col_idx] == 'max':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)
            elif ideal_direction == 'top' and minmaxs[col_idx] == 'min':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
            if ideal_direction == 'bottom' and minmaxs[col_idx] == 'max':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
            elif ideal_direction == 'bottom' and minmaxs[col_idx] == 'min':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)

            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)

    ### loop over all solutions/rows & plot on parallel axis plot
    for i in range(objs_reorg.shape[0]):
        if color_by_continuous is not None:
            color = get_color(objs_reorg[columns_axes[color_by_continuous]].iloc[i],
                              color_by_continuous, color_palette_continuous,
                              color_by_categorical, color_dict_categorical)
        elif color_by_categorical is not None:
            color = get_color(objs[color_by_categorical].iloc[i],
                              color_by_continuous, color_palette_continuous,
                              color_by_categorical, color_dict_categorical)

        ### order lines according to ascending or descending values of one of the objectives?
        if zorder_by is None:
            zorder = 4
        else:
            zorder = get_zorder(objs_reorg[columns_axes[zorder_by]].iloc[i],
                                zorder_num_classes, zorder_direction)

        ### apply any brushing?
        if brushing_dict is not None:
            if satisfice.iloc[i]:
                alpha = alpha_base
                lw = lw_base
            else:
                alpha = alpha_brush
                lw = 1
                zorder = 2
        else:
            alpha = alpha_base
            lw = lw_base

        ## loop over objective/column pairs & plot lines between parallel axes
        for j in range(objs_reorg.shape[1]-1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j+1]]
            x = [j, j+1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    ### add top/bottom ranges
    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j])), [j, 1.02], ha='center', va='bottom',
                    zorder=5, fontsize=fontsize)
        if j == len(columns_axes)-1:
            ax.annotate(str(round(bottoms[j])) + '+', [j, -0.02], ha='center', va='top',
                        zorder=5, fontsize=fontsize)
        else:
            ax.annotate(str(round(bottoms[j])), [j, -0.02], ha='center', va='top',
                        zorder=5, fontsize=fontsize)

        ax.plot([j,j], [0,1], c='k', zorder=1)

    ### other aesthetics
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_visible(False)

    if ideal_direction == 'top':
        ax.arrow(-0.15,0.1,0,0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(-0.15,0.9,0,-0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3,0.5), ha='center', va='center',
                rotation=90, fontsize=fontsize)

    #ax.set_xlim(-0.4, 4.2) # REASON WHY ONLY 5 VARS
    ax.set_xlim(-0.4, len(columns_axes) - 0.6)
    ax.set_ylim(-0.4,1.1)

    for i,l in enumerate(axis_labels):
        ax.annotate(l, xy=(i,-0.12), ha='center', va='top', fontsize=fontsize)
    ax.patch.set_alpha(0)


    ### colorbar for continuous legend
    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(),
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4,
                          label=axis_labels[color_by_continuous], pad=0.03,
                          alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous,
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)
        ### categorical legend
    elif color_by_categorical is not None:
        leg = []
        for label,color in color_dict_categorical.items():
            leg.append(Line2D([0], [0], color=color, lw=3,
                              alpha=alpha_base, label=label))
        _ = ax.legend(handles=leg, loc='lower center',
                      ncol=max(3, len(color_dict_categorical)),
                      bbox_to_anchor=[0.5,-0.07], frameon=False, fontsize=fontsize)

    ### save figure
    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', transparent=True, dpi=600)


def custom_parallel_coordinates_highlight_cluster(objs, columns_axes=None, axis_labels=None, ideal_direction='top', minmaxs=None, color_by_continuous=None, color_palette_continuous=None, color_by_categorical=None, color_palette_categorical=None, colorbar_ticks_continuous=None, color_dict_categorical=None, zorder_by=None, zorder_num_classes=10, zorder_direction='ascending', alpha_base=0.8, brushing_dict=None, alpha_brush=0.05, lw_base=1.5, fontsize=14, figsize=(11,6), save_fig_filename=None, cluster_column_name='Cluster', title=None, highlight_indices=None, highlight_colors=None):
    assert ideal_direction in ['top','bottom']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max','min']
    assert color_by_continuous is None or color_by_categorical is None
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] >= threshold)

            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)

    # Ensure baseline and highlight colors and labels
    baseline_present = 0 in objs.index
    highlight_labels = [f"median {i+1}" for i in range(len(highlight_indices))]
    highlight_colors = highlight_colors

    if baseline_present:
        highlight_indices = [0] + highlight_indices
        highlight_labels = ["baseline"] + highlight_labels
        highlight_colors = ["black"] + highlight_colors

    for i in range(objs_reorg.shape[0]):
        idx_value = objs.index[i]
        if idx_value == 0 and baseline_present:
            color = "black"
            zorder = 20  # Bring to the very front
            lw = 4  # Make line wider
            label = "baseline"
        elif idx_value in highlight_indices:
            color = highlight_colors[highlight_indices.index(idx_value)]
            zorder = 15  # Bring to the front
            lw = 4  # Make line wider
            label = highlight_labels[highlight_indices.index(idx_value)]
        elif color_by_categorical is not None and cluster_column_name in objs.columns:
            cluster_value = objs[cluster_column_name].iloc[i]
            color = color_dict_categorical.get(cluster_value, 'grey')  # Use color_dict_categorical for color
            zorder = 4
            lw = lw_base
            label = None
        else:
            color = color_dict_categorical[1]  # Fallback color if no cluster info
            zorder = 4
            lw = lw_base
            label = None

        alpha = alpha_base

        for j in range(objs_reorg.shape[1] - 1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j])), [j, 1.02], ha='center', va='bottom', zorder=5, fontsize=fontsize)
        ax.annotate(str(round(bottoms[j])), [j, -0.02], ha='center', va='top', zorder=5, fontsize=fontsize)
        ax.plot([j, j], [0, 1], c='k', zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    if ideal_direction == 'top':
        ax.arrow(-0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(-0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3, 0.5), ha='center', va='center', rotation=90, fontsize=fontsize)

    ax.set_xlim(-0.4, len(columns_axes) - 0.6)
    ax.set_ylim(-0.4, 1.1)

    for i, l in enumerate(axis_labels):
        ax.annotate(l, xy=(i, -0.12), ha='center', va='top', fontsize=fontsize)

    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(),
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4,
                          label=axis_labels[color_by_continuous], pad=0.03,
                          alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous,
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)
    elif color_by_categorical is not None or highlight_indices is not None:
        leg = []
        if color_by_categorical is not None:
            for label, color in color_dict_categorical.items():
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))
        if highlight_indices is not None:
            for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))

        _ = ax.legend(handles=leg, loc='lower center', ncol=max(3, len(color_dict_categorical)),
                      bbox_to_anchor=[0.5, -0.07], frameon=False, fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', transparent=True, dpi=600)

median_colors = ["#DC143C", "#FF8C00", "#4169E1"]
#median_colors = ["green", "pink", "purple"]
color_dict_categorical = {1: 'firebrick', 2: 'goldenrod', 3: 'cornflowerblue'}
color_dict_categorical_1 = {1: 'firebrick'}
color_dict_categorical_2 = {1: 'goldenrod'}
color_dict_categorical_3 = {1: 'cornflowerblue'}


def custom_parallel_coordinates_highlight_iqr(objs, columns_axes=None, axis_labels=None, ideal_direction='top', minmaxs=None,
                                              color_by_continuous=None, color_palette_continuous=None, color_by_categorical=None,
                                              color_palette_categorical=None, colorbar_ticks_continuous=None,
                                              color_dict_categorical=None, zorder_by=None, zorder_num_classes=10,
                                              zorder_direction='ascending', alpha_base=0.8, brushing_dict=None,
                                              alpha_brush=0.05, lw_base=1.5, fontsize=14, figsize=(11,6),
                                              save_fig_filename=None, cluster_column_name='Cluster', title=None,
                                              highlight_indices=None, highlight_colors=None,
                                              filter_indices=None, iqr_data=None):
    assert ideal_direction in ['top','bottom']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max','min']
    assert color_by_continuous is None or color_by_categorical is None
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    # Filter the DataFrame by the provided indices
    if filter_indices is not None:
        objs = objs.loc[filter_indices]
        if iqr_data is not None:
            iqr_data = iqr_data.loc[filter_indices]

    fig, ax = plt.subplots(1, 1, figsize=figsize, gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] >= threshold)

            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)

    # Highlight IQR shading with cluster colors, except for the last column
    for i, idx in enumerate(objs.index):
        cluster_value = objs[cluster_column_name].loc[idx]
        color = color_dict_categorical.get(cluster_value, 'lightgrey')  # Default to light grey if cluster not found

        for j, col in enumerate(columns_axes[:-1]):  # Exclude the last column (Cluster)
            iqr_bottom = objs[col].iloc[i] - (iqr_data[col].iloc[i] / 2)
            iqr_top = objs[col].iloc[i] + (iqr_data[col].iloc[i] / 2)

            # Normalize the IQR bounds
            iqr_bottom_norm = (iqr_bottom - bottoms[j]) / (tops[j] - bottoms[j])
            iqr_top_norm = (iqr_top - bottoms[j]) / (tops[j] - bottoms[j])

            rect = Rectangle([j - 0.05, iqr_bottom_norm], 0.1, iqr_top_norm - iqr_bottom_norm)
            pc = PatchCollection([rect], facecolor=color, alpha=0.3, zorder=2)
            ax.add_collection(pc)

    # Ensure baseline and highlight colors and labels
    baseline_present = 0 in objs.index
    highlight_labels = [f"median {i+1}" for i in range(len(highlight_indices))]
    highlight_colors = highlight_colors

    if baseline_present:
        highlight_indices = [0] + highlight_indices
        highlight_labels = ["baseline"] + highlight_labels
        highlight_colors = ["black"] + highlight_colors

    for i in range(objs_reorg.shape[0]):
        idx_value = objs.index[i]
        if idx_value == 0 and baseline_present:
            color = "black"
            zorder = 20  # Bring to the very front
            lw = 4  # Make line wider
            label = "baseline"
        elif idx_value in highlight_indices:
            color = highlight_colors[highlight_indices.index(idx_value)]
            zorder = 15  # Bring to the front
            lw = 4  # Make line wider
            label = highlight_labels[highlight_indices.index(idx_value)]
        elif color_by_categorical is not None and cluster_column_name in objs.columns:
            cluster_value = objs[cluster_column_name].iloc[i]
            color = color_dict_categorical.get(cluster_value, 'grey')  # Use color_dict_categorical for color
            zorder = 4
            lw = lw_base
            label = None
        else:
            color = color_dict_categorical[1]  # Fallback color if no cluster info
            zorder = 4
            lw = lw_base
            label = None

        alpha = alpha_base

        for j in range(objs_reorg.shape[1] - 1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j])), [j, 1.02], ha='center', va='bottom', zorder=5, fontsize=fontsize)
        ax.annotate(str(round(bottoms[j])), [j, -0.02], ha='center', va='top', zorder=5, fontsize=fontsize)
        ax.plot([j, j], [0, 1], c='k', zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    if ideal_direction == 'top':
        ax.arrow(-0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(-0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3, 0.5), ha='center', va='center', rotation=90, fontsize=fontsize)

    ax.set_xlim(-0.4, len(columns_axes) - 0.6)
    ax.set_ylim(-0.4, 1.1)

    for i, l in enumerate(axis_labels):
        ax.annotate(l, xy=(i, -0.12), ha='center', va='top', fontsize=fontsize)

    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(),
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4,
                          label=axis_labels[color_by_continuous], pad=0.03,
                          alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous,
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)
    elif color_by_categorical is not None or highlight_indices is not None:
        leg = []
        if color_by_categorical is not None:
            for label, color in color_dict_categorical.items():
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))
        if highlight_indices is not None:
            for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))

        _ = ax.legend(handles=leg, loc='lower center', ncol=max(3, len(color_dict_categorical)),
                      bbox_to_anchor=[0.5, -0.07], frameon=False, fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', transparent=True, dpi=600)

"""highlight_indices = [0, 4, 15, 261, 320, 360]
highlight_descriptions = [
    "Business As Usual",
    "Higher Flows",
    "A Saltier Delta",
    "New Balance",
    "Carry It Forward",
    "Prioritized Drinking Water"
]

highlight_colors = ["black", "red", "blue", "green", "orange", "purple"]"""

def process_scenario_dataframe(df):
    # Mapping of the columns we are interested in
    column_mapping = {
        'NDO': 'NDO',
        'X2_APR': 'X2_APR',
        'X2_OCT': 'X2_OCT',
        'SAC_IN': 'SAC_IN',
        'SJR_IN': 'SJR_IN',
        'ES_YBP_IN': 'ES_YBP_IN',
        'TOTAL_DELTA_IN': 'TOTAL_DELTA_IN',
        'CVP_SWP_EXPORTS': 'CVP_SWP_EXPORTS',
        'OTHER_EXPORTS': 'OTHER_EXPORTS',
        'ADJ_CVP_SWP_EXPORTS': 'ADJ_CVP_SWP_EXPORTS',
        'DEL_NOD_TOTAL': 'DEL_NOD_TOTAL',
        'DEL_NOD_AG_TOTAL': 'DEL_NOD_AG_TOTAL',
        'DEL_NOD_MI_TOTAL': 'DEL_NOD_MI_TOTAL',
        'DEL_SJV_AG_TOTAL': 'DEL_SJV_AG_TOTAL',
        'DEL_SJV_MI_TOTAL': 'DEL_SJV_MI_TOTAL',
        'DEL_SJV_TOTAL': 'DEL_SJV_TOTAL',
        'DEL_SOCAL_MI_TOTAL': 'DEL_SOCAL_MI_TOTAL',
        'DEL_CCOAST_MI_TOTAL': 'DEL_CCOAST_MI_TOTAL',
        'STO_NOD_TOTAL_APR': 'STO_NOD_TOTAL_APR',
        'STO_NOD_TOTAL_OCT': 'STO_NOD_TOTAL_OCT',
        'STO_SOD_TOTAL_APR': 'STO_SOD_TOTAL_APR',
        'STO_SOD_TOTAL_OCT': 'STO_SOD_TOTAL_OCT'
    }

    # Select relevant columns based on the second level of MultiIndex
    selected_columns = [col for col in df.columns if col[1] in column_mapping]
    selected_df = df[selected_columns]

    # Rename columns according to the mapping
    new_columns = [column_mapping[col[1]] for col in selected_df.columns]
    selected_df.columns = new_columns

    # Reorder and select only the required columns
    desired_order = ['DEL_NOD_AG_TOTAL', 'DEL_SJV_AG_TOTAL', 'DEL_NOD_MI_TOTAL', 'DEL_SJV_MI_TOTAL',
                     'DEL_SOCAL_MI_TOTAL', 'CVP_SWP_EXPORTS', 'NDO', 'SAC_IN', 'SJR_IN', 'X2_APR',
                     'X2_OCT', 'STO_NOD_TOTAL_OCT', 'STO_SOD_TOTAL_OCT']

    ordered_df = selected_df[desired_order]
    ordered_df.columns = [
        "Sac Valley AG Deliveries", "SJ Valley AG Deliveries", "Sac Valley Municipal Deliveries",
        "SJ Valley Municipal Deliveries", "SoCal Municipal Deliveries", "Delta Exports",
        "Delta Outflows", "Sac River Inflows", "SJ River Inflows", "X2 Salinity (Apr)",
        "X2 Salinity (Oct)", "North of Delta Storage (Sep)", "South of Delta Storage (Sep)"
    ]

    return ordered_df

def calculate_statistics(dataframes, scenario_names):
    medians = []
    std_devs = []
    percentiles_90 = []
    percentiles_10 = []

    for df, name in zip(dataframes, scenario_names):
        processed_df = process_scenario_dataframe(df)

        # Calculate median, standard deviation, 90th percentile, and 10th percentile
        median_values = processed_df.median()
        std_dev_values = processed_df.std()
        percentile_90_values = processed_df.quantile(0.90)
        percentile_10_values = processed_df.quantile(0.10)

        # Assign names to Series
        median_values.name = name
        std_dev_values.name = name
        percentile_90_values.name = name
        percentile_10_values.name = name

        # Append results to lists
        medians.append(median_values)
        std_devs.append(std_dev_values)
        percentiles_90.append(percentile_90_values)
        percentiles_10.append(percentile_10_values)

    # Combine all medians, standard deviations, and percentiles into DataFrames
    median_df = pd.DataFrame(medians)
    std_dev_df = pd.DataFrame(std_devs)
    percentile_90_df = pd.DataFrame(percentiles_90)
    percentile_10_df = pd.DataFrame(percentiles_10)

    return median_df, std_dev_df, percentile_90_df, percentile_10_df


def custom_parallel_coordinates_highlight_scenarios(objs, columns_axes=None, axis_labels=None, ideal_direction='top',
                                                    minmaxs=None, color_by_continuous=None, color_palette_continuous=None,
                                                    color_by_categorical=None, color_palette_categorical=None,
                                                    colorbar_ticks_continuous=None, color_dict_categorical=None,
                                                    zorder_by=None, zorder_num_classes=10, zorder_direction='ascending',
                                                    alpha_base=0.8, brushing_dict=None, alpha_brush=0.05, lw_base=1.5,
                                                    fontsize=14, figsize=(22,10), save_fig_filename=None,
                                                    cluster_column_name='Cluster', title=None, highlight_indices=None,
                                                    highlight_colors=None, highlight_descriptions=None):
    assert ideal_direction in ['top','bottom']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max','min']
    assert color_by_continuous is None or color_by_categorical is None
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    # Plot all scenarios in light grey
    for i in range(objs_reorg.shape[0]):
        if objs.index[i] not in highlight_indices:
            for j in range(objs_reorg.shape[1] - 1):
                y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
                x = [j, j + 1]
                ax.plot(x, y, c='lightgrey', alpha=0.5, zorder=2, lw=0.5)

    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] >= threshold)

            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)

    # Ensure highlight colors and labels
    if highlight_indices is not None:
        highlight_labels = highlight_descriptions if highlight_descriptions else [f"Scenario {i+1}" for i in range(len(highlight_indices))]
        for i in range(objs_reorg.shape[0]):
            idx_value = objs.index[i]
            if idx_value in highlight_indices:
                color = highlight_colors[highlight_indices.index(idx_value)]
                zorder = 15  # Bring to the front
                lw = 3  # Make line wider
                label = highlight_labels[highlight_indices.index(idx_value)]

                for j in range(objs_reorg.shape[1] - 1):
                    y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
                    x = [j, j + 1]
                    ax.plot(x, y, c=color, alpha=alpha_base, zorder=zorder, lw=lw, label=label if j == 0 else "")

    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j])), [j, 1.02], ha='center', va='bottom', zorder=5, fontsize=fontsize, color='black')
        ax.annotate(str(round(bottoms[j])), [j, -0.02], ha='center', va='top', zorder=5, fontsize=fontsize, color='black')
        ax.plot([j, j], [0, 1], c='black', alpha=0.3, zorder=1)

    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='center', va='top', fontsize=fontsize)
    ax.tick_params(axis='x', colors='black', pad=10)
    ax.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(-0.4, len(columns_axes) - 0.6)
    ax.set_ylim(-0.1, 1.1)

    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(),
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4,
                          label=axis_labels[color_by_continuous], pad=0.03,
                          alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous,
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, color='black', pad=20)

    # Calculate the space needed for x-axis labels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])

    # Adjust the bottom of the plot to make room for labels and legend
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.15  # Estimated height of the legend as a fraction of figure height

    # Adjust subplot parameters
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    # Create legend
    leg = []
    if highlight_indices is not None:
        for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
            leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))

    # Add legend below the x-axis labels
    if leg:
        leg = ax.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                        ncol=len(highlight_indices), frameon=False, fontsize=fontsize)
        for text in leg.get_texts():
            text.set_color('black')

    plt.tight_layout()

    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=600, facecolor='white')

    return fig, ax

def custom_parallel_coordinates_highlight_variability(objs, variability_data, columns_axes=None, axis_labels=None,
                                                      alpha_base=0.8, alpha_shade=0.2, lw_base=1.5,
                                                      fontsize=14, figsize=(22, 10), save_fig_filename=None,
                                                      title=None, highlight_indices=None,
                                                      highlight_colors=None, highlight_descriptions=None):
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Calculate dynamic y-axis limits for each column
    y_mins = objs - variability_data
    y_maxs = objs + variability_data
    bottoms = y_mins.min()
    tops = y_maxs.max()

    # Normalize data
    objs_norm = (objs - bottoms) / (tops - bottoms)
    var_norm = variability_data / (tops - bottoms)

    # Plot highlighted scenarios with shading
    if highlight_indices is not None:
        highlight_labels = highlight_descriptions if highlight_descriptions else [f"Scenario {i+1}" for i in range(len(highlight_indices))]
        for i, idx in enumerate(highlight_indices):
            if i >= len(highlight_colors) or i >= len(highlight_labels):
                break
            color = highlight_colors[i]
            label = highlight_labels[i]

            y = objs_norm.loc[idx]
            var = var_norm.loc[idx]

            # Plot the main line
            ax.plot(range(len(columns_axes)), y, c=color, alpha=alpha_base, zorder=15, lw=2, label=label)

            # Add shading for variability without dividing the standard deviation by half
            for j in range(len(columns_axes) - 1):
                y_lower = [y.iloc[j] - var.iloc[j], y.iloc[j + 1] - var.iloc[j + 1]]  # 1 SD below
                y_upper = [y.iloc[j] + var.iloc[j], y.iloc[j + 1] + var.iloc[j + 1]]  # 1 SD above
                x = [j, j + 1]
                ax.fill_between(x, y_lower, y_upper, color=color, alpha=alpha_shade, zorder=10)

    # Set up axes
    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', va='top', fontsize=fontsize)
    ax.tick_params(axis='x', pad=10)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove y-axis ticks and labels
    ax.yaxis.set_visible(False)

    # Add vertical lines for each metric
    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='grey', linestyle=':', alpha=0.5, zorder=1, ymin=-0.1, ymax=1.1)

    # Annotate min and max values for each column at the top and bottom of vertical lines
    for j, (col, bot, top) in enumerate(zip(columns_axes, bottoms, tops)):
        ax.annotate(f'{bot:.0f}', (j, -0.1), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=fontsize-2, color='black')
        ax.annotate(f'{top:.0f}', (j, 1.1), xytext=(0, -5),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=fontsize-2, color='black')

    # Add title
    if title:
        ax.set_title(title, fontsize=fontsize+2, pad=20)

    # Add legend below the plot
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.05  # Estimated height of the legend as a fraction of figure height

    # Adjust subplot parameters
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    # Add legend below the x-axis labels
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                    ncol=len(highlight_indices), frameon=False, fontsize=fontsize)

    plt.tight_layout()

    if save_fig_filename:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=600, facecolor='white')

    return fig, ax


def custom_parallel_coordinates_highlight_quantile(objs, lower_bound_data, upper_bound_data, columns_axes=None, axis_labels=None,
                                                      alpha_base=0.8, alpha_shade=0.2, lw_base=1.5,
                                                      fontsize=14, figsize=(22, 10), save_fig_filename=None,
                                                      title=None, highlight_indices=None,
                                                      highlight_colors=None, highlight_descriptions=None):
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Calculate dynamic y-axis limits for each column
    y_mins = lower_bound_data
    y_maxs = upper_bound_data
    bottoms = y_mins.min()
    tops = y_maxs.max()

    # Normalize data
    objs_norm = (objs - bottoms) / (tops - bottoms)
    lower_norm = (lower_bound_data - bottoms) / (tops - bottoms)
    upper_norm = (upper_bound_data - bottoms) / (tops - bottoms)

    # Plot highlighted scenarios with shading
    if highlight_indices is not None:
        highlight_labels = highlight_descriptions if highlight_descriptions else [f"Scenario {i+1}" for i in range(len(highlight_indices))]
        for i, idx in enumerate(highlight_indices):
            if i >= len(highlight_colors) or i >= len(highlight_labels):
                break
            color = highlight_colors[i]
            label = highlight_labels[i]

            y = objs_norm.loc[idx]
            y_lower = lower_norm.loc[idx]
            y_upper = upper_norm.loc[idx]

            # Plot the main line
            ax.plot(range(len(columns_axes)), y, c=color, alpha=alpha_base, zorder=15, lw=2, label=label)

            # Add shading between the 10th and 90th percentile
            for j in range(len(columns_axes) - 1):
                y_lower_values = [y_lower.iloc[j], y_lower.iloc[j + 1]]  # 10th percentile
                y_upper_values = [y_upper.iloc[j], y_upper.iloc[j + 1]]  # 90th percentile
                x = [j, j + 1]
                ax.fill_between(x, y_lower_values, y_upper_values, color=color, alpha=alpha_shade, zorder=10)

    # Set up axes
    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', va='top', fontsize=fontsize)
    ax.tick_params(axis='x', pad=10)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove y-axis ticks and labels
    ax.yaxis.set_visible(False)

    # Add vertical lines for each metric
    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='grey', linestyle=':', alpha=0.5, zorder=1, ymin=-0.1, ymax=1.1)

    # Annotate min and max values for each column at the top and bottom of vertical lines
    for j, (col, bot, top) in enumerate(zip(columns_axes, bottoms, tops)):
        ax.annotate(f'{bot:.0f}', (j, -0.1), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=fontsize-2, color='black')
        ax.annotate(f'{top:.0f}', (j, 1.1), xytext=(0, -5),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=fontsize-2, color='black')

    # Add title
    if title:
        ax.set_title(title, fontsize=fontsize+2, pad=20)

    # Add legend below the plot
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.05  # Estimated height of the legend as a fraction of figure height

    # Adjust subplot parameters
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    # Add legend below the x-axis labels
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                    ncol=len(highlight_indices), frameon=False, fontsize=fontsize)

    plt.tight_layout()

    if save_fig_filename:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=600, facecolor='white')

    return fig, ax


"""BASELINE AT ZERO"""
def custom_parallel_coordinates_highlight_scenarios_baseline_at_zero(objs, columns_axes=None, axis_labels=None,
                                                                     color_dict_categorical=None,
                                                                     alpha_base=0.8, lw_base=1.5,
                                                                     fontsize=14, figsize=(22,8),
                                                                     save_fig_filename=None,
                                                                     title=None, highlight_indices=None,
                                                                     highlight_colors=None, highlight_descriptions=None):
    """
    Plots parallel coordinates for relative (% difference) data, typically:
      - Row 0 => baseline scenario (always 0%)
      - Row 1 => alternative scenario (± some % from baseline).

    Main change: we automatically set the y-axis to [global_min, global_max]
    rather than a fixed ±20% or symmetrical range.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    global_min = objs[columns_axes].min().min()
    global_max = objs[columns_axes].max().max()

    for idx in objs.index:
        if highlight_indices is not None and idx not in highlight_indices:
            ax.plot(range(len(columns_axes)), objs.loc[idx, columns_axes],
                    c='grey', alpha=0.1, zorder=5, lw=0.5)

    if highlight_indices is not None:
        highlight_labels = (highlight_descriptions if highlight_descriptions
                            else [f"Scenario {i+1}" for i in range(len(highlight_indices))])

        for i, idx in enumerate(highlight_indices):
            color = highlight_colors[i]
            label = highlight_labels[i]
            ax.plot(range(len(columns_axes)), objs.loc[idx, columns_axes],
                    c=color, alpha=alpha_base, zorder=15, lw=3, label=label)

    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    ax.set_ylim(global_min, global_max)

    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', fontsize=fontsize)

    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis='y', colors='black', labelsize=fontsize)

    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3, zorder=1)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('black')

    ax.yaxis.grid(True, linestyle=':', alpha=0.3, color='gray')

    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, color='black', pad=20)
    ax.set_ylabel('Percentage Change from Baseline', fontsize=fontsize, color='black')

    plt.tight_layout()


    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])

    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.15
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    if highlight_indices is not None:
        leg = ax.legend(loc='upper center',
                        bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                        ncol=len(highlight_indices), frameon=False, fontsize=fontsize)

    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=600, facecolor='white')

    return fig, ax

def custom_parallel_coordinates_relative_with_baseline_values(
        objs_rel,
        baseline_abs,
        axis_label_map,
        columns_axes=None,
        axis_labels=None,
        alpha_base=0.8,
        lw_base=1.5,
        fontsize=14,
        figsize=(22,8),
        save_fig_filename=None,
        title=None,
        highlight_indices=None,
        highlight_colors=None,
        highlight_descriptions=None
):
    """
    Similar to your original function, but:
      1) We look up the units from axis_label_map for each axis/column
      2) We append "(Baseline)" to each numeric label
      3) We use the updated logic to display baseline values below the axis.

    Parameters
    ----------
    objs_rel : pd.DataFrame
        Typically 2 rows => [baseline(0%), alternative(±%)], columns = variables.
    baseline_abs : pd.DataFrame or pd.Series
        1 row or Series, same columns, with the absolute baseline means.
    axis_label_map : dict
        Your dictionary mapping axis label => {calsim_vars, units, months}.
        We can use it to fetch the 'units' for each axis label.
    columns_axes, axis_labels : lists, optional
        Same usage as before.
    ...
    """
    def get_units_for_axis_label(label):
        # If label not found, fallback to ""
        if label in axis_label_map:
            return axis_label_map[label]['units']
        return ""

    # 1) Decide which columns to plot
    if columns_axes is None:
        columns_axes = objs_rel.columns
    if axis_labels is None:
        axis_labels = columns_axes

    # 2) If baseline_abs is a 1-row DataFrame, convert to Series
    if isinstance(baseline_abs, pd.DataFrame):
        if baseline_abs.shape[0] == 1:
            baseline_abs = baseline_abs.iloc[0]
        else:
            raise ValueError("baseline_abs DataFrame must have exactly 1 row")

    # 3) Setup figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # 4) Min/Max from the relative data
    global_min = objs_rel[columns_axes].min().min()
    global_max = objs_rel[columns_axes].max().max()

    # 5) Plot non-highlighted rows in grey
    for idx in objs_rel.index:
        if highlight_indices is not None and idx not in highlight_indices:
            ax.plot(range(len(columns_axes)), objs_rel.loc[idx, columns_axes],
                    c='grey', alpha=0.1, lw=0.5, zorder=5)

    # 6) Plot highlighted rows
    if highlight_indices is not None:
        if highlight_descriptions is None:
            highlight_descriptions = highlight_indices
        for i, idx in enumerate(highlight_indices):
            color = highlight_colors[i]
            label = highlight_descriptions[i]
            ax.plot(
                range(len(columns_axes)),
                objs_rel.loc[idx, columns_axes],
                c=color,
                alpha=alpha_base,
                lw=3,
                label=label,
                zorder=10
            )

    # 7) Axis scaling
    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    ax.set_ylim(global_min, global_max)
    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', fontsize=fontsize)

    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis='y', labelsize=fontsize)

    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('black')

    ax.yaxis.grid(True, linestyle=':', alpha=0.3, color='gray')

    if title:
        ax.set_title(title, fontsize=fontsize+2, color='black', pad=20)
    ax.set_ylabel('Percentage Change from Baseline', fontsize=fontsize)

    # 8) Adjust layout for x-tick labels & legend
    plt.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max(label.get_window_extent(renderer).height for label in ax.get_xticklabels())
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.15
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    if highlight_indices is not None:
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
            ncol=len(highlight_indices),
            frameon=False,
            fontsize=fontsize
        )

    # 9) Annotate baseline absolute values + units
    margin = 0.05 * (global_max - global_min)
    label_y = global_min - margin

    for i, col in enumerate(columns_axes):
        base_val = baseline_abs[col]
        if pd.isna(base_val):
            txt_label = "NaN"
        else:
            units_str = get_units_for_axis_label(col)
            txt_label = f"{base_val:,.0f} {units_str} (Baseline)"

        ax.text(
            i, label_y, txt_label,
            ha='center', va='top',
            fontsize=fontsize*0.8,
            rotation=45,
            color='black'
        )

    if save_fig_filename:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=600, facecolor='white')

    return fig, ax