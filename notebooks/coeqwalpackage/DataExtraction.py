"""IMPORTS"""
import os
import datetime as dt
import pandas as pd
import numpy as np

# Import custom modules - NEED WINDOWS OS
import AuxFunctions as af, cs3, csPlots, cs_util as util, dss3_functions_reference as dss
import csPlots as csplt
import cqwlutils as cu

"""READ NAMES & CONSTRUCT STUDY VARIABLES"""
def read_names(DssList, DssTab, DssMin, DssMax, hdr=True):
    dsshdr, dssname = cu.read_from_excel(DssList, DssTab, DssMin, DssMax, hdr=True)
    dss_names = []
    for i in range(len(dssname)):
        dss_names.append(dssname[i][0] + DssExt)
    return dss_names

def abbrev_names(DssList, DssTab, AbbrMin, AbbrMax, hdr=True):
    abbrhdr, abbrname = cu.read_from_excel(DssList, DssTab, AbbrMin, AbbrMax, hdr=True)
    abbr_names = []
    for i in range(len(abbrname)):
        abbr_names.append(abbrname[i][0])
    return abbr_names

def dss_names_to_csv(dss_names):
    dss_df = (pd.DataFrame(dss_names))
    return dss_df.to_csv(os.path.join(DataDir, DssOut))

"""READ VARIABLES"""
def get_var_df(VarList, VarTab,VarMin,VarMax,hdr=True):
    hdr, vars = cu.read_from_excel(VarList, VarTab,VarMin,VarMax,hdr=True)
    var_df = pd.DataFrame(data=vars, columns=hdr)
    return var_df

def var_df_to_csv(var_df, DataDir,VarOut):
    var_df.to_csv(os.path.join(DataDir,VarOut))
    return VarOut

"""
HELPER FUNCTIONS
"""
def add_sum_column(df, new_col, required_cols, verbose=True):
    """
    Safely add a new column to df by summing over required_cols,
    but only if *all* required columns exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns
    new_col : tuple
        Target column key (MultiIndex tuple)
    required_cols : list of tuples
        List of column MultiIndex tuples to sum
    verbose : bool
        Whether to print missing/added info

    Returns
    -------
    pd.DataFrame
        DataFrame with new column added (if all inputs exist)
    """

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        if verbose:
            print(f"⚠️ Skipping {new_col}: missing {len(missing)} columns")
            for m in missing:
                print(f"   - {m}")
        return df  # no modification
    else:
        df[new_col] = df.loc[:, required_cols].sum(axis=1)
        if verbose:
            print(f"✅ Added column {new_col} from {len(required_cols)} inputs")
        return df


def add_combined_column_if_exists(
    df,
    target_col,
    add_cols,
    sub_cols=None,
    multiplier=1.0,
    divisor_col=None,
    record_used_cols=None,
    verbose=True
):
    # Track all components referenced
    used = []

    # Check for missing columns
    required_cols = list(add_cols)
    if sub_cols:
        required_cols += list(sub_cols)
    if divisor_col:
        required_cols.append(divisor_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️ Skipping {target_col}: missing columns:")
        for col in missing:
            print(f"  - {col}")
        return df

    # If present, record used columns
    for col in required_cols:
        if col in df.columns:
            used.append(col)

    if record_used_cols is not None:
        record_used_cols.extend(used)

    # Helper to reduce DataFrame to Series if necessary
    def reduce_df(x):
        if isinstance(x, pd.DataFrame):
            if x.shape[1] == 1:
                return x.iloc[:, 0]
            elif x.shape[1] == 2:
                return x.iloc[:, 0] + x.iloc[:, 1]
            else:
                return (x.iloc[:, 0] + x.iloc[:, 1]) / x.iloc[:, 2]
        return x

    # Compute numerator
    numerator = sum(reduce_df(df[c]) for c in add_cols)
    if sub_cols:
        numerator -= sum(reduce_df(df[c]) for c in sub_cols)

    # Apply divisor
    if divisor_col:
        divisor = reduce_df(df[divisor_col])
        numerator = numerator / divisor

    # Apply multiplier
    df[target_col] = numerator * multiplier

    if verbose:
        print(f"✅ Added column {target_col} from {len(required_cols)} inputs")

    return df

# ----------------------------------------------------------------------
# Helper for:  (A+B)/X  +  (C+D)/Y
# ----------------------------------------------------------------------

def add_two_term_ratio_if_exists(
    df,
    target_col,
    term1_num_cols, term1_den_col,
    term2_num_cols, term2_den_col,
    record_used_cols=None,
    verbose=True
):
    used = []

    required_cols = term1_num_cols + [term1_den_col] + term2_num_cols + [term2_den_col]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"⚠️ Skipping {target_col}: missing columns:")
        for col in missing:
            print(f"  - {col}")
        return df

    for col in required_cols:
        if col in df.columns:
            used.append(col)

    if record_used_cols is not None:
        record_used_cols.extend(used)

    # --- Compute term 1 ---
    num1 = sum(df[c].squeeze() for c in term1_num_cols)
    den1 = df[term1_den_col].squeeze()
    term1 = num1 / den1

    # --- Compute term 2 ---
    num2 = sum(df[c].squeeze() for c in term2_num_cols)
    den2 = df[term2_den_col].squeeze()
    term2 = num2 / den2

    df[target_col] = term1 + term2
    
    if verbose:
        print(f"✅ Added column {target_col} from {len(required_cols)} inputs")

    return df

"""
CREATE DATASETS ACROSS STUDIES
(Aux functions to read and process studies (for single and multiple studies). Note: contain options to create hard-coded additional compound variables (if more are needed, add to these codes))
"""

def preprocess_study_dss(df, dss_name, datetime_start_date, datetime_end_date, addsl=True, addres = True, addpump = True, adddelcvp = True, adddelcvpag = True, addcvpscex = True, addcvpprf = True, adddelcvpswp = True, add_nod_storage = True, add_sod_storage = True, add_del_nod_ag = True, add_del_nod_mi = True, add_del_sod_mi = True, add_del_sod_ag = True, add_total_exports = True, add_del_swp_total = True, add_awoann_xa = True):
    dvar_list = []
    combined_df = pd.DataFrame()
    
    for i, r in df.iterrows():
        if r["Part C:"] == '':
            dvar_list.append(f'/{r["Part B:"]}/')
        else:
            dvar_list.append(f'/{r["Part B:"]}/{r["Part C:"]}/')

    # print('dvar_list:')
    # print(dvar_list)

    # Create a blank python "calsim" object
    thiscs3 = cs3.calsim()

    # add start and end dates
    print('Start: ')
    print(datetime_start_date)
    print('End: ')
    print(datetime_end_date)   
    thiscs3.StartDate = datetime_start_date
    thiscs3.EndDate = datetime_end_date

    # add path to DSS
    DSS_FP = dss_name
    thiscs3.DV_FP = DSS_FP

    # Retrieve the DSS data variables from the DSS file
    thiscs3.DVdata = cs3.csDVdata(thiscs3)
    thiscs3.DVdata.getDVts(filter=dvar_list)

    df = thiscs3.DVdata.DVtsDF.copy(deep=True)

    # create manual add column for missing constant top levels
    S_FOLSMLEVEL6_monthly_taf_values = 967
    df[('MANUAL-ADD','S_FOLSMLEVEL6DV','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_FOLSMLEVEL6_monthly_taf_values
    S_MLRTNLEVEL5_monthly_taf_values = 524
    df[('MANUAL-ADD','S_MLRTNLEVEL5DV','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_MLRTNLEVEL5_monthly_taf_values
    S_OROVLLEVEL6DV_monthly_taf_values = 3424.8
    df[('MANUAL-ADD','S_OROVLLEVEL6DV','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_OROVLLEVEL6DV_monthly_taf_values
    S_MELONLEVEL5DV_monthly_taf_values = 2420
    df[('MANUAL-ADD','S_MELONLEVEL5DV','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_MELONLEVEL5DV_monthly_taf_values
    
    # create aggregate variables using add_combined_column_if_exists

    if add_nod_storage:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'NOD_STORAGE', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
            add_cols=[
                ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_NBLDB', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')
            ]
        )
    
    if add_sod_storage:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'SOD_STORAGE', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
            add_cols=[
                ('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_NHGAN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_PEDRO', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_MCLRE', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_HNSLY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')
            ]
        )
    
    if add_del_swp_total:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_SWP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_SWP_PAG', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_SWP_PMI', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if add_del_nod_ag:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_NOD_AG', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_SWP_PAG_N', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if add_del_sod_ag:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_SOD_AG', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_SWP_PAG_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if add_del_sod_mi:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_SOD_MI', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_PMI_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_SWP_PMI_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if add_del_nod_mi:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_NOD_MI', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_PMI_N_WAMER', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_SWP_PMI_N', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if add_total_exports:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'TOTAL_EXPORTS', 'EXPORTS-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'C_DMC003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'C_CAA003_SWP', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'C_CAA003_CVP', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if addsl:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'S_SLTOT', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
            add_cols=[
                ('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')
            ]
        )
    
    if addpump:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'D_TOTAL', 'CHANNEL-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'C_DMC000', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'C_CAA003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if addres:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'S_RESTOT', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
            add_cols=[
                ('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')
            ]
        )
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'S_RESTOT_NOD', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
            add_cols=[
                ('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')
            ]
        )
    
    if adddelcvp:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_TOTAL_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_TOTAL_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if adddelcvpswp:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_CVPSWP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_SWP_PAG_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if adddelcvpag:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_CVP_PAG_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if addcvpscex:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_CVP_PSCEX_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if addcvpprf:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'DEL_CVP_PRF_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            add_cols=[
                ('CALSIM', 'DEL_CVP_PRF_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'DEL_CVP_PRF_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ]
        )
    
    if add_awoann_xa:
        add_combined_column_if_exists(
            df,
            target_col=('CALCULATED', 'AWOANN_ALL_DV', 'ANNUAL-APPLIED-WATER-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
            add_cols=[
                ('CALSIM', 'AWOANN_64_XADV', 'ANNUAL-APPLIED-WATER', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'AWOANN_72_XA1DV', 'ANNUAL-APPLIED-WATER', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'AWOANN_72_XA2DV', 'ANNUAL-APPLIED-WATER', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'AWOANN_72_XA3DV', 'ANNUAL-APPLIED-WATER', '1MON', 'L2020A', 'PER-AVER', 'TAF'),
                ('CALSIM', 'AWOANN_73_XADV', 'ANNUAL-APPLIED-WATER', '1MON', 'L2020A', 'PER-AVER', 'TAF')
            ]
        )

    
    new_columns = [(col[0], col[1], *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']

    # df.head(5)
    return df


def preprocess_compound_data_dss(df, ScenarioDir, dss_names, index_names, min_datetime, max_datetime, addsl=True, addres = True, addpump = True, adddelcvp = True, adddelcvpag = True, addcvpscex = True, addcvpprf = True, adddelcvpswp = True, add_nod_storage = True, add_sod_storage = True, add_del_nod_ag = True, add_del_nod_mi = True, add_del_sod_mi = True, add_del_sod_ag = True, add_total_exports = True, add_del_swp_total = True, add_awoann_xa = True):
    dvar_list = []
    combined_df = pd.DataFrame()
    
    for i, r in df.iterrows():
        if r["Part C:"] == '':
            dvar_list.append(f'/{r["Part B:"]}/')
        else:
            dvar_list.append(f'/{r["Part B:"]}/{r["Part C:"]}/')


    for i in range(len(dss_names)):
        #get DSS and scenario index name
        dss_name = dss_names[i]
        index_name = index_names[i]
        print(dss_name)
        print(index_name)

        # Create a blank python "calsim" object
        thiscs3 = cs3.calsim()

        # add start and end dates
        thiscs3.StartDate = min_datetime
        thiscs3.EndDate = max_datetime

        # add path to DSS
        DSS_FP = os.path.join(ScenarioDir, dss_name)
        thiscs3.DV_FP = DSS_FP

        # Retrieve the DSS data variables from the DSS file
        thiscs3.DVdata = cs3.csDVdata(thiscs3)
        thiscs3.DVdata.getDVts(filter=dvar_list)

        df = thiscs3.DVdata.DVtsDF.copy(deep=True)

        # create manual add column for missing constant top levels
        S_FOLSMLEVEL6_monthly_taf_values = 967
        df[('MANUAL-ADD','S_FOLSMLEVEL6DV','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_FOLSMLEVEL6_monthly_taf_values
        S_MLRTNLEVEL5_monthly_taf_values = 524
        df[('MANUAL-ADD','S_MLRTNLEVEL5DV','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_MLRTNLEVEL5_monthly_taf_values
        S_OROVLLEVEL6DV_monthly_taf_values = 3424.8
        df[('MANUAL-ADD','S_OROVLLEVEL6DV ','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_OROVLLEVEL6DV_monthly_taf_values
        S_MELONLEVEL5DV_monthly_taf_values = 2420
        df[('MANUAL-ADD','S_MELONLEVEL5DV ','STORAGE-ZONE','1MON','L2020A','PER-CUM','TAF')] = S_MELONLEVEL5DV_monthly_taf_values
        
        # create aggregate variables using add_combined_column_if_exists

        # --- NOD Storage ---
        if add_nod_storage:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','NOD_STORAGE','STORAGE-CALC','1MON','L2020A','PER-AVER','TAF'),
                add_cols=[
                    ('CALSIM','S_TRNTY','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_SHSTA','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_OROVL','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_FOLSM','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_NBLDB','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                ]
            )
        
        # --- SOD Storage ---
        if add_sod_storage:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','SOD_STORAGE','STORAGE-CALC','1MON','L2020A','PER-AVER','TAF'),
                add_cols=[
                    ('CALSIM','S_SLUIS_CVP','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_SLUIS_SWP','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_MELON','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_NHGAN','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_MLRTN','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_PEDRO','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_MCLRE','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_HNSLY','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                ]
            )
        
        # --- SWP Deliveries ---
        if add_del_swp_total:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_SWP_TOTAL','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_SWP_PAG','DELIVERY-SWP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_SWP_PMI','DELIVERY-SWP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- NOD AG Deliveries ---
        if add_del_nod_ag:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_NOD_AG','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_PAG_N','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_SWP_PAG_N','DELIVERY-SWP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_PSC_N','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- SOD AG Deliveries ---
        if add_del_sod_ag:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_SOD_AG','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_PAG_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_SWP_PAG_S','DELIVERY-SWP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_PEX_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- SOD Municipal & Industrial ---
        if add_del_sod_mi:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_SOD_MI','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_PMI_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_SWP_PMI_S','DELIVERY-SWP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- NOD Municipal & Industrial ---
        if add_del_nod_mi:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_NOD_MI','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_PMI_N_WAMER','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_SWP_PMI_N','DELIVERY-SWP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- Total Exports ---
        if add_total_exports:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','TOTAL_EXPORTS','EXPORTS-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','C_DMC003','CHANNEL','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','C_CAA003_SWP','FLOW-DELIVERY','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','C_CAA003_CVP','FLOW-DELIVERY','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- SL Total ---
        if addsl:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','S_SLTOT','STORAGE-CALC','1MON','L2020A','PER-AVER','TAF'),
                add_cols=[
                    ('CALSIM','S_SLUIS_CVP','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_SLUIS_SWP','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                ]
            )
        
        # --- Pumping Total ---
        if addpump:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','D_TOTAL','CHANNEL-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','C_DMC000','CHANNEL','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','C_CAA003','CHANNEL','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- Reservoir Totals ---
        if addres:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','S_RESTOT','STORAGE-CALC','1MON','L2020A','PER-AVER','TAF'),
                add_cols=[
                    ('CALSIM','S_OROVL','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_MELON','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_SHSTA','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_MLRTN','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_FOLSM','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_TRNTY','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                ]
            )
        
            add_combined_column_if_exists(
                df,
                ('CALCULATED','S_RESTOT_NOD','STORAGE-CALC','1MON','L2020A','PER-AVER','TAF'),
                add_cols=[
                    ('CALSIM','S_OROVL','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_SHSTA','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_TRNTY','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','S_FOLSM','STORAGE','1MON','L2020A','PER-AVER','TAF'),
                ]
            )
        
        # --- CVP TOTAL ---
        if adddelcvp:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_CVP_TOTAL','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_TOTAL_N','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_TOTAL_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- CVP+SWP TOTAL ---
        if adddelcvpswp:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_CVPSWP_TOTAL','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALCULATED','DEL_CVP_TOTAL','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_PAG_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_SWP_PAG_S','DELIVERY-SWP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_PEX_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- CVP PAG ---
        if adddelcvpag:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_CVP_PAG_TOTAL','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_PAG_N','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_PAG_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- CVP PSC + EX ---
        if addcvpscex:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_CVP_PSCEX_TOTAL','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_PSC_N','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_PEX_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- CVP PRF ---
        if addcvpprf:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','DEL_CVP_PRF_TOTAL','DELIVERY-CALC','1MON','L2020A','PER-AVER','CFS'),
                add_cols=[
                    ('CALSIM','DEL_CVP_PRF_N','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                    ('CALSIM','DEL_CVP_PRF_S','DELIVERY-CVP','1MON','L2020A','PER-AVER','CFS'),
                ]
            )
        
        # --- AWOANN XA ---
        if add_awoann_xa:
            add_combined_column_if_exists(
                df,
                ('CALCULATED','AWOANN_ALL_DV','ANNUAL-APPLIED-WATER-CALC','1MON','L2020A','PER-AVER','TAF'),
                add_cols=[
                    ('CALSIM','AWOANN_64_XADV','ANNUAL-APPLIED-WATER','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','AWOANN_72_XA1DV','ANNUAL-APPLIED-WATER','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','AWOANN_72_XA2DV','ANNUAL-APPLIED-WATER','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','AWOANN_72_XA3DV','ANNUAL-APPLIED-WATER','1MON','L2020A','PER-AVER','TAF'),
                    ('CALSIM','AWOANN_73_XADV','ANNUAL-APPLIED-WATER','1MON','L2020A','PER-AVER','TAF'),
                ]
            )

        new_columns = [(col[0], f'{col[1]}_{index_name[:]}', *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(new_columns)
        df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']
        combined_df = pd.concat([combined_df, df], axis=1)

    return combined_df

def preprocess_demands_deliveries(DemandFilePath, DemandFileTab, DemMin, DemMax, study_name, dvdss_name, svdss_name, datetime_start_date, datetime_end_date, aggregate_demands = True, aggregate_deliveries = True):
    # there's an excel file that contains a big table of demand unit information
    # we want the information relating to the deliveries and demands for each 
    # demand unit

    # set the filepath of the excel file
    xlfp = DemandFilePath

    # index names
    expected_names = ['A','B','C','E','F','Type','Units']

    # # set the run name identifier for the output csv filename
    # run_name_long = study_name

    # # set an output directory
    # outdir = r'full/path/to/this/folder/demands_deliveries_process'

    # set the dss filepath for the deliveries (DV file) and demands (SV file)
    dv_fp = dvdss_name
    sv_fp = svdss_name

    #datetime_start_date = dt.datetime(1921,10,31)
    #datetime_end_date = dt.datetime(2021, 9,30)

    # read the data from columns A-O in sheet 'all_demand_units'
    # get the du listing
    hdr, du_ = cu.read_from_excel(xlfp, DemandFileTab, DemMin, DemMax, hdr=True)
    du_df = pd.DataFrame(du_, columns = hdr)

    # print(du_df)

    # there are duplicates in the table because there are sometimes multiple irrigation
    # districts per demand unit - drop those duplicates now
    all_du = du_df[['Demand_Variable','Delivery_Variable']].drop_duplicates()

    # get rid of any lines that don't have delivery variables defined (non-project or external delivery points)
    # all_du = all_du[all_du['Delivery_Variable']!='None']

    #%% prep and retrieve demand variables from SV (input) file

    # create a list of just the demands variables
    demands_list = []
    demands_dssvar_list = []
    for d in all_du.Demand_Variable:
        if d not in ['UD_EBMUD (calculated in WRESL)', '1911.5 TAF']: #skip including UD_EBMUD and the 1911.5 for now - we'll get those later
            if '+' in d:
                lus = d.split('+')
                for i in lus:
                    demands_list.append(i.strip(' '))
            else:
                demands_dssvar_list.append(d.strip())
                demands_list.append(d.strip())

    # print("demands_list:")
    # print(demands_list)
    # print("demands_dssvar_list:")
    # print(demands_dssvar_list)
    
    # most demadns are in the SV file, but there are a few in the DV file that need
    # to be extracted -  this next section divides variables into and SV and DV set
    this_dat = []
    sv_list = []
    dv_list = []
    for i in demands_list: #all_du.Demand_Variable.sort_values():
        
        if i[0:2]=='UD' or i.upper()=='AW_NIDDC_NA3' or i.upper()=='D_LYONS_TCN000_DEM' or i.upper()=='AW_ELDID_NA1' or i.upper()=='UD_CSPSO' :
            # urban demands - SV list
            sv_list.append(i.upper())
        elif i.upper()=='DEM_D_CAA046_71_PA7_PIN':
            # for now, exclude 71_PA7 until we figure out how best to deal 
            # with SWP Table A, Carryover, and Interruptible demands tabulations
            pass
        else:
            dv_list.append(i.upper())     
    # print("dv_list:")        
    # print(dv_list)        

    # ensure the sv_list doesn't ahve any duplicates - not sure why UD_25_PU keeps showing up twice
    sv_list = list(set(sv_list))

    # create a dataframe suitable for use with coeqwal functions
    demand_var_dv_df = pd.DataFrame(data=dv_list, columns=['Part B:'])   
    demand_var_dv_df['Part C:'] = [""]*len(demand_var_dv_df)     

    # print('demand_var_dv_df:')
    # print(demand_var_dv_df)
    # demand_var_dv_df.to_csv("demand_var_dv_df.csv")
    
    demand_var_sv_df = pd.DataFrame(data=sv_list, columns=['Part B:'])   
    demand_var_sv_df['Part C:'] = [""]*len(demand_var_sv_df)   

    # print('demand_var_sv_df:')
    # print(demand_var_sv_df)
    # demand_var_sv_df.to_csv("demand_var_sv_df.csv")

    # the dex.preprocess_study_dss function was modified to deal with a variable listing
    # that just has a "B-part" - consider adapting 
    demands_sv_df = preprocess_study_dss(demand_var_sv_df, sv_fp, datetime_start_date, datetime_end_date,
                                        addsl=False, addres = False, addpump = False, adddelcvp = False, 
                                        adddelcvpag = False, addcvpscex = False, addcvpprf = False, 
                                        adddelcvpswp = False, add_nod_storage = False, add_sod_storage = False, 
                                        add_del_nod_ag = False, add_del_nod_mi = False, add_del_sod_mi = False, 
                                        add_del_sod_ag = False, add_total_exports = False, 
                                        add_del_swp_total = False, add_awoann_xa = False)
    # we can get the demands from the DV file too
    demands_dv_df = preprocess_study_dss(demand_var_dv_df, dv_fp, datetime_start_date, datetime_end_date,
                                        addsl=False, addres = False, addpump = False, adddelcvp = False, 
                                        adddelcvpag = False, addcvpscex = False, addcvpprf = False, 
                                        adddelcvpswp = False, add_nod_storage = False, add_sod_storage = False, 
                                        add_del_nod_ag = False, add_del_nod_mi = False, add_del_sod_mi = False, 
                                        add_del_sod_ag = False, add_total_exports = False, 
                                        add_del_swp_total = False, add_awoann_xa = False)

    # combine the two demands files together
    demands_df = pd.concat([demands_sv_df, demands_dv_df], axis=1)

    # drop duplicate columns
    demands_df = demands_df.loc[:, ~demands_df.columns.duplicated()]

    # record all the intermediate columns used in aggregation
    used_cols = []

    # add the Calaveras info for JLIND from calaveras_dist.table
# month   sewd_ag   sewd_mi cacwd_ag cacwd_mi
# 1       2.6       6.22      2.6        5.47
# 2       0.08      5.66      0.08       9.55
# 3       0.0       7.00      0.0        13.38
# 4       0.0       7.67      0.0        15.84
# 5       0.04      9.28      0.04       15.90
# 6       1.56      9.38      1.56       12.61
# 7       9.8       9.90      9.8        8.32
# 8       18.44     10.01     18.44      4.19
# 9       19.16     9.76      19.16      3.55
# 10      21.36     9.55      21.36      3.56
# 11      17.36     8.18      17.36      3.22
# 12      9.59      7.38      9.59       4.41
    cacwd_mi = [15.84,15.90,12.61,8.32,4.19,3.55,3.56,3.22,4.41,5.47,9.55,13.38]
    demands_df[('MANUAL-ADD','UD_JLIND','URBAN-DEMAND','1MON','L2020A','PER-CUM','TAF')] = (0.6 * 3.5 * (np.array(cacwd_mi)[demands_df.index.month - 1]) / 100)

    # add the table info for UD_PLMAS from UF_MFFDDelivery.table
    #  0.672 (convert to CFS) * PLMASMonthDist (lookup from UF_MFFDelivery.table in column PLMASPatt)
    # month	PLMASPatt	LCCWDPatt		
# 1	0.09	0.02		
# 2	0.05	0.00		
# 3	0.05	0.00		
# 4	0.04	0.00		
# 5	0.04	0.00		
# 6	0.05	0.01		
# 7	0.06	0.03		
# 8	0.09	0.17		
# 9	0.12	0.23		
# 10	0.15	0.21		
# 11	0.14	0.26		
# 12	0.12	0.08		
    PLMASPatt = [0.04,0.04,0.05,0.06,0.09,0.12,0.15,0.14,0.12,0.09,0.05,0.05]
    taf_to_cfs = 1233481.84 / (demands_df.index.days_in_month * 86400)
    demands_df[('MANUAL-ADD','UD_PLMAS','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS')] = (0.672 * taf_to_cfs * (np.array(PLMASPatt)[demands_df.index.month - 1]))

 # #    define D_ANC000_ANGLS_DEM { !units CFS 
	# # case october  {condition month == oct
	# # 	value 2.6}
	# # case november {condition month == nov
	# # 	value 1.7} 
	# # case december {condition month == dec
	# # 	value 1.3}
	# # case january  {condition month == jan
	# # 	value 1.3}
	# # case february {condition month == feb
	# # 	value 1.4}
	# # case march    {condition month == mar
	# # 	value 1.3}
	# # case april    {condition month == apr
	# # 	value 2.}
	# # case mmay     {condition month == may
	# # 	value 2.9}
	# # case june     {condition month == jun
	# # 	value 4.}
	# # case july     {condition month == jul
	# # 	value 5.2}
	# # case august   {condition month == aug
	# # 	value 4.9}
	# # case sept     {condition month == sep
	# # 	value 4.4}}
 #    # CORRECT TABLE
 #    ANGLS_DEM = [15.84,15.90,12.61,8.32,4.19,3.55,3.56,3.22,4.41,5.47,9.55,13.38]
 #    demands_df[('MANUAL-ADD','D_ANC000_ANGLS_DEM','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS')] = np.array(ANGLS_DEM)[demands_df.index.month - 1]

    # add a flat demand for MWD and ANTOC and CPLT
    MWD_yearly_taf_value = 1911.5
    ANTOC_monthly_cfs_value = 25
    CPLT_monthly_taf_value = 0
    CPLT_monthly_cfs_value = 0
    days_in_month = demands_df.index.days_in_month
    ANTOC_monthly_cfs_values = ANTOC_monthly_cfs_value * 0.001984 * days_in_month
    demands_df[('MANUAL-ADD','TABLEA_CONTRACT_MWD','URBAN-DEMAND','1MON','L2020A','PER-CUM','TAF')] = len(demands_df)*[MWD_yearly_taf_value/12]
    demands_df[('MANUAL-ADD','UD_ANTOC','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS')] = ANTOC_monthly_cfs_values
    demands_df[('MANUAL-ADD','U_CPLT','URBAN-DEMAND','1MON','L2020A','PER-CUM','TAF')] = CPLT_monthly_taf_value
    # also add the CFS value for the TAF definitions to provide both columns
    demands_df[('MANUAL-ADD','U_CPLT','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS')] = CPLT_monthly_cfs_value
    demands_df[('MANUAL-ADD','TABLEA_CONTRACT_MWD','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS')] = ((MWD_yearly_taf_value / 12) * 1000 * 43560) / (86400 * demands_df.index.days_in_month)

# define D_ANC000_ANGLS_DEM { !units CFS 
# 	case october  {condition month == oct
# 		value 2.6}
# 	case november {condition month == nov
# 		value 1.7} 
# 	case december {condition month == dec
# 		value 1.3}
# 	case january  {condition month == jan
# 		value 1.3}
# 	case february {condition month == feb
# 		value 1.4}
# 	case march    {condition month == mar
# 		value 1.3}
# 	case april    {condition month == apr
# 		value 2.}
# 	case mmay     {condition month == may
# 		value 2.9}
# 	case june     {condition month == jun
# 		value 4.}
# 	case july     {condition month == jul
# 		value 5.2}
# 	case august   {condition month == aug
# 		value 4.9}
# 	case sept     {condition month == sep
# 		value 4.4}}    

    # make dictionary
    month_to_value = {
        10: 2.6,  # Oct
        11: 1.7,  # Nov
        12: 1.3,  # Dec
         1: 1.3,  # Jan
         2: 1.4,  # Feb
         3: 1.3,  # Mar
         4: 2.0,  # Apr
         5: 2.9,  # May
         6: 4.0,  # Jun
         7: 5.2,  # Jul
         8: 4.9,  # Aug
         9: 4.4,  # Sep
}

    # ensure datetime index
    demands_df.index = pd.to_datetime(demands_df.index)
    
    # compute 1D array of monthly values
    upang_values = demands_df.index.month.map(month_to_value)
    
    # ensure it's a 1D Series
    upang_values = pd.Series(upang_values, index=demands_df.index)
    
    # assign values
    col = ('MANUAL-ADD','D_ANC000_ANGLS_DEM','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS')
    demands_df[col] = upang_values.values  # always 1D
    
    # check assignment
    # test = pd.DataFrame({
    #     "date": demands_df.index,
    #     "month": demands_df.index.month,
    #     "assigned_value": demands_df[col]
    # })    
    # print(test.head(20))
    # print("Shape of assigned column:", demands_df[col].shape)

    # print("demands_df:")
    # print(demands_df.head(5))
    # demands_df.to_csv("demands_df.csv")
    
    # aggregate demand variables
    if aggregate_demands:
            
        # ---------- UD_NAPA ----------
        cols_ud_napa = [
            ('CALSIM', 'SWP_CO_NAPA', 'SWP_DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'SWP_IN_NAPA', 'SWP_DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'SWP_TA_NAPA', 'SWP_DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_BKR004_NBA009_NAPA_PLS', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]
        
        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','UD_NAPA','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_napa,
            sub_cols=None,
            multiplier=1.0,
            record_used_cols=used_cols,
        )
                
        # ---------- UD_AMCYN ----------
        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','UD_AMCYN','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_BKR004_NBA009_NAPA', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ],
            sub_cols=[
                ('CALSIM', 'D_BKR004_NBA009_NAPA_PLS', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS')
            ],
            multiplier=0.179,
            record_used_cols=used_cols,
        )
                
        # ---------- D_MWD ----------
        cols_d_mwd = [
            ('CALSIM', 'D_PRRIS_MWDSC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_ESB413_MWDSC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_WSB031_MWDSC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_ESB433_MWDSC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_CAA194_KERNB_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]        
        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','D_MWD_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_d_mwd,
            sub_cols=None,
            multiplier=1.0,
            record_used_cols=used_cols,
        )
                
        # ---------- UD_AMADR_NU ----------
        cols_ud_amadr = [
            ('CALSIM', 'DEMAND_AMADR_CAWP_', 'DEBUG', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'DEMAND_AMADR_AWS_',  'DEBUG', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]
        
        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','UD_AMADR_NU','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_amadr,
            sub_cols=None,
            multiplier=1.0,
            record_used_cols=used_cols,
        )
        
        # --- CSB103 ---
        # DEM_D_CSB103_BRBRA_PMI = (SHORT_D_CSB103_BRBRA_PMI + D_CSB103_BRBRA_PMI)/perdv_swp_34 
        cols_ud_csb103 = [
            ('CALSIM', 'SHORT_D_CSB103_BRBRA_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_CSB103_BRBRA_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_csb103 = [
            ('CALSIM', 'PERDV_SWP_34', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_CSB103_BRBRA_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_csb103,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_csb103[0],
            record_used_cols=used_cols,
        )

        # --- CSB038 ---
        # DEM_D_CSB038_OBISPO_PMI = (short_D_CSB038_OBISPO_PMI + D_CSB038_OBISPO_PMI)/perdv_swp_35  
        cols_ud_csb038 = [
            ('CALSIM', 'SHORT_D_CSB038_OBISPO_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_CSB038_OBISPO_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_csb038 = [
            ('CALSIM', 'PERDV_SWP_35', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_CSB038_OBISPO_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_csb038,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_csb038[0],
            record_used_cols=used_cols,
        )
        
        # --- DEM_VNTRA_PMI ---
        # dem_VNTRA_PMI = (short_D_CSTIC_VNTRA_PMI + D_CSTIC_VNTRA_PMI)/perdv_swp_39 + (short_D_PYRMD_VNTRA_PMI + D_PYRMD_VNTRA_PMI)/perdv_swp_38        
        demands_df = add_two_term_ratio_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_VNTRA_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
        
            term1_num_cols=[
                ('CALSIM','SHORT_D_CSTIC_VNTRA_PMI','DELIVERY-SHORTAGE','1MON','L2020A','PER-AVER','CFS'),
                ('CALSIM','D_CSTIC_VNTRA_PMI','FLOW-DELIVERY','1MON','L2020A','PER-AVER','CFS')
            ],
            term1_den_col=('CALSIM','PERDV_SWP_39','SWP-OUTPUT','1MON','L2020A','PER-AVER','PERCENT'),
        
            term2_num_cols=[
                ('CALSIM','SHORT_D_PYRMD_VNTRA_PMI','DELIVERY-SHORTAGE','1MON','L2020A','PER-AVER','CFS'),
                ('CALSIM','D_PYRMD_VNTRA_PMI','FLOW-DELIVERY','1MON','L2020A','PER-AVER','CFS')
            ],
            term2_den_col=('CALSIM','PERDV_SWP_38','SWP-OUTPUT','1MON','L2020A','PER-AVER','PERCENT'),
            record_used_cols=used_cols,
        )

        # --- ESB324 ---
        # dem_D_ESB324_AVEK_PMI = (short_D_ESB324_AVEK_PMI + D_ESB324_AVEK_PMI)/perdv_swp_4 
        cols_ud_esb324 = [
            ('CALSIM', 'SHORT_D_ESB324_AVEK_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_ESB324_AVEK_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_esb324 = [
            ('CALSIM', 'PERDV_SWP_4', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_ESB324_AVEK_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_esb324,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_esb324[0],
            record_used_cols=used_cols,
        )

        # --- ESB347 ---
        # dem_D_ESB347_PLMDL_PMI = (short_D_ESB347_PLMDL_PMI + D_ESB347_PLMDL_PMI)/perdv_swp_29 
        cols_ud_esb347 = [
            ('CALSIM', 'SHORT_D_ESB347_PLMDL_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_ESB347_PLMDL_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_esb347 = [
            ('CALSIM', 'PERDV_SWP_29', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_ESB347_PLMDL_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_esb347,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_esb347[0],
            record_used_cols=used_cols,
        )

        # --- ESB414 ---
        # dem_D_ESB414_BRDNO_PMI = (short_D_ESB414_BRDNO_PMI + D_ESB414_BRDNO_PMI)/perdv_swp_30
        cols_ud_esb414 = [
            ('CALSIM', 'SHORT_D_ESB414_BRDNO_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_ESB414_BRDNO_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_esb414 = [
            ('CALSIM', 'PERDV_SWP_30', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_ESB414_BRDNO_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_esb414,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_esb414[0],
            record_used_cols=used_cols,
        )
 
        # --- ESB415 ---
        # dem_D_ESB415_GABRL_PMI = (short_D_ESB415_GABRL_PMI + D_ESB415_GABRL_PMI)/perdv_swp_31 
        cols_ud_esb415 = [
            ('CALSIM', 'SHORT_D_ESB415_GABRL_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_ESB415_GABRL_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_esb415 = [
            ('CALSIM', 'PERDV_SWP_31', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_ESB415_GABRL_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_esb415,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_esb415[0],
            record_used_cols=used_cols,
        )

        # --- ESB420 ---
        # dem_D_ESB420_GRGNO_PMI = (short_D_ESB420_GRGNO_PMI + D_ESB420_GRGNO_PMI)/perdv_swp_32 
        cols_ud_esb420 = [
            ('CALSIM', 'SHORT_D_ESB420_GRGNO_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_ESB420_GRGNO_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_esb420 = [
            ('CALSIM', 'PERDV_SWP_32', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_ESB420_GRGNO_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_esb420,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_esb420[0],
            record_used_cols=used_cols,
        )

        # --- DEM_ACFC ---
        # dem_ACFC_PMI = (short_D_SBA009_ACFC_PMI + D_SBA009_ACFC_PMI)/perdv_swp_1 + (short_D_SBA020_ACFC_PMI + D_SBA020_ACFC_PMI)/perdv_swp_2
        demands_df = add_two_term_ratio_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_ACFC','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
        
            term1_num_cols=[
                ('CALSIM','SHORT_D_SBA009_ACFC_PMI','DELIVERY-SHORTAGE','1MON','L2020A','PER-AVER','CFS'),
                ('CALSIM','D_SBA009_ACFC_PMI','FLOW-DELIVERY','1MON','L2020A','PER-AVER','CFS')
            ],
            term1_den_col=('CALSIM','PERDV_SWP_1','SWP-OUTPUT','1MON','L2020A','PER-AVER','PERCENT'),
        
            term2_num_cols=[
                ('CALSIM','SHORT_D_SBA020_ACFC_PMI','DELIVERY-SHORTAGE','1MON','L2020A','PER-AVER','CFS'),
                ('CALSIM','D_SBA020_ACFC_PMI','FLOW-DELIVERY','1MON','L2020A','PER-AVER','CFS')
            ],
            term2_den_col=('CALSIM','PERDV_SWP_2','SWP-OUTPUT','1MON','L2020A','PER-AVER','PERCENT'),
            record_used_cols=used_cols,
        )

        # --- SBA029 ---
        # dem_D_SBA029_ACWD_PMI = (short_D_SBA029_ACWD_PMI + D_SBA029_ACWD_PMI)/perdv_swp_3 
        cols_ud_sba029 = [
            ('CALSIM', 'SHORT_D_SBA029_ACWD_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_SBA029_ACWD_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_sba029 = [
            ('CALSIM', 'PERDV_SWP_3', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_SBA029_ACWD_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_sba029,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_sba029[0],
            record_used_cols=used_cols,
        )

        # --- SBA036 ---
        # dem_D_SBA036_SCVWD_PMI = (short_D_SBA036_SCVWD_PMI + D_SBA036_SCVWD_PMI)/perdv_swp_35
        cols_ud_sba029 = [
            ('CALSIM', 'SHORT_D_SBA036_SCVWD_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_SBA036_SCVWD_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_sba029 = [
            ('CALSIM', 'PERDV_SWP_35', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_SBA036_SCVWD_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_sba029,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_sba029[0],
            record_used_cols=used_cols,
        )

        # ---------- D_SVWRD_CSTLN_PMI ----------
        cols_ud_amadr = [
            ('CALSIM', 'D_SBA009_ACFC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_SBA020_ACFC_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]
        
        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','UD_AMADR_NU','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_amadr,
            sub_cols=None,
            multiplier=1.0,
            record_used_cols=used_cols,
        )
        # --- SVWRD ---
        # dem_D_SVRWD_CSTLN_PMI = (short_D_SVRWD_CSTLN_PMI + D_SVRWD_CSTLN_PMI)/perdv_swp_11
        cols_ud_svwrd = [
            ('CALSIM', 'SHORT_D_SVRWD_CSTLN_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_SVRWD_CSTLN_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_svwrd = [
            ('CALSIM', 'PERDV_SWP_11', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_SVRWD_CSTLN_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_svwrd,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_svwrd[0],
            record_used_cols=used_cols,
        )
    
        # --- KCWA ---
        # dem_D_CAA194_KERNA_PMI = (short_D_CAA194_KERNA_PMI + D_CAA194_KERNA_PMI)/perdv_swp_15 
        cols_ud_kerna = [
            ('CALSIM', 'SHORT_D_CAA194_KERNA_PMI', 'DELIVERY-SHORTAGE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ('CALSIM', 'D_CAA194_KERNA_PMI',  'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
        ]  
        div_col_ud_kerna = [
            ('CALSIM', 'PERDV_SWP_15', 'SWP-OUTPUT', '1MON', 'L2020A', 'PER-AVER', 'PERCENT'),
        ]  

        demands_df = add_combined_column_if_exists(
            demands_df,
            target_col=('CALCULATED','DEM_D_CAA194_KERNA_PMI','URBAN-DEMAND','1MON','L2020A','PER-CUM','CFS'),
            add_cols=cols_ud_kerna,
            sub_cols=None,
            multiplier=1.0,
            divisor_col=div_col_ud_kerna[0],
            record_used_cols=used_cols,
        )
    
    # drop duplicate columns
    demands_df = demands_df.loc[:, ~demands_df.columns.duplicated()]

    # --- DROP EVERYTHING USED ---
    used_cols = list(set(used_cols))  # dedupe
    demands_df = demands_df.drop(columns=used_cols)    
    print(f"Dropped {len(used_cols)} intermediate columns.")  
    
    # print("Monthly demands:")
    # print("demands_df:")
    # print(demands_df)

    #%%  Demands data - convert from CFS to TAF and write out to csvs
    demands_taf_df = cu.convert_all_cfs_to_taf(demands_df)
    
    # print("demands_taf_df:")
    # print(demands_taf_df.head(5))
    
    #%% prep and retrieve delivery variables from DV (results) file

    delivs_list = []
    for d in all_du.Delivery_Variable:
        if d not in ['DN_EBMUD', 'del_swp_mwd']:
            if '+' in d:
                lus = d.split('+')
                for i in lus:
                    delivs_list.append(i.strip(' '))
            else:
                lu1 = d.strip()
                lu2 = lu1
                delivs_list.append(lu2)
    # add delivery for MWD
    delivs_list.append('DEL_SWP_MWD')   

    deliv_var_df = pd.DataFrame(data=delivs_list, columns=['Part B:'])
    deliv_var_df['Part C:'] = [""]*len(deliv_var_df)

    delivs_cfs_df = preprocess_study_dss(deliv_var_df, dv_fp, datetime_start_date, datetime_end_date,
                                        addsl=False, addres = False, addpump = False, adddelcvp = False, 
                                        adddelcvpag = False, addcvpscex = False, addcvpprf = False, 
                                        adddelcvpswp = False, add_nod_storage = False, add_sod_storage = False, 
                                        add_del_nod_ag = False, add_del_nod_mi = False, add_del_sod_mi = False, 
                                        add_del_sod_ag = False, add_total_exports = False, 
                                        add_del_swp_total = False, add_awoann_xa = False)

    #%% check for missing data

    deliv_cols = list(delivs_cfs_df.columns.get_level_values('B'))
    missing_delivs = []
    for delvar in delivs_list:
        # print('Processing ' + delvar)
        if delvar.strip('/').upper() not in deliv_cols:
            missing_delivs.append(delvar)
            
    # print("Monthly deliveries:")
    # print("delivs_cfs_df:")
    # print(delivs_cfs_df.head(5))
    
    used_cols = []
    
    # aggregate delivery variables
    if aggregate_deliveries:      

        # --- DN_06_NA ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_06_NA','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_06_NA', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_06_NA', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_07N_NA ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_07N_NA','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_07N_NA', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_07N_NA', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_07S_NA ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_07S_NA','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_07S_NA', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_07S_NA', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_15N_NA1 ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_15N_NA1','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_15N_NA1', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_15N_NA1', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_15S_NA1 ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_15S_NA1','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_15S_NA1', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_15S_NA1', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_16_NA1 ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_16_NA1','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_16_NA1', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_16_NA1', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_17N_NA ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_17N_NA','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_17N_NA', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_17N_NA', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_20_NA2 ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_20_NA2','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_20_NA2', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_20_NA2', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_26S_NA ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_26S_NA','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_26S_NA', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_26S_NA', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_60S_NA1 ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_60S_NA1','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_60S_NA1', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_60S_NA1', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- DN_60S_NA2 ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','DN_60S_NA2','SW_DELIVERY-NET','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'GP_60S_NA2', 'GW-PUMPING', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'RU_60S_NA2', 'REUSE', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_AMCYN ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_AMADR_NU','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_WTPAMC_AMCYN', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_WTPJAC_AMCYN', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_AMADR_NU ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_AMADR_NU','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_TGC003_AMADR_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_TBAUD_AMADR_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
         # --- D_CACWD ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_CACWD','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_MFM007_WSPNT_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_BCM003_WSPNT_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_VNTRA_MPMI ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_VNTRA_PMI','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_CSTIC_VNTRA_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_PYRMD_VNTRA_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_ACFC_PMI ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_ACFC_PMI','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_SBA009_ACFC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_SBA020_ACFC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_AMADR_NU ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_AMADR_NU','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_TGC003_AMADR_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_TBAUD_AMADR_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_AMCYN ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_AMCYN','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_WTPAMC_AMCYN', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_WTPJAC_AMCYN', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_ANTOC ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_ANTOC','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_SJR006_ANTOC', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_CCC007_ANTOC', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_FRFLD ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_FRFLD','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_WTPNBR_FRFLD', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_WTPWMN_FRFLD', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_GRSVL ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_GRSVL','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_CSD014_GRSVL', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_DES006_GRSVL', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
                
        # --- D_WSPNT_NU ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_WSPNT_NU','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_MFM007_WSPNT_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_BCM003_WSPNT_NU', 'DIVERSION', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )
        
        # --- D_ACFC_PMI ---
        delivs_cfs_df = add_combined_column_if_exists(
            delivs_cfs_df,
            target_col=('CALCULATED','D_ACFC_PMI','FLOW-DELIVERY','1MON','L2020A','PER-CUM','CFS'),
            add_cols=[
                ('CALSIM', 'D_SBA009_ACFC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
                ('CALSIM', 'D_SBA020_ACFC_PMI', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),
            ],
            record_used_cols=used_cols,
        )

    # drop duplicate columns
    delivs_cfs_df = delivs_cfs_df.loc[:, ~delivs_cfs_df.columns.duplicated()]

    # --- DROP EVERYTHING USED ---
    used_cols = list(set(used_cols))  # dedupe
    delivs_cfs_df = delivs_cfs_df.drop(columns=used_cols)
    print(f"Dropped {len(used_cols)} intermediate columns.")
    
    #%%  Delivery data - convert from CFS to TAF and write out to csvs
    delivs_taf_df = cu.convert_all_cfs_to_taf(delivs_cfs_df)

    # preserve MultiIndex column names (auto-match level count)
    if isinstance(delivs_taf_df.columns, pd.MultiIndex):
        delivs_taf_df.columns = delivs_taf_df.columns.set_names(expected_names[:delivs_taf_df.columns.nlevels])
    
    annual_del_data = []
    for c in delivs_cfs_df.columns:
        # print('Annualizing delivs_cfs_df column')
        dd =csplt.annualize(delivs_cfs_df.loc[:, [c]], on='YE-FEB', how='auto-sum') #contract years are Mar-Feb; the 'auto-sum' argument handles the CFS->TAF conversion
        # print('Appending annualized column')
        annual_del_data.append(dd)
        
    annual_del_df = pd.concat(annual_del_data, axis=1)
    annual_del_df.index = annual_del_df.index.year

    # print("Annual deliveries:")
    # print(annual_del_df.head(5))
    
    # mean_annual_del = annual_del_df.mean(axis=0) # caclualte the mean across all years
    mean_annual_del = annual_del_df.mean(axis=0).to_frame().T

    # print("Mean annual deliveries:")
    # print(mean_annual_del.head(5))
    
    # preserve MultiIndex column names (auto-match level count)
    if isinstance(demands_taf_df.columns, pd.MultiIndex):
        demands_taf_df.columns = demands_taf_df.columns.set_names(expected_names[:demands_taf_df.columns.nlevels])
    # print("Monthly demands:")
    # print(demands_taf_df.head(5))
    annual_dem_data = []
    for c in demands_taf_df.columns:
        # print('Annualizing demands_taf_df column')
        dd =csplt.annualize(demands_taf_df.loc[:, [c]], on='YE-FEB', how='auto-sum')
        # print('Appending annualized column')
        annual_dem_data.append(dd)
    annual_dem_df = pd.concat(annual_dem_data, axis=1)
    annual_dem_df.index = annual_dem_df.index.year
    # print("Annual demands:")
    # print(annual_dem_df.head(5))

    # mean_annual_dem = annual_dem_df.mean(axis=0) # caclualte the mean across all years
    mean_annual_dem = annual_dem_df.mean(axis=0).to_frame().T
    # print("Mean annual demands:")
    # print(mean_annual_dem.head(5))


    return demands_taf_df, delivs_taf_df, annual_dem_df, annual_del_df, mean_annual_dem, mean_annual_del

def fix_sr_leading_zero(col_tuple):
    # col_tuple is like ('IWFM', 'SR1:L1', ...)
    part = col_tuple[1]
    # If 'SR' followed by single digit without leading zero, add zero
    import re
    m = re.match(r'SR(\d+)(:L\d+)', part)
    if m:
        number = int(m.group(1))
        fixed_part = f"SR{number:02d}{m.group(2)}"  # Adds leading zero if single digit
        return (col_tuple[0], fixed_part) + col_tuple[2:]
    return col_tuple
    
def fix_sr_remove_leading_zero(col_tuple):
    part = col_tuple[1]
    import re
    m = re.match(r'SR0*(\d+)(:L\d+)', part)
    if m:
        fixed_part = f"SR{int(m.group(1))}{m.group(2)}"
        return (col_tuple[0], fixed_part) + col_tuple[2:]
    return col_tuple

def preprocess_GW_data_study_dss(df, dss_name, datetime_start_date, datetime_end_date, addSRlevels=True, num_vars = 66, convertAcFtToTaf = True):
    dvar_list = []
    combined_df = pd.DataFrame()
    
    for i, r in df.iterrows():
        if r["Part C:"] == '':
            dvar_list.append(f'/{r["Part B:"]}/')
        else:
            dvar_list.append(f'/{r["Part B:"]}/{r["Part C:"]}/')

    # print('dvar_list:')
    # print(dvar_list)

    # Create a blank python "calsim" object
    thiscs3 = cs3.calsim()

    # add start and end dates
    print('Start: ')
    print(datetime_start_date)
    print('End: ')
    print(datetime_end_date)   
    thiscs3.StartDate = datetime_start_date
    thiscs3.EndDate = datetime_end_date

    # add path to DSS
    DSS_FP = dss_name
    thiscs3.DV_FP = DSS_FP

    # Retrieve the DSS data variables from the DSS file
    thiscs3.DVdata = cs3.csDVdata(thiscs3)
    thiscs3.DVdata.getDVts(filter=dvar_list)

    df = thiscs3.DVdata.DVtsDF.copy(deep=True)

# create aggregate variables
    if addSRlevels:
        # df[('CALCULATED', 'SR10:TOT', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT')] = df.loc[:,[('IWFM', 'SR10:L1', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT'),('IWFM', 'SR10:L2', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT'),('IWFM', 'SR10:L3', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT')]].sum(axis=1)
        for i in range(1, num_vars+1):
            zone = f"SR{i:02d}"
            target_col = ('CALCULATED', f'{zone}:TOT', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.')
            
            # Build list of source columns to sum: L1, L2, L3
            source_cols = [
                ('IWFM', f'{zone}:L1', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),
                ('IWFM', f'{zone}:L2', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),
                ('IWFM', f'{zone}:L3', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),
            ]
            
            fixed_source_cols = [fix_sr_remove_leading_zero(c) for c in source_cols]
            df[target_col] = df.loc[:, fixed_source_cols].sum(axis=1)

    new_columns = [(col[0], col[1], *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']

    if convertAcFtToTaf: 
        # Identify columns that end in 'AC.FT.'
        acft_cols = [col for col in df.columns if col[-1] == "AC.FT."]

        # Divide those columns by 1000
        df[acft_cols] = df[acft_cols] / 1000

        # Rename the last level of the column from 'AC.FT.' to 'TAF'
        renamed_cols = [
            col[:-1] + ("TAF",) if col in acft_cols else col
            for col in df.columns
        ]

        # Apply new column names
        df.columns = pd.MultiIndex.from_tuples(renamed_cols)

    # df.head(5)
    return df


def preprocess_compound_GW_data_dss(df, ScenarioDir, dss_names, index_names, min_datetime, max_datetime, addSRlevels=True, num_vars = 66, convertAcFtToTaf = True):
    dvar_list = []
    combined_df = pd.DataFrame()
    
    print("num_vars:" + str(num_vars))

    for i, r in df.iterrows():
        if r["Part C:"] == '':
            dvar_list.append(f'/{r["Part B:"]}/')
        else:
            dvar_list.append(f'/{r["Part B:"]}/{r["Part C:"]}/')


    for i in range(len(dss_names)):
        #get DSS and scenario index name
        dss_name = dss_names[i]
        index_name = index_names[i]
        print(dss_name)
        print(index_name)

        # Create a blank python "calsim" object
        thiscs3 = cs3.calsim()

        # add start and end dates
        thiscs3.StartDate = min_datetime
        thiscs3.EndDate = max_datetime

        # add path to DSS
        DSS_FP = os.path.join(ScenarioDir, dss_name)
        thiscs3.DV_FP = DSS_FP

        # Retrieve the DSS data variables from the DSS file
        thiscs3.DVdata = cs3.csDVdata(thiscs3)
        thiscs3.DVdata.getDVts(filter=dvar_list)

        df = thiscs3.DVdata.DVtsDF.copy(deep=True)

# create aggregate variables
        if addSRlevels:
            # df[('CALCULATED', 'SR10:TOT', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.')] = df.loc[:,[('IWFM', 'SR10:L1', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),('IWFM', 'SR10:L2', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),('IWFM', 'SR10:L3', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.')]].sum(axis=1)
            for j in range(1, num_vars+1):
                zone = f"SR{j:02d}"
                # print("zone: " + zone)
                # print(df.columns.get_level_values(1).unique())
                target_col = ('CALCULATED', f'{zone}:TOT', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.')
                # print("target_col:")
                # print(target_col)
                # Build list of source columns to sum: L1, L2, L3
                source_cols = [
                    ('IWFM', f'{zone}:L1', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),
                    ('IWFM', f'{zone}:L2', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),
                    ('IWFM', f'{zone}:L3', 'GW_STORAGE', '1MON', 'GW_STORAGE_AT_CALSIM_REGIONS', 'PER-CUM', 'AC.FT.'),
                ]
                # print("source_cols:")
                # print(source_cols)
                fixed_source_cols = [fix_sr_remove_leading_zero(c) for c in source_cols]
                # print("df.columns.tolist():")
                # print(df.columns.tolist())

                # for col in df.columns:
                #     if col[1].startswith('SR'):
                #         print(col)
                        
                #existing_cols = [col for col in source_cols if col in df.columns]
                existing_cols = [col for col in fixed_source_cols if col in df.columns]
                # print("existing_cols:")
                # print(existing_cols)
                if existing_cols:
                    df[target_col] = df.loc[:, existing_cols].sum(axis=1)
                    print(f"Added {target_col} from {existing_cols}")
                else:
                    print(f"Zone {zone}: No matching source columns found.")
                #df[target_col] = df.loc[:, source_cols].sum(axis=1)
                df[target_col] = df.loc[:, fixed_source_cols].sum(axis=1)
                # print("df at iteration " + str(j) + ":")
                # print(df)

        new_columns = [(col[0], f'{col[1]}_{index_name[:]}', *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
        # print("new_columns:")
        # print(new_columns)
        df.columns = pd.MultiIndex.from_tuples(new_columns)
        df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']
        combined_df = pd.concat([combined_df, df], axis=1)

    if convertAcFtToTaf: 
        # Identify columns that end in 'AC.FT.'
        acft_cols = [col for col in combined_df.columns if col[-1] == "AC.FT."]

        # Divide those columns by 1000
        combined_df[acft_cols] = combined_df[acft_cols] / 1000

        # Rename the last level of the column from 'AC.FT.' to 'TAF'
        renamed_cols = [
            col[:-1] + ("TAF",) if col in acft_cols else col
            for col in combined_df.columns
        ]

        # Apply new column names
        combined_df.columns = pd.MultiIndex.from_tuples(renamed_cols)
        # print("combined_df at iteration " + str(i) + ":")
        # print(combined_df)

    # print("final combined_df:")
    # print(combined_df)
    return combined_df

