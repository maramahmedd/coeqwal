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
CREATE DATASETS ACROSS STUDIES
(Aux functions to read and process studies (for single and multiple studies). Note: contain options to create hard-coded additional compound variables (if more are needed, add to these codes))
"""

def preprocess_study_dss(df, dss_name, datetime_start_date, datetime_end_date, addsl=True, addres = True, addpump = True, adddelcvp = True, adddelcvpag = True, addcvpscex = True, addcvpprf = True, adddelcvpswp = True, add_nod_storage = True, add_sod_storage = True, add_del_nod_ag = True, add_del_nod_mi = True, add_del_sod_mi = True, add_del_sod_ag = True, add_total_exports = True, add_del_swp_total = True):
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
    if add_nod_storage:
        df[('CALCULATED', 'NOD_STORAGE', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_NBLDB', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

    if add_sod_storage:
        df[('CALCULATED', 'SOD_STORAGE', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_NHGAN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_PEDRO', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MCLRE', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_HNSLY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

    if add_del_swp_total:
        df[('CALCULATED', 'DEL_SWP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_SWP_PAG', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PMI', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if add_del_nod_ag:
        df[('CALCULATED', 'DEL_NOD_AG', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PAG_N', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if add_del_sod_ag:
        df[('CALCULATED', 'DEL_SOD_AG', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PAG_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if add_del_sod_mi:
        df[('CALCULATED', 'DEL_SOD_MI', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PMI_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PMI_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if add_del_nod_mi:
        df[('CALCULATED', 'DEL_NOD_MI', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PMI_N_WAMER', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PMI_N', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)
 
    if add_total_exports:
        df[('CALCULATED', 'TOTAL_EXPORTS', 'EXPORTS-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'C_DMC003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003_SWP', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003_CVP', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)
 
    if addsl:
        df[('CALCULATED', 'S_SLTOT', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

    if addpump:
        df[('CALCULATED', 'D_TOTAL', 'CHANNEL-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'C_DMC000', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if addres:
        df[('CALCULATED', 'S_RESTOT', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)
        df[('CALCULATED', 'S_RESTOT_NOD', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF') ]].sum(axis=1)

    if adddelcvp:
        df[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_TOTAL_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_TOTAL_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if adddelcvpswp:
        df[('CALCULATED', 'DEL_CVPSWP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PAG_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if adddelcvpag:
        df[('CALCULATED', 'DEL_CVP_PAG_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if addcvpscex:
        df[('CALCULATED', 'DEL_CVP_PSCEX_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if addcvpprf:
        df[('CALCULATED', 'DEL_CVP_PRF_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PRF_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PRF_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    new_columns = [(col[0], col[1], *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']

    # df.head(5)
    return df


def preprocess_compound_data_dss(df, ScenarioDir, dss_names, index_names, min_datetime, max_datetime, addsl=True, addres = True, addpump = True, adddelcvp = True, adddelcvpag = True, addcvpscex = True, addcvpprf = True, adddelcvpswp = True, add_nod_storage = True, add_sod_storage = True, add_del_nod_ag = True, add_del_nod_mi = True, add_del_sod_mi = True, add_del_sod_ag = True, add_total_exports = True, add_del_swp_total = True):
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

# create aggregate variables
        if add_nod_storage:
            df[('CALCULATED', 'NOD_STORAGE', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_NBLDB', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

        if add_sod_storage:
            df[('CALCULATED', 'SOD_STORAGE', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_NHGAN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_PEDRO', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MCLRE', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_HNSLY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

        if add_del_swp_total:
            df[('CALCULATED', 'DEL_SWP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_SWP_PAG', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PMI', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if add_del_nod_ag:
            df[('CALCULATED', 'DEL_NOD_AG', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PAG_N', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if add_del_sod_ag:
            df[('CALCULATED', 'DEL_SOD_AG', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PAG_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if add_del_sod_mi:
            df[('CALCULATED', 'DEL_SOD_MI', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PMI_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PMI_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if add_del_nod_mi:
            df[('CALCULATED', 'DEL_NOD_MI', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PMI_N_WAMER', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PMI_N', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)
 
        if add_total_exports:
            df[('CALCULATED', 'TOTAL_EXPORTS', 'EXPORTS-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'C_DMC003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003_SWP', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003_CVP', 'FLOW-DELIVERY', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)
 
        if addsl:
            df[('CALCULATED', 'S_SLTOT', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

        if addpump:
            df[('CALCULATED', 'D_TOTAL', 'CHANNEL-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'C_DMC000', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if addres:
            df[('CALCULATED', 'S_RESTOT', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)
            df[('CALCULATED', 'S_RESTOT_NOD', 'STORAGE-CALC', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF') ]].sum(axis=1)

        if adddelcvp:
            df[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_TOTAL_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_TOTAL_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if adddelcvpswp:
            df[('CALCULATED', 'DEL_CVPSWP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_PAG_S', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if adddelcvpag:
            df[('CALCULATED', 'DEL_CVP_PAG_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if addcvpscex:
            df[('CALCULATED', 'DEL_CVP_PSCEX_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if addcvpprf:
            df[('CALCULATED', 'DEL_CVP_PRF_TOTAL', 'DELIVERY-CALC', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PRF_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PRF_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        new_columns = [(col[0], f'{col[1]}_{index_name[:]}', *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(new_columns)
        df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']
        combined_df = pd.concat([combined_df, df], axis=1)

    return combined_df

def preprocess_demands_deliveries(DemandFilePath, DemandFileTab, DemMin, DemMax, study_name, dvdss_name, svdss_name, datetime_start_date, datetime_end_date):
    # there's an excel file that contains a big table of demand unit information
    # we want the information relating to the deliveries and demands for each 
    # demand unit

    # set the filepath of the excel file
    xlfp = DemandFilePath


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
    all_du = all_du[all_du['Delivery_Variable']!='None']

    #%% prep and retrieve demand variables from SV (input) file

    # create a list of just the demands variables
    demands_list = []
    demands_dssvar_list = []
    for d in all_du.Demand_Variable:
        if d not in ['UD_EBMUD (calculated in WRESL)', '1911.5 TAF']: #skip including UD_EBMUD and the 1911.5 for now - we'll get those later
            demands_dssvar_list.append(d.strip())
            demands_list.append(d.strip())
    
    # most demadns are in the SV file, but there are a few in the DV file that need
    # to be extracted -  this next section divides variables into and SV and DV set
    this_dat = []
    sv_list = []
    dv_list = []
    for i in demands_list: #all_du.Demand_Variable.sort_values():
        
        if i[0:2]=='UD' or i.upper()=='AW_NIDDC_NA3' :
            # urban demands - SV list
            sv_list.append(i.upper())
        elif i.upper()=='DEM_D_CAA046_71_PA7_PIN':
            # for now, exclude 71_PA7 until we figure out how best to deal 
            # with SWP Table A, Carryover, and Interruptible demands tabulations
            pass
        else:
            dv_list.append(i.upper())     
            
    # ensure the sv_list doesn't ahve any duplicates - not sure why UD_25_PU keeps showing up twice
    sv_list = list(set(sv_list))

    # create a dataframe suitable for use with coeqwal functions
    demand_var_dv_df = pd.DataFrame(data=dv_list, columns=['Part B:'])   
    demand_var_dv_df['Part C:'] = [""]*len(demand_var_dv_df)     

    # print('demand_var_dv_df:')
    # print(demand_var_dv_df)

    demand_var_sv_df = pd.DataFrame(data=sv_list, columns=['Part B:'])   
    demand_var_sv_df['Part C:'] = [""]*len(demand_var_sv_df)   

    # print('demand_var_sv_df:')
    # print(demand_var_sv_df)

    # the dex.preprocess_study_dss function was modified to deal with a variable listing
    # that just has a "B-part" - consider adapting 
    demands_sv_df = preprocess_study_dss(demand_var_sv_df, sv_fp, datetime_start_date, datetime_end_date,
                                        addsl=False, addres = False, addpump = False, adddelcvp = False, 
                                        adddelcvpag = False, addcvpscex = False, addcvpprf = False, 
                                        adddelcvpswp = False, add_nod_storage = False, add_sod_storage = False, 
                                        add_del_nod_ag = False, add_del_nod_mi = False, add_del_sod_mi = False, 
                                        add_del_sod_ag = False, add_total_exports = False, 
                                        add_del_swp_total = False)
    # we can get the demands from the DV file too
    demands_dv_df = preprocess_study_dss(demand_var_dv_df, dv_fp, datetime_start_date, datetime_end_date,
                                        addsl=False, addres = False, addpump = False, adddelcvp = False, 
                                        adddelcvpag = False, addcvpscex = False, addcvpprf = False, 
                                        adddelcvpswp = False, add_nod_storage = False, add_sod_storage = False, 
                                        add_del_nod_ag = False, add_del_nod_mi = False, add_del_sod_mi = False, 
                                        add_del_sod_ag = False, add_total_exports = False, 
                                        add_del_swp_total = False)

    # combine the two demands files together
    demands_df = pd.concat([demands_sv_df, demands_dv_df], axis=1)

    # add a flat demand for MWD
    demands_df[('Manual-Add','TABLEA_CONTRACT_MWD','URBAN-DEMAND','1MON','L2020A','PER-CUM','TAF')] = len(demands_df)*[1911.5/12]
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
    delivs_list.append('DEL_SWP_MWD')   

    deliv_var_df = pd.DataFrame(data=delivs_list, columns=['Part B:'])
    deliv_var_df['Part C:'] = [""]*len(deliv_var_df)

    delivs_df = preprocess_study_dss(deliv_var_df, dv_fp, datetime_start_date, datetime_end_date,
                                        addsl=False, addres = False, addpump = False, adddelcvp = False, 
                                        adddelcvpag = False, addcvpscex = False, addcvpprf = False, 
                                        adddelcvpswp = False, add_nod_storage = False, add_sod_storage = False, 
                                        add_del_nod_ag = False, add_del_nod_mi = False, add_del_sod_mi = False, 
                                        add_del_sod_ag = False, add_total_exports = False, 
                                        add_del_swp_total = False)

    #%% check for missing data

    deliv_cols = list(delivs_df.columns.get_level_values('B'))

    missing_delivs = []
    for delvar in delivs_list:
        # print('Processing ' + delvar)
        if delvar.strip('/').upper() not in deliv_cols:
            missing_delivs.append(delvar)
            
    #%%  Delivery data - convert from CFS to TAF and write out to csvs

    # convert any CFS data into TAF, then aggregate to annual
    annual_del_data = []
    for c in delivs_df.columns:
        # print('Annualizing delivs_df column')
        dd =csplt.annualize(delivs_df.loc[:, [c]], on='YE-FEB', how='auto-sum') #contract years are Mar-Feb; the 'auto-sum' argument handles the CFS->TAF conversion
        # print('Appending annualized column')
        annual_del_data.append(dd)
    annual_del_df = pd.concat(annual_del_data, axis=1)
    # print('annual_del_df:')
    # print(annual_del_df)

    # # TODO: change output location to be in the "ExtractedData" folders within each scenario
    # ofp = os.path.join(outdir, f'{run_name_long}_DELIVERY-ANNUAL.csv') 

    # annual_del_df.to_csv(ofp, header=True) # write out annual totals to csv file

    mean_annual_del = annual_del_df.mean(axis=0) # caclualte the mean across all years
    # # TODO: change output location to be in the "ExtractedData" folders within each scenario
    # ofp2 = os.path.join(outdir,f'{run_name_long}_DELIVERY-ANNUAL-MEAN.csv')
    # mean_annual_del.to_csv(ofp2, header=True) # writes out a "mean of all years" file

    #%% Demand Data - the same CFS-to-TAF conversion and write out to CSV needs ot happen here

    annual_dem_data = []
    for c in demands_df.columns:
        # print('Annualizing demands_df column')
        dd =csplt.annualize(demands_df.loc[:, [c]], on='YE-FEB', how='auto-sum')
        # print('Appending annualized column')
        annual_dem_data.append(dd)
    annual_dem_df = pd.concat(annual_dem_data, axis=1)
    # print('annual_dem_df:')
    # print(annual_dem_df)

    # # TODO: change output location to be in the "ExtractedData" folders within each scenario
    # ofp = os.path.join(outdir, f'{run_name_long}_DEMANDS-ANNUAL.csv') 

    # annual_dem_df.to_csv(ofp, header=True) # write out annual totals to csv file

    mean_annual_dem = annual_dem_df.mean(axis=0) # caclualte the mean across all years
    # # TODO: change output location to be in the "ExtractedData" folders within each scenario
    # ofp2 = os.path.join(outdir,f'{run_name_long}_DEMANDS-ANNUAL-MEAN.csv')
    # mean_annual_dem.to_csv(ofp2, header=True) # writes out a "mean of all years" file

    return annual_dem_df, annual_del_df, mean_annual_dem, mean_annual_del