"""IMPORTS"""
import os
import datetime as dt
import pandas as pd
import numpy as np

# Import custom modules - NEED WINDOWS OS
import AuxFunctions as af, cs3, csPlots, cs_util as util, dss3_functions_reference as dss

"""READ NAMES & CONSTRUCT STUDY VARIABLES"""
def read_names(DssList, DssTab, DssMin, DssMax, hdr=True):
    dsshdr, dssname = af.read_from_excel(DssList, DssTab, DssMin, DssMax, hdr=True)
    dss_names = []
    for i in range(len(dssname)):
        dss_names.append(dssname[i][0] + DssExt)
    return dss_names

def abbrev_names(DssList, DssTab, AbbrMin, AbbrMax, hdr=True):
    abbrhdr, abbrname = af.read_from_excel(DssList, DssTab, AbbrMin, AbbrMax, hdr=True)
    abbr_names = []
    for i in range(len(abbrname)):
        abbr_names.append(abbrname[i][0])
    return abbr_names

def dss_names_to_csv(dss_names):
    dss_df = (pd.DataFrame(dss_names))
    return dss_df.to_csv(os.path.join(DataDir, DssOut))

"""READ VARIABLES"""
def get_var_df(VarList, VarTab,VarMin,VarMax,hdr=True):
    hdr, vars = af.read_from_excel(VarList, VarTab,VarMin,VarMax,hdr=True)
    var_df = pd.DataFrame(data=vars, columns=hdr)
    return var_df

def var_df_to_csv(var_df, DataDir,VarOut):
    var_df.to_csv(os.path.join(DataDir,VarOut))
    return VarOut

"""
CREATE DATASETS ACROSS STUDIES
(Aux functions to read and process studies (for single and multiple studies). Note: contain options to create hard-coded additional compound variables (if more are needed, add to these codes))
"""

def preprocess_study_dss(df, dss_name, datetime_start_date, datetime_end_date, addsl=True, addres = True, addpump = True, adddelcvp = True, adddelcvpag = True, addcvpscex = True, addcvpprf = True, adddelcvpswp = True, add_nod_storage = True, add_sod_storage = True, add_del_nod_ag = True, add_del_nod_mi = True, add_del_sod_mi = True, add_del_sod_ag = True, add_total_exports = True):
    dvar_list = []
    combined_df = pd.DataFrame()
    
    for i, r in df.iterrows():
        dvar_list.append(f'/{r["Part B:"]}/{r["Part C:"]}/')

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
        df[('CALCULATED', 'S_SLTOT', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

    if addpump:
        df[('CALCULATED', 'D_TOTAL', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'C_DMC000', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if addres:
        df[('CALCULATED', 'S_RESTOT', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)
        df[('CALCULATED', 'S_RESTOT_NOD', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF') ]].sum(axis=1)

    if adddelcvp:
        df[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_TOTAL_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_TOTAL_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if adddelcvpswp:
        df[('CALCULATED', 'DEL_CVPSWP_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_TOTA', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if adddelcvpag:
        df[('CALCULATED', 'DEL_CVP_PAG_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if addcvpscex:
        df[('CALCULATED', 'DEL_CVP_PSCEX_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    if addcvpprf:
        df[('CALCULATED', 'DEL_CVP_PRF_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PRF_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PRF_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

    new_columns = [(col[0], col[1], *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']

    df.head(5)
    return df


def preprocess_compound_data_dss(df, ScenarioDir, dss_names, index_names, min_datetime, max_datetime, addsl=True, addres = True, addpump = True, adddelcvp = True, adddelcvpag = True, addcvpscex = True, addcvpprf = True, adddelcvpswp = True, add_nod_storage = True, add_sod_storage = True, add_del_nod_ag = True, add_del_nod_mi = True, add_del_sod_mi = True, add_del_sod_ag = True, add_total_exports = True):
    dvar_list = []
    combined_df = pd.DataFrame()
    
    for i, r in df.iterrows():
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
            df[('CALCULATED', 'S_SLTOT', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_SLUIS_CVP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SLUIS_SWP', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)

        if addpump:
            df[('CALCULATED', 'D_TOTAL', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'C_DMC000', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'C_CAA003', 'CHANNEL', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)


        if addres:
            df[('CALCULATED', 'S_RESTOT', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_MELON', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_MLRTN', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')]].sum(axis=1)
            df[('CALCULATED', 'S_RESTOT_NOD', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF')] = df.loc[:,[('CALSIM', 'S_OROVL', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'),('CALSIM', 'S_SHSTA', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_TRNTY', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF'), ('CALSIM', 'S_FOLSM', 'STORAGE', '1MON', 'L2020A', 'PER-AVER', 'TAF') ]].sum(axis=1)

        if adddelcvp:
            df[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_TOTAL_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_TOTAL_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if adddelcvpswp:
            df[('CALCULATED', 'DEL_CVPSWP_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALCULATED', 'DEL_CVP_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_SWP_TOTA', 'DELIVERY-SWP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if adddelcvpag:
            df[('CALCULATED', 'DEL_CVP_PAG_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PAG_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PAG_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if addcvpscex:
            df[('CALCULATED', 'DEL_CVP_PSCEX_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PSC_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PEX_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        if addcvpprf:
            df[('CALCULATED', 'DEL_CVP_PRF_TOTAL', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')] = df.loc[:,[('CALSIM', 'DEL_CVP_PRF_N', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS'),('CALSIM', 'DEL_CVP_PRF_S', 'DELIVERY-CVP', '1MON', 'L2020A', 'PER-AVER', 'CFS')]].sum(axis=1)

        new_columns = [(col[0], f'{col[1]}_{index_name[:]}', *col[2:]) if len(col) > 1 else (col[0], '') for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(new_columns)
        df.columns.names = ['A', 'B', 'C', 'D', 'E', 'F', 'Units']
        combined_df = pd.concat([combined_df, df], axis=1)

    return combined_df