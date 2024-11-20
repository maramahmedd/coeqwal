# -*- coding: utf-8 -*-
"""
    Collection of objects used to organize, read, and write CalSim-related data
    
    Originally written for use with CalSim3, but should work with CalSimII as well
    
    Note - this is definitely a work-in-progress - functionality  may evolve,
    break, or work differently as things are developed.
    
    
"""
import os, sys
import copy


import pandas as pnd
idxslc = pnd.IndexSlice
import numpy as np
import datetime as dt
from collections import OrderedDict as Odict

import cs_util as util
import AuxFunctions as af
    
DEFAULTS = {
        'CSII': {
                'beginDate': dt.date(1921,10,31),
                'endDate': dt.date(2003,9,30),
                'Apt': 'CALSIM',
                'Ept': '1MON'},
                
        'CS3': {
                'beginDate': dt.date(1921,10,31),
                'endDate': dt.date(2015,9,30),
                'Apt': 'CALSIM',
                'Ept': '1MON'}}
        
class calsimhydro(object):
    
    def __init__(self,projDir=''):
        self.projDir = projDir
        self.IDC = None
        self.Rice = None
        self.Refuge = None
        self.Urban = None
        
        
    def read_MainRfro(self, mainfp):
        with open(mainfp, 'r') as mf:
            rl = mf.readlines()
        
        rlnc = [r for r in rl if r[0].upper()!='C']
        
        scenName = rlnc[0].split()[0].strip()
        
        lu_by_soil_fp = rlnc[1].split()[0].strip()
        param_fp = rlnc[2].split()[0].strip()
        precip_fp = rlnc[3].split()[0].strip()
        outMonPrecip_fp = rlnc[4].split()[0].strip()
        outEffPrecip_fp = rlnc[5].split()[0].strip()
        outUnitInf_fp = rlnc[6].split()[0].strip()
        
        beginDT =util.interp_csh_dt(rlnc[7].split()[0].strip())
        endDT = util.interp_csh_dt(rlnc[8].split()[0].strip())
        dssTimeUnitIN = rlnc[9].split()[0]
        dssTimeUnitOUT = rlnc[11].split()[0]
        
        NREGN = int(rlnc[12].split()[0])  # number of subregions
        NLUCodes = int(rlnc[13].split()[0])  # number of land use codes
        
        # get the land use codes
        lu_dict = Odict()  # dictionary to hold land use codes and any description that's provided
        for lu in range(NLUCodes):
            val, desc = rlnc[14+lu].split('/')
            tmp = desc.split()
#            if len(tmp)>1:
#                desc2 = desc.split()
#            else:
#                desc2 = [desc]
            lu_dict[val.strip()] = [desc.strip()] 
            
        # get the WBAs
        wba_list = []
        for w in range(NREGN):
            wba_list.append(rlnc[14+NLUCodes+w].split('/')[0].strip())
        
        return([scenName, lu_by_soil_fp, param_fp, precip_fp, outMonPrecip_fp,
                NREGN, NLUCodes, lu_dict, wba_list])
        
class idc:
    
    def __init__(self, mainfp):
        self.MainFilepath = mainfp
        self.BaseDir = os.path.dirname(os.path.dirname(mainfp))
        self.IDC_Dict = self.read_MainIDC(self.MainFilepath)
        self.Demand_Units = {}  # make this a dictionary of demand unit objects
        self.IDC_DU_Budget = None
    
    def read_MainIDC(self, mainfp):
        with open(mainfp, 'r') as mf:
            rl = mf.readlines()
            
        rlnc = [r for r in rl if r[0].upper()!='C']
                    
        idc_main_dict=Odict()
        idc_main_dict['InitCond_fp']= rlnc[0].split()[0].strip()
        idc_main_dict['Param_fp'] = rlnc[1].split()[0].strip()
        idc_main_dict['LandUse_fp'] = rlnc[2].split()[0].strip()
        idc_main_dict['OutputPaths_fp'] = rlnc[3].split()[0].strip()
        idc_main_dict['PrecipIn_fp'] = rlnc[4].split()[0].strip()
        idc_main_dict['InfiltPrecip_fp'] = rlnc[5].split()[0].strip()
        idc_main_dict['MinMoist_fp'] = rlnc[6].split()[0].strip()
        idc_main_dict['IrrEff_fp'] = rlnc[7].split()[0].strip()
        idc_main_dict['UrbanSpec_fp'] = rlnc[8].split()[0].strip()
        idc_main_dict['UrbanDem_fp'] = rlnc[9].split()[0].strip()
        idc_main_dict['ET_fp'] = rlnc[10].split()[0].strip()
        idc_main_dict['RetFlowFact_fp'] = rlnc[11].split()[0].strip()
        idc_main_dict['Reuse_fp']=rlnc[12].split()[0].strip()
        idc_main_dict['IDCoutput_fp'] = rlnc[13].split()[0].strip()
        idc_main_dict['IDCbudget_fp'] = rlnc[14].split()[0].strip()
        
        idc_main_dict['BeginDate'] =util.interp_csh_dt(rlnc[16].split()[0].strip())
        idc_main_dict['EndDate'] = util.interp_csh_dt(rlnc[17].split()[0].strip())
        idc_main_dict['DSSTimeUnit'] = rlnc[18].split()[0].strip()
        
        idc_main_dict['NREGN'] = int(rlnc[19].split()[0].strip())  #<-- number of demand units
        idc_main_dict['NCROP'] = int(rlnc[20].split()[0].strip())  #<-- number of crops
        idc_main_dict['KINFILT'] = int(rlnc[21].split()[0].strip())  #<-- infilt due to precip is computed (0) or is read in from file (1)
        
        idc_cropcodes = []
        for c in range(idc_main_dict['NCROP']):
            idc_cropcodes.append(rlnc[27+c].split()[0].strip())
        idc_cropcodes.append('UR') # add urban
        idc_cropcodes.append('NV') # add native vegetation
        idc_cropcodes.append('WL') # add wetlands
        idc_main_dict['CropCodes'] = idc_cropcodes
        
        idc_du = []
        for d in range(idc_main_dict['NREGN']):
            idc_du.append(rlnc[27+idc_main_dict['NCROP']+d].split()[0].strip())
            
        idc_main_dict['DemandUnits'] = idc_du
        
        return(idc_main_dict)
        
    def get_du_landuse_paths(self):
        lu_fp = self.IDC_Dict['LandUse_fp']
        
        with open(os.path.join(self.BaseDir, lu_fp),'r') as of:
            rl = of.readlines()
            
        rlnc = [r for r in rl if (r[0].upper()!='C')]
        rlnc = [r for r in rlnc if (r[0]!='*')]
        
        try:
            FACTLN = float(rlnc[0].split()[0].strip())
        except:
            FACTLN = None
            print("couldn't convert FACTLN to a float")
        DSSFL = rlnc[1].split()[0].strip()
        fullDSSFL = os.path.join(self.BaseDir, DSSFL)
        
        # assume the rest of the file is the listing of DU-crop index - DSS path info, one per line
        lu_path_dict = {}
        for r in range(2, len(rlnc)):
            
            iregn = rlnc[r].split()[0].strip()
            crtype = rlnc[r].split()[1].strip()
            
            try:
                iregn = int(iregn)
            except:
                print("Couldn't convert DU index into integer")
            
            try:
                crtype = int(crtype)
            except:
                print("couldn't convert crop type index into integer")
            
            cpath = rlnc[r].split()[2].strip()
            lu_path_dict[(iregn, crtype)] = cpath
        
        return([FACTLN, DSSFL,fullDSSFL, lu_path_dict])

    def get_irreff_paths(self):
        irreff_fp = self.IDC_Dict['IrrEff_fp']
        
        with open(os.path.join(self.BaseDir, irreff_fp),'r') as of:
            rl = of.readlines()
            
        rlnc = [r for r in rl if (r[0].upper()!='C')]
        rlnc = [r for r in rlnc if (r[0]!='*')]
        
        DSSFL = rlnc[0].split()[0].strip()
        fullDSSFL = os.path.join(self.BaseDir, DSSFL)
        
        # assume the rest of the file is the listing of DU-crop index - DSS path info, one per line
        irreff_path_dict = {}
        for r in range(2, len(rlnc)):
            
            iregn = rlnc[r].split()[0].strip()
            crtype = rlnc[r].split()[1].strip()
            
            try:
                iregn = int(iregn)
            except:
                print("Couldn't convert DU index into integer")
            
            try:
                crtype = int(crtype)
            except:
                print("couldn't convert crop type index into integer")
            
            cpath = rlnc[r].split()[2].strip()
            irreff_path_dict[(iregn, crtype)] = cpath
        
        return([DSSFL,fullDSSFL, irreff_path_dict])
       

    def init_DU_list(self):
        # check if you have a listing of demand units, then loop through
        # and create demand unit objects to go wtih them
        if len(self.IDC_Dict['DemandUnits'])==0:
            return
        for du in self.IDC_Dict['DemandUnits']:
            self.Demand_Units[du]= demand_unit(duID=du)
    
    def get_land_use(self, land_use_startDate = dt.datetime(4000, 1,31,0,0),
                     isHistorical=False,
                     checkWetlands=True,
                     ctime24=True,
                     ntimes=12):
        
        [FACTLN, DSSFL, fullDSSFL, lu_path_dict] = self.get_du_landuse_paths()

        luDSS = util.dssFile(fp=fullDSSFL)
        luDSS.get_cat_paths()
                
        for (iregn, crtype) in lu_path_dict: # iterate through all demand unit/land use type combinations
            du = self.IDC_Dict['DemandUnits'][iregn-1]
            luID = self.IDC_Dict['CropCodes'][crtype-1]
            cpath = lu_path_dict[(iregn, crtype)]
            
            thisvar = util.dssVar(luDSS, cpath=cpath)
            
            if isHistorical:
                thisvar.getRTS(dt.datetime(1920,10,31,0,0),ctime24=True, ntimes=1248)
                thisLU = thisvar.RTS
            else:
                thisvar.getRTS(land_use_startDate,ctime24=True, ntimes=12)
                thisLU = thisvar.RTS.iloc[0]
            
            
            self.Demand_Units[du].LandUse[luID] = [thisLU, thisvar.Units, thisvar.RecordType]
            
        if checkWetlands:
            # check if there is any data for wetland areas in each DU
            dus = list(set([t[0] for t in lu_path_dict.keys()]))
            for iregn in dus:
                du = self.IDC_Dict['DemandUnits'][iregn-1]
                luID = 'WL'
                cpath = '/CALSIM/%s_WL/LANDUSE//1MON/EXISTING/' %du
                
                thisvar = util.dssVar(luDSS, cpath=cpath)
                thisvar.getRTS(land_use_startDate, ctime24=True, ntimes=12)
                thisLU = thisvar.RTS.iloc[0]
                self.Demand_Units[du].LandUse[luID] = [thisLU, thisvar.Units, thisvar.RecordType]
                
            
        luDSS.closeDSS()
        
    def get_irr_eff(self, irreff_startDate = dt.datetime(4000, 1,31,0,0),
                     ctime24=True,
                     ntimes=12):
        
        [DSSFL, fullDSSFL, path_dict] = self.get_irreff_paths()

        irreffDSS = util.dssFile(fp=fullDSSFL)
        #irreffDSS.get_cat_paths()
                
        for (iregn, crtype) in path_dict: # iterate through all demand unit/land use type combinations
            du = self.IDC_Dict['DemandUnits'][iregn-1]
            luID = self.IDC_Dict['CropCodes'][crtype-1]
            cpath = path_dict[(iregn, crtype)]
            
            thisvar = util.dssVar(irreffDSS, cpath=cpath)
            
            thisvar.getRTS(irreff_startDate,ctime24=True, ntimes=12)
                        
            thisIrrEff= thisvar.RTS
            self.Demand_Units[du].IrrigEff[luID] = [thisIrrEff, thisvar.Units, thisvar.RecordType]
            
        irreffDSS.closeDSS()       
        
        
    def get_idc_params(self):
        
        paramFP = self.IDC_Dict['Param_fp']
        
        with open(os.path.join(self.BaseDir, paramFP),'r') as of:
            rl = of.readlines()
            
            rlnc = [r for r in rl if (r[0].upper()!='C')]
            rlnc = [r for r in rlnc if (r[0]!='*')]
            rlnc = [r for r in rlnc if r.strip()!='']
            nregn =self.IDC_Dict['NREGN']
            ncrop = self.IDC_Dict['NCROP']
            nlu = ncrop+2
            grp = 0
            # first group is field capacity table - cols = crops (1-22); rows=du/iregn
            fcdf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn],columns=self.IDC_Dict['CropCodes'])
            # then (ef)fective porosity
            grp =1 
            efdf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], columns = self.IDC_Dict['CropCodes'])
            #then curve number - although this is not used in the idc implementation in calsimhydro
            grp=2
            cndf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], columns = self.IDC_Dict['CropCodes'])
            
            grp = 3
            # no native vegetation in the return flow specification, adjust columns appropariately
            noNVcols = [cr for cr in self.IDC_Dict['CropCodes'] if cr != 'NV']
            icrfdf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], dtype='i',columns = noNVcols)
            
            grp=4
            # water use parameters - indexes for where to get reuse data/info
            noURNVcols = [cr for cr in self.IDC_Dict['CropCodes'] if cr not in ['UR', 'NV']]
            wuparcols = ['PERV', 'ICPRECIP'] + noURNVcols + ['ICRUFURB', 'FURDPRF','FURDPSR']
            wupardf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], dtype='f',columns = wuparcols)
            
            grp=5
            icinfilt = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], dtype='i',columns = self.IDC_Dict['CropCodes'])
            endidx5 = (grp+1)*nregn
            
            # rooting depths by crop
            grp=6 
            rootdepConvFact = float(rlnc[endidx5].split()[0].strip())
            rootdepDF = read_text_array(rlnc[endidx5+1:endidx5+1+nlu], dtype='f',columns=['RootingDepth_ft'])        
            
            # add the fc, ef, rootdep data to demand unit objects
            for idx,row in fcdf.iterrows():
                du = self.IDC_Dict['DemandUnits'][idx-1]
                self.Demand_Units[du].FieldCapacity = row
#                for luID in row.index:
#                    self.Demand_Units[du].FieldCapacity[luID] = row[luID]

            for idx,row in efdf.iterrows():
                du = self.IDC_Dict['DemandUnits'][idx-1]
                self.Demand_Units[du].Porosity= row
#                for luID in row.index:
#                    self.Demand_Units[du].FieldCapacity[luID] = row[luID]
                    
            # rooting depth is defined just for crop, not DU, but assigning it
            # by DU for completeness
            for du in self.Demand_Units:
                for idx, row in rootdepDF.iterrows():
                    luID = self.IDC_Dict['CropCodes'][idx-1]
                    for rd in row.index:
                        self.Demand_Units[du].RootingDepth[luID]= [row[rd]*rootdepConvFact, 'feet']
                        
                
    def get_et_paths(self):
        
        et_fp = self.IDC_Dict['ET_fp']
    
        with open(os.path.join(self.BaseDir, et_fp),'r') as of:
            rl = of.readlines()
            
        rlnc = [r for r in rl if (r[0].upper()!='C')]
        rlnc = [r for r in rlnc if (r[0]!='*')]
        rlnc = [r for r in rlnc if r.split()!='']
        
        etfact = float(rlnc[0].split()[0].strip())
        DSSFL = rlnc[1].split()[0].strip()
        fullDSSFL = os.path.join(self.BaseDir, DSSFL)
        
        # assume the rest of the file is the listing of DU-crop index - DSS path info, one per line
        et_path_dict = {}
        
        for r in range(2, len(rlnc)):
            
            iregn = rlnc[r].split()[0].strip()
            crtype = rlnc[r].split()[1].strip()
            
            try:
                iregn = int(iregn)
            except:
                print("Couldn't convert DU index into integer")
            
            try:
                crtype = int(crtype)
            except:
                print("couldn't convert crop type index into integer")
            
            cpath = rlnc[r].split()[2].strip()
            et_path_dict[(iregn, crtype)] = cpath
        
        return([DSSFL,fullDSSFL, et_path_dict, etfact])
        
    
    def get_et_ts(self, et_startDate = None, et_endDate = None, ctime24=True):
    
        [DSSFL, fullDSSFL, path_dict, etfact] = self.get_et_paths()

        if et_startDate==None:
            et_startDate = self.IDC_Dict['BeginDate']
        if et_endDate ==None:
            et_endDate = self.IDC_Dict['EndDate']

        if len(path_dict)>100:
            print("lookout...just about to read in a bunch (%s) of DSS time series..." %len(path_dict))

        etDSS = util.dssFile(fp=fullDSSFL)

        i = 0
        
        for (iregn, crtype) in path_dict: # iterate through all demand unit/land use type combinations
            
            du = self.IDC_Dict['DemandUnits'][iregn-1]
            
            # if an extra 'bare soil' category is in the ET data, add it to
            # the scheme here manually
            if crtype>len(self.IDC_Dict['CropCodes']):
                luID = 'SL'
            else:
                luID = self.IDC_Dict['CropCodes'][crtype-1]
            
            cpath = path_dict[(iregn, crtype)]
            
            i +=1
            # provide a progress bar in the console
            util.progress(i, len(path_dict), status='Retrieving %s ET Time Series' %len(path_dict))
            
            thisvar = util.dssVar(etDSS, cpath=cpath)
            
            thisvar.getRTS(et_startDate,ctime24=True, endDateTime=et_endDate)
            
            thisET= thisvar.RTS
            
            if thisvar.Units.upper() in ['IN/MONTH','IN', 'INCH', 'INCHES']:
                thisET = thisET*etfact
                newUnits = 'FT/MONTH'
            else:
                newUnits = thisvar.Units
                        
            
            self.Demand_Units[du].ET_ts[luID] = [thisET, newUnits, thisvar.RecordType]
            
        etDSS.closeDSS()   
        
        
    def get_IDC_DU_budget(self, du=['all'], variables=['all'],cropType=['all'],
                          startDate=None,
                          endDate=None):
        # TODO: make start/end date variable inputs kwargs; make consistent across all functions
        # TOOD: build a datetime index automatically from start/end dates
        
        if startDate==None:
            startDate = self.IDC_Dict['BeginDate']
        if endDate ==None:
            endDate = self.IDC_Dict['EndDate']
        
        if 'all' in du:
            dus = self.Demand_Units
        else:
            dus = {x: v for x, v in self.Demand_Units.items() if x in du}
            
        idcbudFP = os.path.join(self.BaseDir, self.IDC_Dict['IDCbudget_fp'])
        #print(idcbudFP)
        idcbudDSS = util.dssFile(fp=idcbudFP)
        
        if 'all' in variables:
            budvarlist = ['AREA','BEGIN_STOR','DEEP_PERC','DISCREPANCY','END_STOR',
                          'ET', 'GAIN_EXP','INFILTR','INFILTR_CHECK', 'PRECIP',
                          'PRM_H2O','RE-USE','RTRN_FLOW','RUNOFF','TOTAL_APP']
        else:
            budvarlist = variables
        
        
        if 'all' in cropType:
            all_crop_codes = [l for l in self.IDC_Dict['CropCodes'] if l !='WL']
        else:
            all_crop_codes = cropType
        
        midx = pnd.MultiIndex.from_product([list(dus.keys()), budvarlist])
        df = pnd.DataFrame(columns=midx)
        for d in dus:  # loop through all demand units
            for bv in budvarlist:  # loop through the budget variables
                
                for n,crp in enumerate(all_crop_codes):  #loop through the crop types
                    
                    if (crp=='NV') & (bv in ['PRM_H2O','RE-USE','RTRN_FLOW', 'TOTAL_APP']):
                        continue
                    
                    if bv=='AREA':
                        cpt = 'AREA'
                    else:
                        cpt = 'VOLUME'
                    cpath = '/IDC_BUDGET/%s_%s/%s//1MON/%s/' %(d,crp,cpt,bv)
                    
                    thisvar = util.dssVar(idcbudDSS, cpath=cpath)
                    
                    thisvar.getRTS(startDate,ctime24=True, endDateTime=endDate)
                    
                    thisdat = thisvar.RTS
                    
                    if thisvar.Units.upper() in ['AF','AC.FT.','AC-FT','A.F.']:
                        thisdat = thisdat/1000.
                        newUnits = 'TAF'
                    else:
                        newUnits = thisvar.Units
                    
                    if thisvar.istat == 5:
                        print("No records were found (data returned is all 902's")
                        continue
                    if n==0:
                        varTot = thisdat
                    else:
                        varTot = varTot+thisdat
                    
                df.loc[:,(d,bv)] = varTot
        
        idcbudDSS.closeDSS()
        
        return(df)
                

class wba(object):
    
    
    def __init__(self, wbaID='',wbaDesc=''):   #, projDir='', lookupDSS=''):
        self.ProjectName = wbaID
        self.Description = wbaDesc
        self.Area = np.nan
        self.PrecipMonTS = None
        self.PrecipDlyTS = None
        self.IrrEff = {}
        self.DemandUnits = {}
        self.Precipitation = None
#        self.Basin = [] #basin()
#        self.Control = [] # control('Decembruary 9999','ThisIsAControlFile.control','I got nothing....')
#        self.ProjDir = projDir
#        self.LookupDSSFile = lookupDSS
#        self.Version = '4.2.1'
#        self.TimeZoneID = 'America/Los_Angeles'
#        self.FilepathSep = '\\'
#        self.Grids = []
#        self.GridInfo = {}
#        self.Events = []
#        self.Templates = {'met':'','basin':'','hms':'','grid':'', 'bat':''}
#        self.Runs = {}
        
    #def get_

#class precip:
#    
#    def __init__
    
    
class demand_unit:
    
    def __init__(self, duID='',duDesc=''):
        self.DemandUnitID = duID
        self.Description = duDesc
        self.Boundary = None
        self.LandUse = {}
        self.IrrigEff = {}
        self.FieldCapacity = None
        self.Porosity = None
        self.RootingDepth = {}
        self.ReturnFlowFraction = None
        self.ReuseFraction = None
        self.ET_ts = {}
        
    
    def report_du_crop(self, cropID):
        thisLU = self.LandUse[cropID]
        thisIrrEff = self.IrrigEff[cropID]
        fc = self.FieldCapacity[cropID]
        ef = self.Porosity[cropID]
        rd = self.RootingDepth[cropID]
        print("\n===============")
        print("Demand Unit: %s" %self.DemandUnitID)
        print("=================")
        print("Land Use Type: %s - %0.2f %s" %(cropID ,thisLU[0], thisLU[1]))
        print("Average irrigation efficiency: %0.2f" %(np.max(thisIrrEff[0])))
        print("Field capacity: %0.2f" %(fc,))
        print("Porosity: %0.2f" %(ef,))
        print("Rooting depth: %0.2f %s" %(rd[0],rd[1]))
        
        
    
class csh_cropPractice(object):

    def __init__(self, cropName=''):
        self.cropName = cropName
        self.Description= None
        self.IrrEffDict = {}
        self.MinSMDict = {}
        
        
    def getCalYrMonths(startMonth=1):
        mths = []
        for m in range(12):
            #thism = min(m+startMonth-12, )
            if m+startMonth>12:
                thism = m+startMonth-12
            else:
                thism = m+startMonth
            mths.append(thism)
            
        return(mths)

def read_text_array(rl, dtype='f', **kwargs):
    
    if 'columns' in kwargs:
        colnames = kwargs['columns']
        nc = len(colnames)
    else:
        nc=None
    
    idx = []
    rows=[]
    for r in rl:
        tmp = r.split()
        if nc!=None:
            tmp2 = [t for t in tmp[1:1+nc]]
            tmp = [tmp[0]] + tmp2
        idx.append(int(tmp[0]))
        if dtype=='f':
            rows.append([float(t) for t in tmp[1:]])
        elif dtype=='i':
            rows.append([int(t) for t in tmp[1:]])
        else:
            rows.append([t for t in tmp[1:]])
        
    if 'columns' in kwargs:
        colnames = kwargs['columns']
        nc = len(colnames)
        df = pnd.DataFrame(index=idx, data=rows, columns=colnames)
    else:
        df = pnd.DataFrame(index=idx, data=rows)
    return(df)   


def read_CSlookup(tablefp, coltypes=None,commentChar='!'):
    with open(tablefp, 'r') as of:
        rl = of.readlines()
        
    rlnc = [r for r in rl if r[0]!=commentChar]
    rlnc = [r for r in rlnc if r.strip()!='']
    tabName = rlnc[0].strip()
    tabCols = rlnc[1].strip().split()
    tabCols.append('Notes')  # make a spot for saving notes/info/comments
    tmp = []

    for l in rlnc[2:]:
        if '!' in l:
            data1, cmnt = l.strip().split('!')
            data = data1.split() 
        else:
            cmnt=''
            data_tmp = l.strip().split()
            data = []
            
            if coltypes==None:
                # try to enforce some sort of numeric typing
                for n,d in enumerate(data_tmp):
                    try:
                        if '.' in d:  
                            data.append(float(d))
                        else:
                            data.append(int(d))
                    except:
                        data.append(str(d))
            else:
                if type(coltypes)==dict:
                    print("Column dtype assignment by dictionary not implemented yet! Sorry!!!")
                    return(None)
                
                if type(coltypes)==list:
                    if len(coltypes)!=len(data_tmp):
                        print("coltypes specifications doesn't match the table at line:\n %s" %l)
                    else:
                        for n,d in enumerate(zip(data_tmp,coltypes)):
                            if d[1]=='f':
                                data.append(float(d[0]))
                            elif d[1]=='i':
                                data.append(int(d[0]))
                            elif d[1]=='s':
                                data.append(str(d[0]))
                            else:
                                data.append(str(d[0]))
                        
         
        data.append(cmnt)
        tmp.append(data)
        
#    if notesCol:
#        tabCols.append('Notes')
    tabdf = pnd.DataFrame(tmp, columns=tabCols)
    
    return(tabdf)
    
    
def write_CSlookup(df, tablefp, hdr_notes="", overwrite=False, indxCol='wateryear'):
    
    if os.path.exists(tablefp) and not overwrite:
        print("Table file already exists at %s" %tablefp)
        print("Set overwrite=True to overwrite this file, otherwise nothing is done")
        return
    
    fn = os.path.basename(tablefp)
    tablname = os.path.splitext(fn)[0]
    colsep = '\t'
    #colsep = ' '*8
    with open(tablefp, 'w') as of:
        if hdr_notes[0] != '!':
            hdr_notes = '!' + hdr_notes
        if hdr_notes[-1] != "\n":
            hdr_notes = hdr_notes + "\n"
        of.write(hdr_notes)
        of.write(tablname+"\n")
        collin = ''
        for c in df.columns:
#            if type(c)==tuple:
                
            collin += c + colsep
        collin += "\n"
        of.write(collin)
        
        for idx, r in df.iterrows():
            rv = r.values
            rowlin = ''
            if indxCol !='':
                if indxCol.upper() in ['WATERYEAR','WY','YEAR','YR']:
                    rowlin = "%d" %(rv[0],) + colsep
                    for rvi in rv[1:]:
                        rowlin += "{:g}".format(rvi) + colsep
                else:
                    for rvi in rv:
                        rowlin += "{:g}".format(rvi) + colsep
            else:
                for rvi in rv:
                    rowlin += "{:g}".format(rvi) + colsep                
            rowlin += "\n"
            of.write(rowlin)
                   
    



class calsim():
    
    def __init__(self, **kwargs): #launchFP):
        
        self.MainFile = ''
        self.Solver = 'XA'
        self.DvarFile = ''
        self.SvarFile = ''
        self.GroundwaterDir = ''
        self.SvarAPart = 'CALSIM'
        self.SvarFPart = '2020D09E'
        self.InitFile = ''
        self.InitFPart = '2020D09E'
        self.TimeStep = '1MON'
        self.StartYear = 1921
        self.StartMonth = 10
        self.StartDay = 31
        self.StopYear = 2003
        self.StopMonth = 9
        self.StopDay = 30
        self.IlpLog = 'No'
        self.IlpLogFormat = 'None'
        self.IlpLogVarValue = 'No'
        self.IlpLogAllCycles = 'No'
        self.WreslPlus = 'no'
        self.AllowSvTsInit = 'no'
        self.DatabaseURL = 'none'
        self.SQLGroup = 'calsim'
        self.OVOption = 0
        self.OVFile = '.'
        self.OutputCycleDatatoDss = 'no'
        self.OutputAllCycleData = 'no'
        self.SelectedCycleOutput = [8,9,10,11]
        self.Reorg=False
        self.Version = '3'
        
        if 'launchFP' in kwargs:
            if 'reorg' in kwargs:
                self.Reorg = kwargs['reorg']
                
            self.LaunchFP = kwargs['launchFP']
            self.LaunchDict = self.getLaunchDict()
            self.projDir = self.getProjDir()
            print(f'Project directory set as: {self.projDir}\n')
            self.getIOinfo()
    
            self.Years = Odict()
            if self.Reorg:
                self.TableDir = os.path.join(self.getProjDir(), 'Run', 'Lookup')
            else:
                self.TableDir = os.path.join(self.getProjDir(), 'CONV', 'Run', 'Lookup')
            
            self.SVdata = csSVdata(self)
            self.DVdata = csDVdata(self)
            
            self.Tables = None
            #self.SVtsDF =  None       
            
        elif 'configFP' in kwargs:
            # TODO: add function to read in config file
            return
        
        elif 'launchDict' in kwargs: # if a specified launch dictionary is provided
            if 'launchFP' in kwargs:
                self.LaunchFP = kwargs['launchFP']
            else:
                self.LaunchFP = ''
            
            if 'calsim_version' in kwargs:
                csversion = kwargs['calsim_version']
            else:
                csversion = 3
                
            # initialize the dictionary so at least we have all the keys
            self.initLaunchDict(csversion=csversion)
            
            # now overwrite the key/values with any info provided
            for k in kwargs['launchDict']:
                self.LaunchDict[k] = kwargs['launchDict'][k]
             
            # TODO: low priority - write out new launch file from dictionary 
#            if self.LaunchFP!=None:
#                if not os.path.exists(self.LaunchFP):
#                    with open(self.LaunchFP,'w')
            
            self.projDir = self.getProjDir()
            self.getIOinfo()
    
            self.Years = Odict()
    
            self.SVdata = csSVdata(self)
            self.DVdata = csDVdata(self)
            
        elif 'temp' in kwargs:  # for cases where it may be helpful to have a temporary calsim object (reading/writing DSS files, for example)
                                # and just need some of the info associated with a calsim study without needing to run anything
            csvers = kwargs['temp']
            if csvers.upper() in ['3','CS3','CALSIM3']:
                csversion=3
            else:
                csversion=2
                
            self.initLaunchDict(csversion=csversion)
            self.StartDate = dt.datetime(self.StartYear, self.StartMonth, self.StartDay,0,0)
            self.EndDate = dt.datetime(self.StopYear, self.StopMonth, self.StopDay,0,0)

        
    def getLaunchDict(self):
        if os.path.exists(self.LaunchFP):
            import xml.etree.ElementTree as ET

            
            tree = ET.parse(self.LaunchFP)
            root = tree.getroot()

            launchDict = {}
            for elem in root:
                k = elem.attrib['key']
                if 'wpp.debugModel.ATTR_WPP_' in k:
                    k = k.replace('wpp.debugModel.ATTR_WPP_','')
                try:
                    v = elem.attrib['value']
                except: # no value with key
                    v = None
                launchDict[k]=v
            
            return(launchDict)
        else:
            return(None)
            
    def initLaunchDict(self, csversion=2):
        if csversion==3:
            endday = '30'
            endmonth = 'sep'
            endyear='2015'
            wreslplus='yes'
        else:
            endday = '30'
            endmonth = 'sep'
            endyear='2003'
            wreslplus='no'
            
        self.LaunchDict = {
                'ALLOWSVTSINIT': '',
                'APART': 'CALSIM',
                'APART_MS2': '',
                'AUTHOR': '',
                'DATATRANSFER_MS2': '',
                'DATE': dt.datetime.strftime(dt.datetime.today(),'%Y-%m-%d'),
                'DESCRIPTION': '',
                'DVAR': 'CONV\\DSS\\2020D09EDV.dss',
                'DVAR_MS2': '',
                'ENDDAY': endday,
                'ENDMONTH': endmonth,
                'ENDYEAR': endyear,
                'FREEXA': 'no',
                'GWDATA': '',
                'GWDATA_MS2': '',
                'INIT': 'common\\DSS\\2020D09EINIT.dss',
                'INITFPART': '2020D09E',
                'INITFPART_MS2': '',
                'INIT_MS2': '',
                'ISFIXDURATION': 'yes',
                'LAUNCHTYPE': '0',
                'MSDURATION': '12',
                'MULTISTUDY': '1',
                'PADELINIT': 'yes',
                'PADURATION': '12',
                'PADVSTARTDAY': '1',
                'PADVSTARTMONTH': '10',
                'PADVSTARTYEAR': '2014',
                'PARESETDVSTART': 'no',
                'PASTARTINTERVAL': '12',
                'PROGRAM': 'CONV\\Run\\mainCONV_SA.wresl',
                'PROGRAM_MS2': '',
                'STARTDAY': '31',
                'STARTMONTH': 'oct',
                'STARTYEAR': '1921',
                'STUDY': '',
                'SVAR': 'common\\DSS\\2020D09ESV.dss',
                'SVAR_MS2': '',
                'SVFPART': '2020D09E',
                'SVFPART_MS2': '',
                'TIMESTEP': '1MON',
                'TIMESTEP_MS2': '1MON',
                'VARIABLEDURATION': '',
                'WRESLPLUS': wreslplus                           
                }
            
    def getProjDir(self):
        if self.LaunchDict is not None:
            projDir = os.path.dirname(self.LaunchFP)
            return(projDir)
        else:
            return(None)
            
    def getIOinfo(self):
        ''' parse out the input, output files, path conventions
           from the launch file dictionary
        '''
        self.DV_FP = os.path.join(self.projDir, self.LaunchDict['DVAR'])
        self.SV_FP = os.path.join(self.projDir, self.LaunchDict['SVAR'])
        self.DV_A_Part = self.LaunchDict['APART']
        self.F_Part = self.LaunchDict['SVFPART']
        self.TimeStep = self.LaunchDict['TIMESTEP']
        self.StartDay = int(self.LaunchDict['STARTDAY'])
        self.StartMonth = dt.datetime.strptime(self.LaunchDict['STARTMONTH'],'%b').month
        self.StartYear = int(self.LaunchDict['STARTYEAR'])
        self.StartDate = dt.datetime(self.StartYear, self.StartMonth, self.StartDay,0,0)
        self.EndDay = int(self.LaunchDict['ENDDAY'])
        self.EndMonth = dt.datetime.strptime(self.LaunchDict['ENDMONTH'],'%b').month
        self.EndYear = int(self.LaunchDict['ENDYEAR'])
        self.EndDate = dt.datetime(self.EndYear, self.EndMonth, self.EndDay)
        
        self.MainFile = os.path.join(self.projDir, self.LaunchDict['PROGRAM'])
        self.DvarFile = self.DV_FP
        self.SvarFile = self.SV_FP
        self.GroundwaterDir = os.path.join(self.projDir, self.LaunchDict['GWDATA'])
        self.SvarAPart = 'CALSIM'
        self.SvarFPart = self.LaunchDict['SVFPART']
        self.InitFile = os.path.join(self.projDir, self.LaunchDict['INIT'])
        self.InitFPart = self.LaunchDict['INITFPART']
        self.TimeStep = self.LaunchDict['TIMESTEP']
        #self.StartYear = int(self.LaunchDict['STARTYEAR'])
        #self.StartMonth = int(self.LaunchDict['STARTMONTH'])
        #self.StartDay = int(self.LaunchDict['STARTDAY'])
        self.StopYear = self.EndYear #int(self.LaunchDict['ENDYEAR'])
        self.StopMonth = self.EndMonth #int(self.LaunchDict['ENDMONTH'])
        self.StopDay = self.EndDay #int(self.LaunchDict['ENDDAY'])
        
        if not os.path.exists(self.DV_FP):
            if self.Reorg:
                oldpath, relpath = self.LaunchDict['DVAR'].upper().split('\\DSS\\')
            else:
                oldpath, relpath = self.LaunchDict['DVAR'].split('CONV')
#            if relpath[0:2]=='\\':
#                relpath = relpath[3:]
            relpath = relpath.replace("\\","/")
            relpath = r'%s' %relpath #relpath.encode('UTF-8')
            #print(relpath)
            #print(os.path.dirname(self.LaunchFP))
            if oldpath=='': # relative path provided in launch file
                self.DV_FP = self.projDir + '\\DSS\\' + self.LaunchDict['DVAR'].split('\\DSS\\')[-1]  #self.LaunchDict['DVAR'].replace(oldpath, self.projDir + "\\")
            else:
                self.DV_FP = self.LaunchDict['DVAR'].replace(oldpath, self.projDir + "\\") #os.path.join(os.path.dirname(self.LaunchFP), relpath)
            if not os.path.exists(self.DV_FP):
                print("Couldn't find the DV file at %s" %self.DV_FP)
    
        if not os.path.exists(self.SV_FP):
            if self.Reorg:
                oldpath, relpath = self.LaunchDict['SVAR'].upper().split('\\DSS\\')
            else:
                oldpath, relpath = self.LaunchDict['SVAR'].split('CONV')
#            if relpath[0:2]=='\\':
#                relpath = relpath[3:]
            relpath = relpath.replace("\\","/")
            relpath = r'%s' %relpath #relpath.encode('UTF-8')
            #print(relpath)
            #print(os.path.dirname(self.LaunchFP))
            if oldpath=='': # relative path provided in launch file
                self.SV_FP = self.projDir + '\\DSS\\' + self.LaunchDict['SVAR'].split('\\DSS\\')[-1]  #self.LaunchDict['DVAR'].replace(oldpath, self.projDir + "\\")
            else:
                self.SV_FP = self.LaunchDict['SVAR'].replace(oldpath, self.projDir + "\\") #os.path.join(os.path.dirname(self.LaunchFP), relpath)
            if not os.path.exists(self.SV_FP):
                print("Couldn't find the SV file at %s" %self.DV_FP)
    
    def write_launch_file(self, new_launch_fp,**kwargs):

        if 'SV_path' in kwargs:
            sv_path = kwargs['SV_path']
        else:
            sv_path = self.LaunchDict['SVAR']
            
        if 'DV_path' in kwargs:
            dv_path = kwargs['DV_path']
        else:
            dv_path = self.LaunchDict['DVAR']
            
        if 'Init_path' in kwargs:
            init_path = kwargs['Init_path']
        else:
            init_path = self.LaunchDict['INIT']
            
        if 'Main_path' in kwargs:
            main_path = kwargs['Main_path']
        else:
            main_path = self.LaunchDict['PROGRAM']
            
        if 'Study_Name' in kwargs:
            study_name = kwargs['Study_Name']
        else:
            study_name = 'auto_calsim_run'
            
        with open(new_launch_fp, 'w') as nfp:
            nfp.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
            nfp.write('<launchConfiguration type="wpp.launchType">\n')
            for k,v in self.LaunchDict.items():
                if k.upper =='SVAR':
                    v= sv_path
                if k.upper=='DVAR':
                    v= dv_path
                if k.upper=='INIT':
                    v= init_path
                if k.upper=='PROGRAM':
                    v = main_path
                    
                if k.upper=="STUDY":
                    v = study_name
                    
                if k=="IFS_NUMBER_SELECT_ENTRIES":
                    s = f'<intAttribute key="wpp.debugModel.ATTR_WPP_{k.upper()}" value="{v}"/>\n'
                else:
                    s = f'<stringAttribute key="wpp.debugModel.ATTR_WPP_{k.upper()}" value="{v}"/>\n'
                nfp.write(s)
            nfp.write("</launchConfiguration>")
                
    def write_config(self, **kwargs):
        
        '''
            write out the config file used when calling WRIMS - info should match
            what is in the launch file used in the WRIMS GUI
        '''
        
        if 'main_dir' in kwargs:
            main_dir = kwargs['main_dir']
        else:
            main_dir = self.projDir
        hdr = '##############################################################################\n'
        hdr += '# this config file automatically written using a python script\n'
        hdr += '#\n'
        hdr += '##############################################################################\n\n\n'
        
        ot = hdr
        
        ot += f'{"MainFile": <20}{self.MainFile}\n'
        ot += f'{"Solver" : <20}{self.Solver}\n'
        ot += f'{"DvarFile": <20}{self.DvarFile}\n'
        ot += f'{"SvarFile": <20}{self.SvarFile}\n'
        ot += f'{"GroundwaterDir": <20}{self.GroundwaterDir}\n'
        ot += f'{"SvarAPart": <20}{self.SvarAPart}\n'
        ot += f'{"SvarFPart": <20}{self.SvarFPart}\n'
        ot += f'{"InitFile": <20}{self.InitFile}\n'
        ot += f'{"InitFPart": <20}{self.InitFPart}\n'
        ot += f'{"TimeStep ": <20}{self.TimeStep }\n'
        ot += f'{"StartYear ": <20}{self.StartYear }\n'
        ot += f'{"StartMonth ": <20}{self.StartMonth }\n'
        ot += f'{"StartDay ": <20}{self.StartDay }\n'
        ot += f'{"StopYear ": <20}{self.StopYear }\n'
        ot += f'{"StopMonth ": <20}{self.StopMonth }\n'
        ot += f'{"StopDay ": <20}{self.StopDay }\n'
        ot += f'{"IlpLog": <20}{self.IlpLog}\n'
        ot += f'{"IlpLogFormat": <20}{self.IlpLogFormat}\n'
        ot += f'{"IlpLogVarValue": <20}{self.IlpLogVarValue}\n'
        ot += f'{"IlpLogAllCycles": <20}{self.IlpLogAllCycles}\n'
        ot += f'{"WreslPlus": <20}{self.WreslPlus}\n'
        ot += f'{"AllowSvTsInit": <20}{self.AllowSvTsInit}\n'
        ot += f'{"DatabaseURL": <20}{self.DatabaseURL}\n'
        ot += f'{"SQLGroup": <20}{self.SQLGroup}\n'
        ot += f'{"OVOption": <20}{self.OVOption}\n'
        ot += f'{"OVFile": <20}{self.OVFile}\n'
        ot += f'{"OutputCycleDatatoDss": <21}{self.OutputCycleDatatoDss}\n'
        ot += f'{"OutputAllCycleData": <21}{self.OutputAllCycleData}\n'
        cycloutstr = ','.join([str(i) for i in self.SelectedCycleOutput])
        ot += f'{"SelectedCycleOutput": <20}' + "'" +  cycloutstr + "'" +'\n'
        
        if self.Reorg:
            configfp = os.path.join(main_dir,'__study.config')
        else:
            configfp = os.path.join(main_dir, 'CONV', '__study.config')
        
        with open(configfp,'w') as of:
            of.write(ot)

    def write_wrims_batch_file(self, wrims_path, **kwargs):
        if 'main_dir' in kwargs:
            main_dir = kwargs['main_dir']
        else:
            main_dir = self.projDir
            
        if 'label' in kwargs:
            label = kwargs['label']
        else:
            label = ''
            
        bat_file_txt = '@echo off\n\n'
        if self.Reorg:
            bat_file_txt += 'set path=' + main_dir + '\Run\External;lib;%path%\n'
        else:
            bat_file_txt += 'set path=' + main_dir + '\CONV\Run\External;lib;%path%\n'
        #bat_file_txt += 'set temp_wrims2=' +  wrims_path + '\jre\\bin\n\n'
        bat_file_txt += 'set temp_wrims2=jre\\bin\n\n'
        #bat_file_txt +=  wrims_path + '\jre\\bin\java -Xmx6144m -Xss1024k -Duser.timezone=UTC -Dname=55553 -Djava.library.path='
        bat_file_txt +=  'jre\\bin\java -Xmx6144m -Xss1024k -Duser.timezone=UTC -Dname=63456 -Djava.library.path=' #-Dname=55553
        if self.Reorg:
            bat_file_txt += '"'+main_dir+'\Run\External;lib" -cp '
            bat_file_txt += '"'+main_dir+'\Run\External;lib\external;lib\WRIMSv2.jar;lib\jep-3.8.2.jar;lib\jna-3.5.1.jar;'
        else:
            bat_file_txt += '"'+main_dir+'\CONV\Run\External;lib" -cp '
            bat_file_txt += '"'+main_dir+'\CONV\Run\External;lib\external;lib\WRIMSv2.jar;lib\jep-3.8.2.jar;lib\jna-3.5.1.jar;'
        bat_file_txt += 'lib\commons-io-2.1.jar;lib\XAOptimizer.jar;lib\lpsolve55j.jar;'
        bat_file_txt += 'lib\coinor.jar;lib\gurobi.jar;lib\heclib.jar;lib\jnios.jar;lib\jpy.jar;'
        bat_file_txt += 'lib\misc.jar;lib\pd.jar;lib\\vista.jar;lib\guava-11.0.2.jar;'
        bat_file_txt += 'lib\javatuples-1.2.jar;lib\kryo-2.24.0.jar;lib\minlog-1.2.jar;'
        bat_file_txt += 'lib\objenesis-1.2.jar;lib\jarh5obj.jar;lib\jarhdf-2.10.0.jar;'
        bat_file_txt += 'lib\jarhdf5-2.10.0.jar;lib\jarhdfobj.jar;lib\slf4j-api-1.7.5.jar;'
        bat_file_txt += 'lib\slf4j-nop-1.7.5.jar;lib\mysql-connector-java-5.1.42-bin.jar;lib\sqljdbc4-2.0.jar"'
        bat_file_txt += ' wrimsv2.components.ControllerBatch -config='
        if self.Reorg:
            bat_file_txt += main_dir+'\__study.config'
        else:
            bat_file_txt += main_dir + '\CONV\__study.config'
            
        batfp = os.path.join(wrims_path, f'run_wrims_{label}.bat')
        with open(batfp,'w') as of:
            of.write(bat_file_txt)

        print("Wrote a batch file (run_wrims.bat) to %s" %wrims_path)
        print("Running batch file....")
        
    def runWRIMS(self):
        import subprocess
        
        wrims_path = r'C:\Users\jgilbert\01_Programs\WRIMS\WRIMS2_GUI_x64_20190918'

        csproj_path = self.projDir

        bat_file_txt = '@echo off\n\n'
        bat_file_txt += 'set path=' + csproj_path + '\CONV\Run\External;lib;%path%\n'
        #bat_file_txt += 'set temp_wrims2=' +  wrims_path + '\jre\\bin\n\n'
        bat_file_txt += 'set temp_wrims2=jre\\bin\n\n'
        #bat_file_txt +=  wrims_path + '\jre\\bin\java -Xmx6144m -Xss1024k -Duser.timezone=UTC -Dname=55553 -Djava.library.path='
        bat_file_txt +=  'jre\\bin\java -Xmx6144m -Xss1024k -Duser.timezone=UTC -Dname=55553 -Djava.library.path='
        bat_file_txt += '"'+csproj_path+'\CONV\Run\External;lib" -cp '
        bat_file_txt += '"'+csproj_path+'\CONV\Run\External;lib\external;lib\WRIMSv2.jar;lib\jep-3.8.2.jar;lib\jna-3.5.1.jar;'
        bat_file_txt += 'lib\commons-io-2.1.jar;lib\XAOptimizer.jar;lib\lpsolve55j.jar;'
        bat_file_txt += 'lib\coinor.jar;lib\gurobi.jar;lib\heclib.jar;lib\jnios.jar;lib\jpy.jar;'
        bat_file_txt += 'lib\misc.jar;lib\pd.jar;lib\\vista.jar;lib\guava-11.0.2.jar;'
        bat_file_txt += 'lib\javatuples-1.2.jar;lib\kryo-2.24.0.jar;lib\minlog-1.2.jar;'
        bat_file_txt += 'lib\objenesis-1.2.jar;lib\jarh5obj.jar;lib\jarhdf-2.10.0.jar;'
        bat_file_txt += 'lib\jarhdf5-2.10.0.jar;lib\jarhdfobj.jar;lib\slf4j-api-1.7.5.jar;'
        bat_file_txt += 'lib\slf4j-nop-1.7.5.jar;lib\mysql-connector-java-5.1.42-bin.jar;lib\sqljdbc4-2.0.jar"'
        bat_file_txt += ' wrimsv2.components.ControllerBatch -config='
        bat_file_txt += csproj_path + '\CONV\__study.config'
        
        batfp = os.path.join(wrims_path, 'run_wrims.bat')
        
        with open(batfp,'w') as of:
            of.write(bat_file_txt)

        print("Wrote a batch file (run_wrims.bat) to %s" %wrims_path)
        print("Running batch file....")

        from subprocess import Popen, PIPE
        os.chdir(wrims_path)
        handle = Popen('run_wrims.bat', stdin=PIPE, stderr=PIPE, stdout=PIPE, shell=True)
        output = handle.stdout.read()
        if handle.returncode != 0:
            print("Error running CalSim!")
            return(output)
        else:
            return('')
        
    def copy_study(self,dest_dir, clean_SV=True, clean_DV=True, clean_INIT=False):
        import shutil, errno
        import glob

        def copyanything(src, dst):
            try:
                shutil.copytree(src, dst)
            except OSError as exc: # python >2.5
                if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                    shutil.copy(src, dst)
                else: raise
                
        # copy the /Run directory
        if self.Reorg:
            src = os.path.join(self.projDir, 'Run')
            dst = os.path.join(dest_dir, 'Run')
            copyanything(src, dst)
            shutil.copytree(os.path.join(self.projDir,'DSS'),
                            os.path.join(dest_dir, 'DSS'))
            # copyanything from above will include all files - delete specified ones if desired
            if clean_SV:
                os.remove(os.path.join(dest_dir,'DSS','input',os.path.basename(self.SV_FP)))
                for f in glob.glob(os.path.join(dest_dir, 'DSS', 'input', '*.dsk')):
                    os.remove(f)
                for f in glob.glob(os.path.join(dest_dir, 'DSS', 'input', '*.dsc')):
                    os.remove(f)
                for f in glob.glob(os.path.join(dest_dir, 'DSS', 'input', '*.dsd')):
                    os.remove(f)
            if clean_DV:
                try:
                    os.remove(os.path.join(dest_dir,'DSS','output',os.path.basename(self.DV_FP)))
                except:
                    print(f"Could not find file {self.DV_FP} to delete it..")
                for f in glob.glob(os.path.join(dest_dir, 'DSS', 'output', '*.ds*')):
                    os.remove(f)
                os.remove(os.path.join(dest_dir, 'DSS','output','GWhyd.out'))
            if clean_INIT:
                os.remove(os.path.join(dest_dir,'DSS','input',os.path.basename(self.InitFile)))
            
        
class csDVdata(calsim):

    def __init__(self, calsim):
        self.csObj = calsim
        self.DV_DSS = None #DSS file object
        self.PathList = None
        self.DVtsDF = None
        
    def getDVts(self, **kwargs):
        '''
            **kwargs:
                filter:     list of strings to search for in all paths returned by catalog call; 
                            no support for wildcards at this time - must be a continuous string
        '''
        ts_startDate = self.csObj.StartDate
        ts_endDate = self.csObj.EndDate
        
        if 'startDate' in kwargs:
            tmp_ts_startDate = kwargs['startDate']
            if type(tmp_ts_startDate)!=dt.datetime:
                print("provided start date is not a datetime object - reverting to study start date of %s" %dt.date.isoformat(ts_startDate))
            else:
                ts_startDate = tmp_ts_startDate
        if 'endDate' in kwargs:
            tmp_ts_endDate = kwargs['endDate']
            if type(tmp_ts_endDate)!=dt.datetime:
                print("provided end date is not a datetime object - reverting to study start date of %s" %dt.date.isoformat(ts_endDate))
            else:
                ts_endDate = tmp_ts_endDate
        
        if not hasattr(self.csObj, 'DV_FP'):
            try:
                self.csObj.DV_FP = self.csObj.DvarFile
            except AttributeError:
                print("calsim object does not have any DV filepath var defined - try setting DvarFile attribute")
                
        if not os.path.exists(self.csObj.DV_FP):
            print("Couldn't find DV file {0}".format(self.csObj.DV_FP))
            return(-1)
        
        # attempting here vvv to avoid re-reading the DSS file if we've alread
        # loaded it in and read from it once
        if self.DV_DSS == None:        
            self.DV_DSS = util.dssFile(fp=self.csObj.DV_FP)

        self.DV_DSS.openDSS()

        if self.DV_DSS.PathList ==[]:
            self.DV_DSS.get_cat_paths()  # this reads the catalog path list into the DV DSS object
        
        if self.DV_DSS.CndPathList ==[]:
            self.DV_DSS.get_condensed_catalog(drop_parts=['D'], group_parts=['B'])

        if 'filt_scope' in kwargs:
            if kwargs['filt_scope'].upper() in ['COND','CND', 'CONDENSE', 'CONDENSED']:
                self.PathList = self.DV_DSS.CndPathList
            else:
                self.PathList = self.DV_DSS.PathList
        else:
            self.PathList = self.DV_DSS.CndPathList

        if 'filter' in kwargs:
            newPathList = []
            if 'filt_method' not in kwargs:
                kwargs['filt_method'] = 'list'
                
            if type(kwargs['filter'])==str: # not provided as a list - tsk tsk!
                print("Warning: filter string not provided as a list - fixing it - but only this time!")
                kwargs['filter'] = list(kwargs['filter'])
                
            if ('filt_method' in kwargs) & (kwargs['filt_method']=='regex'):
                import re
                print("Filtering path list using regular expressions")
                filt_list_re = []
                for fstr in kwargs['filter']:
                    print("trying filter string %s" %fstr)
                    r = re.compile(fstr, re.IGNORECASE)
                    filt_list_re.append(r)
                    newList = list(filter(r.match, self.PathList))
                    newPathList.append(newList)
                    print("...found %d matches" %len(newList))
                newPathList = [item for sublist in newPathList for item in sublist]    #flatten out any lists within the big list
            else:
                print("Filtering path list")
    
                for pl in self.PathList:
                    for fstr in kwargs['filter']:
                        if fstr.upper() in pl:
                            newPathList.append(pl)
        else:
            newPathList = self.PathList.copy()
        

        # initialize the dataframe where we'll store the data
        self.initDVDF(startDate=ts_startDate, endDate=ts_endDate)
        
        listDF = []
        i = 0
        numPaths = len(newPathList)
        for cpath in newPathList:
            #print("Retrieving data for: {0}".format(cpath))
            # provide a progress bar in the console
            util.progress(i,numPaths, status='Retrieving %s DV Time Series' %(numPaths))

            [dum1, ptA, ptB, ptC, ptD, ptE, ptF, dum2]=cpath.split('/')
            tmpvar = util.dssVar(self.DV_DSS, cpath=cpath)
            
            tmpvar.getRTS(ts_startDate,ctime24=True, endDateTime=ts_endDate)

            tmpDF = self.DVtsDF.copy()
            tmpDF[ptA, ptB, ptC, ptE, ptF, tmpvar.RecordType,
                         tmpvar.Units] = tmpvar.RTS
            listDF.append(tmpDF)
            i +=1
        self.DVtsDF = pnd.concat(listDF, axis='columns')
        self.DV_DSS.closeDSS()        
        
    def initDVDF(self, startDate='', endDate=''):
        # TODO: this assumes all data are 1-monthly values; 
        #       find a way to handle data with different time resolution
        
        if startDate=='':
            startDate = self.csObj.StartDate
        if endDate=='':
            endDate = self.csObj.EndDate
        # the following lines set up the time index
        psdt = dt.datetime.strftime(startDate,'%Y-%m-%d')
        pedt = dt.datetime.strftime(endDate, '%Y-%m-%d')
        pidx = pnd.period_range(start=psdt, end=pedt, freq='M')

        dtList = af.perToDTlist(pidx, include_time=True)

        # initialize an empty dataframe based on time index and expected
        # DSS path name components
        col_mindex = pnd.MultiIndex(levels=[[]]*7,
                             codes=[[]]*7,
                             names=[u'A', u'B', u'C',u'E',u'F',u'Type',u'Units'])

        self.DVtsDF = pnd.DataFrame(index=dtList, columns = col_mindex )        
        
class csSVdata(calsim):

    def __init__(self, calsim):
        self.csObj = calsim
        self.SV_DSS = None # DSS file object
        self.PathList = None
        self.SVtsDF = None

    def getSVts(self, **kwargs): #, ts_startDate=calsim.StartDate, ts_endDate=calsim.EndDate):
        '''
            **kwargs:
                filter:     list of strings to search for in all paths returned by catalog call; 
                            no support for wildcards at this time - must be a continuous string
        '''
#        ts_startDate = self.csObj.StartDate
#        ts_endDate = self.csObj.EndDate

        if not os.path.exists(self.csObj.SV_FP):
            print("Couldn't find SV file {0}".format(self.csObj.SV_FP))
            return(-1)
                
        self.SV_DSS = util.dssFile(fp=self.csObj.SV_FP)

        self.SV_DSS.openDSS()

        self.SV_DSS.get_cat_paths()  # this reads the catalog path list into the SV DSS object
        if 'group_parts' in kwargs:
            if type(kwargs['group_parts']) != list:
                grppts = list(kwargs['group_parts'])
            else:
                grppts = kwargs['group_parts']
            self.SV_DSS.get_condensed_catalog(drop_parts=['D'], group_parts=grppts)
        else:
            self.SV_DSS.get_condensed_catalog(drop_parts=['D'], group_parts=['B'])


        self.PathList = self.SV_DSS.CndPathList

        if 'filter' in kwargs:
            
            newPathList = []
            if 'filt_method' not in kwargs:
                kwargs['filt_method'] = 'list'
            
            print("Filtering path list")
            if type(kwargs['filter'])==str: # not provided as a list - tsk tsk!
                print("Warning: filter string not provided as a list - fixing it - but only this time!")
                kwargs['filter'] = list(kwargs['filter'])
                
            if ('filt_method' in kwargs) & (kwargs['filt_method']=='regex'):
                import re
                print("Filtering path list using regular expressions")
                filt_list_re = []
                for fstr in kwargs['filter']:
                    print("trying filter string %s" %fstr)
                    r = re.compile(fstr)
                    filt_list_re.append(r)
                    newList = list(filter(r.match, self.PathList))
                    newPathList.append(newList)
                    print("...found %d matches" %len(newList))
                newPathList = [item for sublist in newPathList for item in sublist]    #flatten out any lists within the big list
            else:     
                for pl in self.PathList:
                    for fstr in kwargs['filter']:
                        if fstr in pl:
                            newPathList.append(pl)
        else:
            newPathList = self.PathList.copy()
        

        # initialize the dataframe where we'll store the data
        #originally defined the DF once, here --> self.initSVDF()
        
        listDF = []
        i = 0
        numPaths = len(newPathList)
        for cpath in newPathList:
            #print("Retrieving data for: {0}".format(cpath))
            # provide a progress bar in the console
            util.progress(i,numPaths, status='Retrieving %s SV Time Series' %(numPaths))

            [dum1, ptA, ptB, ptC, ptD, ptE, ptF, dum2]=cpath.split('/')
            tmpvar = util.dssVar(self.SV_DSS, cpath=cpath)
            
            ts_startDate = self.csObj.StartDate
            ts_endDate = self.csObj.EndDate

            tmpvar.getStartEnd(ts_startDate, ts_endDate, window=12)
            if tmpvar.RecordStart < ts_startDate:
                print("using extended beginning date: %s" %tmpvar.RecordStart)
                ts_startDate = tmpvar.RecordStart
            
            if tmpvar.RecordEnd > ts_endDate:
                print("using extended end date: %s" %tmpvar.RecordEnd)
                ts_endDate = tmpvar.RecordEnd
            
            self.initSVDF(startDate=ts_startDate, endDate=ts_endDate)
            
            tmpvar.getRTS(ts_startDate,ctime24=True, endDateTime=ts_endDate)
            # tmpDF = pnd.DataFrame(tmpvar.RTS)
            # tmpDF.columns = ['value']
            # tmpDF['A'] = ptA
            # tmpDF['B'] = ptB
            # tmpDF['C'] = ptC
            # tmpDF['E'] = ptE
            # tmpDF['F'] = ptF
            # tmpDF['Type'] = tmpvar.RecordType.upper()
            # tmpDF['Units'] = tmpvar.Units.upper()
            # listDF.append(tmpDF)
            # self.SVtsDF[ptA, ptB, ptC, ptE, ptF, tmpvar.RecordType,
            #             tmpvar.Units] = tmpvar.RTS
            tmpDF = self.SVtsDF.copy()
            tmpDF[ptA, ptB, ptC, ptE, ptF, tmpvar.RecordType,
                         tmpvar.Units] = tmpvar.RTS
            listDF.append(tmpDF)
            i +=1
        self.SVtsDF = pnd.concat(listDF, axis='columns', join='outer')
        self.SV_DSS.closeDSS()

    def initSVDF(self, startDate=None, endDate=None):
        # TODO: this assumes all data are 1-monthly values; 
        #       find a way to handle data with different time resolution
        
        if startDate==None:
            startDate=self.csObj.StartDate
            
        if endDate==None:
            endDate = self.csObj.EndDate

        # the following lines set up the time index
        psdt = dt.datetime.strftime(startDate,'%Y-%m-%d')
        pedt = dt.datetime.strftime(endDate, '%Y-%m-%d')
        pidx = pnd.period_range(start=psdt, end=pedt, freq='M')

        dtList = af.perToDTlist(pidx, include_time=True)

        # initialize an empty dataframe based on time index and expected
        # DSS path name components
        col_mindex = pnd.MultiIndex(levels=[[]]*7,
                             codes=[[]]*7,
                             names=[u'A', u'B', u'C',u'E',u'F',u'Type',u'Units'])

        self.SVtsDF = pnd.DataFrame(index=dtList, columns = col_mindex )


    def setSVts(self, dssfp, ts_startDate=None, ts_endDate=None,
                use_double_precision=True, ctime24=True, debug=False):
        ''' 
        assumes a time series dataframe with have been created
            and added to the object; writes out results to filepath=dssfp
            
            ts_startDate: datetime object indicating date of time series to start
                          at for writing; if None (default) then the first value
                          in the time series will serve as the start
            
            ts_endDate: datetime object indicating date of time series at which
                        to end writing; if None (default), then the last value
                        in the time series will serve as the end
                        
        '''
        
        if not type(self.SVtsDF) == pnd.core.frame.DataFrame:
            print("Error - no valid dataframe provided - try again!")
            return
    
        # check the column indexing to make sure we have what we need to write 
        # to DSS
        cols = self.SVtsDF.columns
        colidxnames = cols.names
        colidxnamesUPP = [u.upper() for u in colidxnames]
        
        reqd_pts = ['A','B','C','E','F', 'TYPE','UNITS']
        
        missingpts = [m for m in reqd_pts if m not in colidxnamesUPP ]
        
        if missingpts ==[]:
            pass
        elif missingpts ==['D']:
            print("No D-part included in dataframe header index. Assuming this" +\
                  " is not required for this time series. Tentatively proceeding..")
        else:
            print("The following parts are missing from the dataframe header\n " +\
                  "and need to be provided to write to the DSS file:")
            for m in missingpts:
                print(m)
            print("*** Exiting! ***")
            return
        
        aidx = colidxnames.index('A')
        bidx = colidxnames.index('B')
        cidx = colidxnames.index('C')
        eidx = colidxnames.index('E')
        fidx = colidxnames.index('F')
        typidx = colidxnamesUPP.index('TYPE')
        unitidx = colidxnamesUPP.index('UNITS')
        
        # see if we can weed out extra columns that shouldnt' be written to DSS
        # assuming 'orig_date' and 'WY' are ones to drop...TODO: add kwargs for
        # specifying extra drop columns or filters
        writecols = []
        dropcols = ['orig_date', 'WY']
        for c in cols:
            if c[0] in dropcols or c[1]=='':
                pass
            else:
                writecols.append(c)
                
        # create a DSS object
        self.SV_DSS = util.dssFile(fp=dssfp)

        # open the DSS file so we can write to it - this should create the
        # file if it doesn't already exist
        self.SV_DSS.openDSS()

        # iterate through column indices, assemble path and write corresponding
        # time series to DSS
        for c in writecols:
            #get data from dataframe for this variable and time slice, if needed
            if ts_startDate == None and ts_endDate==None:
                data = self.SVtsDF.loc[:,c]
                cdate = dt.datetime.strftime(data.index[0], '%d%b%Y')
            else:
                if ts_endDate==None:
                    fmt_stDate = dt.datetime.strftime(ts_startDate, '%Y-%m-%d')
                    cdate = dt.datetime.strftime(ts_startDate, '%d%b%Y')
                    data = self.SVtsDF.loc[fmt_stDate:,c]
                elif ts_startDate==None:
                    fmt_endDate =dt.datetime.strftime(ts_endDate, '%Y-%m-%d')
                    data = self.SVtsDF.loc[:fmt_endDate,c]
                    cdate = dt.datetime.strftime(data.index[0], '%d%b%Y')
                else: #then use both ends of time slice
                    fmt_stDate = dt.datetime.strftime(ts_startDate, '%Y-%m-%d')
                    cdate = dt.datetime.strftime(ts_startDate, '%d%b%Y')
                    fmt_endDate =dt.datetime.strftime(ts_endDate, '%Y-%m-%d')
                    data = self.SVtsDF.loc[fmt_stDate:fmt_endDate, c]

            
            if ctime24:
                ctime='2400'
            else:
                ctime= dt.datetime.strftime(data.index[0], '%H%M')
                
            #cpath = '/'+'/'.join([c[aidx],c[bidx],c[cidx],'',c[eidx],c[fidx]]) + '/'
            
            # create a DSS variable object
            tmpvar = util.dssVar(self.SV_DSS, A=c[aidx], B=c[bidx],C=c[cidx],
                                  D='', E=c[eidx], F=c[fidx])
            if debug:
                print("Cpath for variable is: " + tmpvar.Cpath)
                
            tmpvar.Type = c[typidx]
            tmpvar.Units = c[unitidx]
            tmpvar.RTS = list(data.values)
            
            # TODO: functionality to add/update coordinates and other supp info,
            #      if desired for double-precision writing
            coords = []
            icdesc = []
            csupp = ''
            ctzone = 'PST'
            tmpvar.SuppInfo = {'coords': coords, 'icdesc': icdesc, 'csupp':csupp, 'ctzone': ctzone}

            if debug:
                print(use_double_precision)
                
            tmpvar.setRTS(cdate, ctime, dbl=use_double_precision)
            
        self.SV_DSS.closeDSS()
        
        
#    def getExceedance(self):
#        ''' a helper function to retrieve the data as exceedance '''
        

class calsim_year:
    
    def __init__(self, NominalYear=4000):
        self.NominalYear = NominalYear
        self.Inflows = None
        self.ResEvap = None
        self.ClosureTerms = None
        self.Demands = None
        self.Other = None
        self.Table = Odict()

class calsim_table(calsim):
    
    def __init__(self, calsim, name, ):
        csObj = calsim
        if name[-6:]!='.table':
            filename = name + '.table'
        else:
            filename = name
        
        if calsim.Reorg:
            if name in ['american_runoff_forecast.table',
                        'sacramento_runoff_forecast.table', 
                        'feather_runoff_forecast.table']:
                self.FilePath = os.path.join(csObj.TableDir, 'gen',filename)
            else:
                self.FilePath = os.path.join(csObj.TableDir,filename)
        else:
            self.FilePath = os.path.join(csObj.TableDir, filename)
        print(f'Reading CalSim input table: {self.FilePath}')
        self.Data = read_CSlookup(self.FilePath)
        
    
class ensembleCS(calsim):  

    def __init__(self, calsim):
        self.csObj = calsim
        self.Seed = None
        
    def setSeed(self, seed):
      self.Seed = seed
      np.random.seed(seed)
      
     
    def prepare_years(self, skip_tables=False):
        ''' an attempt to prepare a set of 'years' objects that can 
            be more easily shuffled
        '''
            
        tdf = self.csObj.SVdata.SVtsDF
        tdf['WY'] = tdf.index.map(af.addWY)
        
        years = range(af.addWY(self.csObj.StartDate)-3,af.addWY(self.csObj.EndDate)+2)
        
        for yr in years:
            thisyr = calsim_year(NominalYear=yr)
            
            if not skip_tables:
                # deal with table data
                for tbk, tbi in self.csObj.Tables.items():
                    cols = tbi.Data.columns
                    uppcols = [c.upper() for c in cols]
                    
                    if 'WATERYEAR' in uppcols:
                        indxCol = cols[uppcols.index('WATERYEAR')]
                    elif 'WY' in uppcols:
                        indxCol = cols[uppcols.index('WY')]
                    elif 'YEAR' in uppcols:
                        indxCol = cols[uppcols.index('YEAR')]
                    else:
                        indxCol = cols[0]  #<-- if all else fails, try to use the first column?
                    
                    if yr in list(tbi.Data[indxCol]):
                        thisData = tbi.Data[tbi.Data[indxCol]==yr]
                        thisyr.Table[tbk] = [indxCol, thisData]
                    else:
                        print("Table %s does not have year %s" %(tbk, yr))
                
            # deal with SV data
            if yr in list(tdf['WY']):
                wydata = tdf[tdf['WY']==yr]
                thisyr.Other = wydata
            
            self.csObj.Years[yr] = thisyr
            self.csObj.TemporaryDF = tdf
        
      
      

    def shuffle_ts(self, exclude=[1919,1920,1921,2016], shuffle_list=[]):
        ''' if the shuffle list is emtpy or None, then do a random shuffling
            and return the result along with the time series
        '''
        
        if self.Seed == None:
            np.random.seed(221953)
            self.Seed = 221953
            
        svMinDate = min(self.csObj.SVdata.SVtsDF.index)
        svMaxDate = max(self.csObj.SVdata.SVtsDF.index)
        svMinWY = min(min(self.csObj.Years), [x.year+1 if x.month>9 else x.year for x in [svMinDate]][0])
        svMaxWY = max(max(self.csObj.Years),[x.year+1 if x.month>9 else x.year for x in [svMaxDate]][0])
        sv_years_span = np.arange(svMinWY, svMaxWY)
        minyr, maxyr = min(min(self.csObj.Years), self.csObj.StartDate.year+1), \
                       max(max(self.csObj.Years), self.csObj.EndDate.year)+1 # adding one to deal with water year indexing
        print(minyr, maxyr)
        years_span  = np.arange(minyr, maxyr)
        

        #reordered_years = Odict()  # a new dictionary for the reordered years objects
        reordered_years = []  # for shuffling algorithms where the same original year can show up multiple times, a dictionary keyed on that year won't work
        reordered_yr_data = [] # do it as a list of years and list of data objects

        sequential_yrs = []  # list of the years covered, in original sequential order (1922, 1923,..2002, 2003, etc)

        if len(shuffle_list)==0 or shuffle_list==None:
            
            yearshuff = np.arange(minyr, maxyr)
            shuff_years_span =  np.setdiff1d(yearshuff, exclude)
            
            if exclude != []:
                yearshuff = np.setdiff1d(yearshuff, exclude)
            np.random.shuffle(yearshuff)
            shuffle_list = yearshuff
            
        else:
            #for ysh in 
            shuff_years_span = shuffle_list
#            tmp = [x for x in np.arange(minyr, maxyr) if x not in exclude] #[x for x in shuffle_list if x not in exclude] 
#            tmp2 = np.setdiff1d(shuffle_list, tmp)
#            shuff_years_span = [x for x in shuffle_list if x in tmp2]
            #print(shuff_years_span)
            
        print("Excluding %s years" %len(exclude))
        print("Shuffling %s years" %len(shuffle_list))

        tmpexclude = exclude.copy()
        new_shuff = []
        for en in tmpexclude:       
            if (en < min(shuff_years_span)) & (en in sv_years_span) :   # this is an attempt to retain the years excluded from shuffling in the right order
                tmpyr = en #tmpexclude[tmpexclude.index(en)]
                #dummy = tmpexclude.pop(tmpexclude.index(tmpyr))
                print("popping year: %s" %tmpyr)
                nyr = calsim_year(NominalYear=tmpyr)
                
                if type(self.csObj.Years[en].Other) == pnd.core.frame.DataFrame: # != None:
                    tmpts = self.csObj.Years[en].Other.copy() # need to make this a copy so we don't change the original data
                    tmpts.rename_axis(index='orig_date', inplace=True) # save the original datetieme for checking
                    tmpts.reset_index(inplace=True)  # move the original datetime to a column so we can re-index
                    nyr.Other = tmpts.copy()
                    del(tmpts)
            
                if len(self.csObj.Years[en].Table) >0:
                    for tk, ti in self.csObj.Years[en].Table.items(): 
                        indxCol = ti[0]
                        tbl = ti[1]
                        #indxCol, tmptbl = self.csObj.Years[shuffle_list[n]].Table
                        tbl[indxCol] = [en]*len(tbl)  #<-- re-assign the year column in the table 
                        nyr.Table[tk] = [indxCol, tbl]
                
                #reordered_years[en] = nyr
                reordered_years.append(en)
                reordered_yr_data.append(nyr)
                new_shuff.append(en)
                sequential_yrs.append(en)
                
        #for n in range(len(shuffle_list)):
        #for iy, n in enumerate(shuffle_list):
        sv_years_span = np.setdiff1d(sv_years_span, sequential_yrs)       #<-- this should remove the excluded years from the list to iterate over
                                                                    # but the 'if' statement in the middle of the next section should catch the years excluded from the shuffle in the middle and at the end
        
        for iy, ns in enumerate(sv_years_span): #enumerate(np.arange(1921, 2003)):
            print("shuffling year, index: %s, %s" %(ns, iy))
            if iy < len(shuffle_list):
                thisYr = shuffle_list[iy]  #shuff_years_span[iy]
                nyr = calsim_year(NominalYear=thisYr) # picks the water year of the original time seequence, in order (ie., 1921, 1922, 1923..2014,2015)
                
                print(" %s --- %s " %(ns, thisYr))
                if type(self.csObj.Years[shuffle_list[iy]].Other) == pnd.core.frame.DataFrame: # != None:
                    tmpts = self.csObj.Years[shuffle_list[iy]].Other.copy(deep=True) # need to make this a copy so we don't change the original data
                    tmpts.rename_axis(index='orig_date', inplace=True) # save the original datetieme for checking
                    tmpts.reset_index(inplace=True)  # move the original datetime to a column so we can re-index
                    nyr.Other = tmpts.copy()
                    del(tmpts)
                
                if len(self.csObj.Years[shuffle_list[iy]].Table) >0:
                    tmpts = self.csObj.Years[shuffle_list[iy]].Table.copy()
                    for tk, ti in tmpts.items(): # self.csObj.Years[shuffle_list[iy]].Table.items(): 
                        if 'nodos_flowselect.table' in tk.lower():
                            print(f'Found the table {tk}')
                        indxCol = ti[0]
                        tbl = ti[1].copy()
                        #indxCol, tmptbl = self.csObj.Years[shuffle_list[n]].Table
                        tbl[indxCol] = [int(ns)]*len(tbl) #[thisYr]*len(tbl)  #<-- re-assign the year column in the table 
                        nyr.Table[tk] = [indxCol, tbl]
                    
                #reordered_years[thisYr] = nyr
                # if 'nodos_flowselect.table' in [l.lower() for l in nyr.Table.keys()]:
                #     print(f'\n\n***********nodos table still in the mix here...')
                #     return(reordered_years, reordered_yr_data, new_shuff, sequential_yrs)
                reordered_years.append(thisYr)
                reordered_yr_data.append(nyr)
                sequential_yrs.append(ns)
                new_shuff.append(shuffle_list[iy])
                tmpyr = ns
            
            for en in tmpexclude:
                if (en == tmpyr+1) & (en in self.csObj.Years): #thisYr+1:   # this is an attempt to retain the years excluded from shuffling in the right order
                    tmpyr = en #tmpexclude[tmpexclude.index(en)]
                    print("popping year: %s" %tmpyr)
                    #dummy = tmpexclude.pop(tmpexclude.index(tmpyr))
                    nyr = calsim_year(NominalYear=tmpyr)
                     
                    if type(self.csObj.Years[en].Other) == pnd.core.frame.DataFrame: # != None:
                        tmpts = self.csObj.Years[en].Other.copy() # need to make this a copy so we don't change the original data
                        tmpts.rename_axis(index='orig_date', inplace=True) # save the original datetieme for checking
                        tmpts.reset_index(inplace=True)  # move the original datetime to a column so we can re-index
                        nyr.Other = tmpts.copy()
                        del(tmpts)
            
                    if len(self.csObj.Years[en].Table) >0:
                        for tk, ti in self.csObj.Years[en].Table.items(): 
                            indxCol = ti[0]
                            tbl = ti[1]
                            #indxCol, tmptbl = self.csObj.Years[shuffle_list[n]].Table
                            tbl[indxCol] = [en]*len(tbl)  #<-- re-assign the year column in the table 
                            nyr.Table[tk] = [indxCol, tbl]
                    
                    #reordered_years[en] = nyr

                    reordered_years.append(en)
                    reordered_yr_data.append(nyr)
                    new_shuff.append(en)
                    sequential_yrs.append(en)
                    #thisYr += 1
                    tmpyr +=1
                    
        if 'nodos_flowselect.table' in [[l.lower() for l in nn.Table.keys()] for nn in reordered_yr_data]:
            print(f'\n\n***********nodos table here at the end too...')
                    
        return(reordered_years, reordered_yr_data, new_shuff, sequential_yrs)
        
        
    def check_Yr2YrIndex(self, reordered_yr, reordered_dat, newcsObj, seqYrs, 
                         prevSacIdx=8.5,  prevSJRIdx=3.45, prevShastaInflow=2418,
                         prevFeatherInflow=[1488.,3434.]):
        '''
            some indices depend on previous year's values (Sac, SJR, Shasta, etc)
            - iterate through shuffled table values and recalculate to ensure 
            consistency
        '''
        if type(prevSacIdx)==int or type(prevSacIdx)==float:
            sacOvrRide=True
        else:
            sacOvrRide=False
            
        if type(prevSJRIdx)==int or type(prevSJRIdx)==float:
            sjrOvrRide=True
        else:
            sjrOvrRide=False            
            
        if type(prevShastaInflow)==int or type(prevShastaInflow)==float:
            shaIndxOvrRide=True
        else:
            shaIndxOvrRide=False
            
        if type(prevFeatherInflow)==list:
            fthrOvrRide=True
        else:
            if type(prevFeatherInflow)==int or type(prevFeatherInflow)==float:
                print("\n******************************************************")
                print("Feather inflow override provided as a single number\nbut we need values for the last 2 years." )
                print(" Assuming the value provided should be applied to both years - this may not be what you intended!!!")
                print("*****************************************************\n")
                prevFeatherInflow = [prevFeatherInflow, prevFeatherInflow]
                fthrOvrRide=True
            else:
                fthrOvrRide=False
            
        for t, tobj in newcsObj.Tables.items(): #self.csObj.Tables:
            print(t)
            biglist = []
            sacindexlist = []
            
            if t.lower() == 'sacvalleyindex.table':
                #prevIdx = 5.15  #<--value for wy 1920 from historical hydrology (Q0); from wateryearindex.xlsx, in turn from CDEC historic index record (for wy 1921)
                if sacOvrRide:
                    prevIdx = prevSacIdx 
                else:
                    prevIdx = 8.5 #8.5 #5.23  #<-- value for wy 1920 from 2025 Q5 ELT hydrology; from ClimateChange_InflowsForecastsYeartypesADjustment_11-30-15.xslx from Rob Leaf/Jacobs
                
                # adjust Shasta index while we're here
                # TODO find a way to incorporate FNF for Shasta - maybe adjust defined inflow by some ratio?
                i4 = self.csObj.SVdata.SVtsDF.loc[:,idxslc[:,'I4']] # we need Shasta inflow volume by WY to calculate the Shasta Index - technically it should be FNF, using defined inflow here for now 
                i4ann = i4.resample('A-Sep').agg(np.sum)
                
                for yr, dat,origyr in zip(reordered_yr, reordered_dat,seqYrs):
                    if t in dat.Table:
                        tmp = dat.Table[t][1]  # <-- data for shuffled year
                        dtbl = newcsObj.Tables[t].Data #<-- data for original year
                        newIdx = np.nan
                        if yr <= 1921:
                            newIdx = tmp.Index.values[0]
                            print(newIdx)
                        else:
                            newIdx = 0.3*tmp.OctMar + 0.4*tmp.AprJul + 0.3*min(10., prevIdx)
                            newIdx = newIdx.values[0]
                            
                        print("OrigYear: %s - Shuff year: %s - new value: %0.2f" %(origyr,yr, newIdx))
                        
                        tmp.Index = newIdx
                        tmp.WaterYear = origyr
                        
                        dtbl.loc[dtbl.WaterYear==origyr,'OctMar'] = tmp.OctMar.iloc[0]
                        dtbl.loc[dtbl.WaterYear==origyr,'AprJul'] = tmp.AprJul.iloc[0]
                        dtbl.loc[dtbl.WaterYear==origyr,'Wysum'] = tmp.Wysum.iloc[0]
                        dtbl.loc[dtbl.WaterYear==origyr,'Index'] = tmp.Index.iloc[0]
                        
                        if sacOvrRide:
                            prevIdx = prevSacIdx
                        else:
                            prevIdx = newIdx
                        #biglist.append(tmp)
                        
                        # save the sac wyt while we're at it
                        if newIdx >= 7.58:   #9.2: <-- 9.2 is for historical hydro; 7.58 for climate change, adjusted to keep proportion of year types similar to historical
                            wyt = 1  # (W)et
                        elif newIdx > 6.06:  #7.8: <-- 7.8 is for historical; see note for wyt1
                            wyt = 2 # (A)bove (N)ormal
                        elif newIdx > 5.15:  #6.5:  <-- 6.5 MAF is for historical; see note for wyt 1
                            wyt=3 # (B)elow (N)ormal
                        elif newIdx > 4.1:  #5.4: <-- 5.4 MAF is for historical; see note for wyt 1
                            wyt=4  # (D)ry
                        else:
                            wyt = 5  # (C)ritical
                        #sacindexlist.append(wyt)
                        
                        # adjust Shasta index here - following calcs in 'wyt PA update_Jun2021.xls' from Nancy (orig from Aaron Miller at DWR)
                        if not shaIndxOvrRide:
                            try:
                                prevShastaInflow = i4ann.loc['%s' %(yr-1)]
                            except:
                                prevShastaInflow = 5500. #TODO find a better default value for this - 5500 TAF is the 1922-2020 average FNF 
                                
                        thisI4ann = i4ann.loc['%s' %yr].iloc[0][0] # get this water year's shasta inflow volume, in TAF
                        if (thisI4ann)/1000. < 3.2:
                            cond1 = 1
                        else:
                            cond1 = 2
                        
                        if (thisI4ann<4000) and (prevShastaInflow<4000):
                            cond2 = 8000 - thisI4ann - prevShastaInflow
                        else:
                            cond2 = 0
                            
                        if cond2 > 800:
                            cond2idx = 1
                        else:
                            cond2idx = 2
                        
                        if (cond1 + cond2idx)<4:
                            SHASTAindex = 4
                        else:
                            SHASTAindex = 1
                        
                        # update the water year type
                        wytTbl = newcsObj.Tables['wytypes.table'].Data
                        wytTbl.loc[wytTbl.wateryear==origyr,'SACindex'] = wyt
                        wytTbl.loc[wytTbl.wateryear==origyr,'Shastaindex'] = SHASTAindex

            
            if t.lower() == 'sjvalleyindex.table': # this table not normally included in calsim
                                                   # studies - have to add it in or otherwise
                                                   # provide it
                sjrindexlist = []
                
                if sjrOvrRide:
                    prevIdx = prevSJRIdx
                else:
                    prevIdx = 3.45  #<-- 3.45 is for Q5 2025 hydrology; use 1921 = 3.23 MAF and 1920= 2.64 MAF for historical (from CDEC WSIHIST)
                for yr, dat, origyr in zip(reordered_yr, reordered_dat, seqYrs):
                    if t in dat.Table:
                        tmp = dat.Table[t][1]
                        dtbl = newcsObj.Tables[t].Data #<-- data for original year
                        newIdx = np.nan
                        
                        if yr <=1921:
                            newIdx = tmp.Index.values[0]
                            print("sjr index: %0.2f\n********" %newIdx)
                        else:
                            newIdx = 0.2*tmp.OctMar + 0.6*tmp.AprJul + 0.2*prevIdx
                            newIdx = newIdx.values[0]
                        
                        tmp.Index = newIdx
                        tmp.WaterYear = origyr
                        
                        dtbl.loc[dtbl.WaterYear==origyr,'OctMar'] = tmp.OctMar.iloc[0]
                        dtbl.loc[dtbl.WaterYear==origyr,'AprJul'] = tmp.AprJul.iloc[0]
                        dtbl.loc[dtbl.WaterYear==origyr,'Wysum'] = tmp.Wysum.iloc[0]
                        dtbl.loc[dtbl.WaterYear==origyr,'Index'] = tmp.Index.iloc[0]
                        
                        if sjrOvrRide:
                            prevIdx = prevSJRIdx
                        else:
                            prevIdx = newIdx
                        
                        #biglist.append(tmp)
                        
                        # save the sac wyt while we're at it
                        if newIdx >= 3.524:   #<-- 3.524 is for 2025 Q5 ELT; need to change to historical value (3.8, CDEC WSIHIST) if doing hist hydr
                            wyt = 1  # (W)et
                        elif newIdx > 2.599:  #<-- 2.599 is for 2025 Q5 ELT; change to (3.1, CDEC WSIHIST) if doing hist hydr
                            wyt = 2 # (A)bove (N)ormal
                        elif newIdx > 2.063: #<-- 2.063 is for 2025 Q5 ELT; change to (2.5, CDEC WSIHIST) if doing hist hydr
                            wyt=3 # (B)elow (N)ormal
                        elif newIdx > 1.694: #<-- 1.694 is for 2025 Q5 ELT; change to (2.1, CDEC WSIHIST) if doing hist hydr
                            wyt=4  # (D)ry
                        else:
                            wyt = 5  # (C)ritical
                        #sjrindexlist.append(wyt)
                        wytTbl = newcsObj.Table['wytypes.table'].Data
                        wytTbl.loc[wytTbl.wateryear==origyr,'SJRindex'] = wyt
                        
#           CAM_Oro_Apr_Jul.table appears to only be used with CAM - commenting out for now
#            if t.lower() == 'cam_oro_apr_jul.table':
#                
#                if fthrOvrRide:
#                    prevIdx1 = prevFeatherInflow[0]
#                    prevIdx2 = prevFeatherInflow[1]
#                else:
#                    prevIdx1 = 1488 # ????: these values are Apr-Jul vol for WY1922 (prevIdx2) & 
#                    prevIdx2 = 3434 # WY1923 (prevIdx1), taken from file:///D:\02_Projects\CalSim\util\position_analysis\wyt%20PA%20update_Jun2021_shuffle_check.xlsx
#            
#                for yr, dat, origyr in zip(reordered_yr, reordered_dat, seqYrs):
#                    if t in dat.Table:
#                        
#                        tmp = dat.Table[t][1]
#                        dtbl = newcsObj.Tables[t].Data #<-- data for original year
#                        newIdx = np.nan
                        
                        
                

#        newDF = pnd.concat(biglist, ignore_index=True, axis=0)
#        newDF = newDF.drop(columns='Notes')
#        
#        return(newDF)
        return(newcsObj)

    def table_shuffle(self, table_years, base_years, resamp_seq_index, 
                      reordered_yr, reordered_dat, seqYrs):
        
        newDFdict = {}
        newIndexes = {}
        for t, tobj in self.csObj.Tables.items(): #self.csObj.Tables:
            print(t)
            biglist = []
            #indxlist = []
            orig_indx = table_years[t]
            
            # deal with any data in tables that precedes where we start with the reshuffling
            if min(orig_indx) < min(resamp_seq_index):
                precede_years = list(range(min(orig_indx), min(base_years)))
            else:
                precede_years = []
            if max(orig_indx)>max(resamp_seq_index):
                post_years = list(range(max(base_years)+1,max(orig_indx)+1))
            else:
                post_years = []
            
            # add any preceding years to the dataset before adding the shuffled years    
            for i,p in enumerate(precede_years):
                indxCol = reordered_dat[i].Table[t][0]
                biglist.append(self.csObj.Tables[t].Data.query('%s==%d' %(indxCol, p))) #<--- assigning original data to preceding years
                
            for i,yr in enumerate(reordered_yr):
                if (yr in precede_years) or (yr in post_years): #(i==0) & (yr==1921):
                    pass
                else:
                    if t in reordered_dat[i].Table:
                        indxCol = reordered_dat[i].Table[t][0]
                        thisdat = self.csObj.Tables[t].Data.query('%s==%d' %(indxCol, yr)) #seqYrs[i]))
                        biglist.append(thisdat)
            
            # add any following years to dataset after adding shuffled years
            for i,p in enumerate(post_years):
                indxCol = reordered_dat[i].Table[t][0]
                biglist.append(self.csObj.Tables[t].Data.query('%s==%d' %(indxCol, p)))    
            
            final_index = precede_years + base_years + post_years
            newDF = pnd.concat(biglist, ignore_index=True, axis=0)
            newDF = newDF.drop(columns='Notes')
            if len(newDF)!=len(orig_indx):
                raise BaseException(f"Data length ({len(newDF)}) and index ({len(orig_indx)}) for table {t} do not match")
            newDF[indxCol] = orig_indx #indxlist
            newDFdict[t.lower()] = newDF
            newIndexes[t.lower()] = final_index
        return([newDFdict, newIndexes])
                
    def checkSacSjrWYT(self, dftabledict,reordered_dat, prevSacIdx=8.5, 
                       prevSJRIdx=3.45, prevShastaInflow=2418, prevSJR5=[3.6, 3.6, 3.6, 3.6],
                       prevFeatherInflow=[1488.,3434.]):                    
        '''
            some indices depend on previous year's values (Sac, SJR, Shasta, etc)
            - iterate through shuffled table values and recalculate to ensure 
            consistency
        '''
        if type(prevSacIdx)==int or type(prevSacIdx)==float:
            sacOvrRide=True
        else:
            sacOvrRide=False
            
        if type(prevSJRIdx)==int or type(prevSJRIdx)==float:
            sjrOvrRide=True
        else:
            sjrOvrRide=False            
            
        if type(prevShastaInflow)==int or type(prevShastaInflow)==float:
            shaIndxOvrRide=True
        else:
            shaIndxOvrRide=False
            
        if type(prevFeatherInflow)==list:
            fthrOvrRide=True
        else:
            if type(prevFeatherInflow)==int or type(prevFeatherInflow)==float:
                print("\n******************************************************")
                print("Feather inflow override provided as a single number\nbut we need values for the last 2 years." )
                print(" Assuming the value provided should be applied to both years - this may not be what you intended!!!")
                print("*****************************************************\n")
                prevFeatherInflow = [prevFeatherInflow, prevFeatherInflow]
                fthrOvrRide=True
            else:
                fthrOvrRide=False                

        # do sac valley index first
        sacdf = dftabledict['sacvalleyindex.table']
        sjrdf = dftabledict['sjvalleyindex.table']
        wytdf = dftabledict['wytypes.table']
        sjwytdf = dftabledict['wytypesjr.table']
        sj5wytdf = dftabledict['wytypesjrave5.table']
        
        newsacdf = copy.deepcopy(sacdf)
        newsjrdf = copy.deepcopy(sjrdf)
        newwytdf = copy.deepcopy(wytdf)
        newsjwytdf = copy.deepcopy(sjwytdf)
        newsj5wytdf = copy.deepcopy(sj5wytdf)
        
        #prevIdx = 5.15  #<--value for wy 1920 from historical hydrology (Q0); from wateryearindex.xlsx, in turn from CDEC historic index record (for wy 1921)
        if sacOvrRide:
            prevSacIdx = prevSacIdx 
        else:
            prevSacIdx = 8.5 #8.5 #5.23  #<-- value for wy 1920 from 2025 Q5 ELT hydrology; from ClimateChange_InflowsForecastsYeartypesADjustment_11-30-15.xslx from Rob Leaf/Jacobs
        
        if sjrOvrRide:
            prevSJRIdx = prevSJRIdx
        else:
            prevSJRIdx = 3.45  #<-- 3.45 is for Q5 2025 hydrology; use 1921 = 3.23 MAF and 1920= 2.64 MAF for historical (from CDEC WSIHIST)
        
        # adjust Shasta index while we're here
        # TODO find a way to incorporate FNF for Shasta - maybe adjust defined inflow by some ratio?
        i4 = self.csObj.SVdata.SVtsDF.loc[:,idxslc[:,'I4']] # we need Shasta inflow volume by WY to calculate the Shasta Index - technically it should be FNF, using defined inflow here for now 
        i4ann = i4.resample('A-Sep').agg(np.sum)
        
        
        sacidxcol = reordered_dat[0].Table['SacValleyIndex.table'][0]
        wytidxcol = reordered_dat[0].Table['wytypes.table'][0]
        sjridxcol = reordered_dat[0].Table['SJValleyIndex.table'][0]
        
        sacyrs = sacdf[sacidxcol].unique()
        for yr in sacyrs:
            newIdx = np.nan
            if yr <=1921:
                newIdx = sacdf.loc[sacdf[sacidxcol]==np.min(sacyrs),['Index']]
                newIdx = newIdx.values[0]
            else:
                tmp = sacdf.loc[sacdf[sacidxcol]==yr]
                newIdx = 0.3*tmp.OctMar + 0.4*tmp.AprJul + 0.3*min(10., prevSacIdx)
                newIdx = newIdx.values[0]
            newsacdf.loc[newsacdf[sacidxcol]==yr,'Index'] = newIdx
            
            # set the adjusted sac wyt while we're here
            if newIdx >= 7.58:   #9.2: <-- 9.2 is for historical hydro; 7.58 for climate change, adjusted to keep proportion of year types similar to historical
                sacwyt = 1  # (W)et
            elif newIdx > 6.06:  #7.8: <-- 7.8 is for historical; see note for wyt1
                sacwyt = 2 # (A)bove (N)ormal
            elif newIdx > 5.15:  #6.5:  <-- 6.5 MAF is for historical; see note for wyt 1
                sacwyt=3 # (B)elow (N)ormal
            elif newIdx > 4.1:  #5.4: <-- 5.4 MAF is for historical; see note for wyt 1
                sacwyt=4  # (D)ry
            else:
                sacwyt = 5  # (C)ritical
            
            newwytdf.loc[wytdf[wytidxcol]==yr,'SACindex'] = sacwyt
        
        sjyrs = sjrdf[sjridxcol].unique()
        for yr in sjyrs:
            
            newIdx = np.nan
                        
            if yr <=1921:
                newIdx = sjrdf.loc[sjrdf[sjridxcol]==np.min(sjyrs),['Index']] #tmp.Index.values[0]
                newIdx = newIdx.values[0]
                print("sjr index: %0.2f\n********" %newIdx)
            else:
                tmp = sjrdf.loc[sjrdf[sjridxcol]==yr]
                newIdx = 0.2*tmp.OctMar + 0.6*tmp.AprJul + 0.2*prevSJRIdx
                newIdx = newIdx.values[0]
            
            newsjrdf.loc[newsjrdf[sjridxcol]==yr,'Index'] = newIdx
            
            # save the sjr wyt while we're at it
            if newIdx >= 3.524:   #<-- 3.524 is for 2025 Q5 ELT; need to change to historical value (3.8, CDEC WSIHIST) if doing hist hydr
                sjrwyt = 1  # (W)et
            elif newIdx > 2.599:  #<-- 2.599 is for 2025 Q5 ELT; change to (3.1, CDEC WSIHIST) if doing hist hydr
                sjrwyt = 2 # (A)bove (N)ormal
            elif newIdx > 2.063: #<-- 2.063 is for 2025 Q5 ELT; change to (2.5, CDEC WSIHIST) if doing hist hydr
                sjrwyt=3 # (B)elow (N)ormal
            elif newIdx > 1.694: #<-- 1.694 is for 2025 Q5 ELT; change to (2.1, CDEC WSIHIST) if doing hist hydr
                sjrwyt=4  # (D)ry
            else:
                sjrwyt = 5  # (C)ritical
                
            newwytdf.loc[newwytdf[wytidxcol]==yr,'SJRindex'] = sjrwyt
            newsjrdf.loc[newsjrdf[sjridxcol]==yr,'index'] = sjrwyt
            newsjwytdf.loc[newsjwytdf[wytidxcol]==yr,'index'] = sjrwyt
            newsj5wytdf.loc[newsj5wytdf[wytidxcol]==yr, 'index'] = np.mean(prevSJR5[-4:]+ [sjrwyt])
            
        # finally, adjust Shasta index here - following calcs in 'wyt PA update_Jun2021.xls' from Nancy (orig from Aaron Miller at DWR)
        
        for yr in wytdf[wytidxcol].unique():
            if not shaIndxOvrRide:
                try:
                    prevShastaInflow = i4ann.loc['%s' %(yr-1)]
                except:
                    prevShastaInflow = 5500. #TODO find a better default value for this - 5500 TAF is the 1922-2020 average FNF 
            
            if yr in i4ann.index.year:
                thisI4ann = i4ann.loc['%s' %yr].iloc[0][0] # get this water year's shasta inflow volume, in TAF
                if (thisI4ann)/1000. < 3.2:
                    cond1 = 1
                else:
                    cond1 = 2
                
                if (thisI4ann<4000) and (prevShastaInflow<4000):
                    cond2 = 8000 - thisI4ann - prevShastaInflow
                else:
                    cond2 = 0
                    
                if cond2 > 800:
                    cond2idx = 1
                else:
                    cond2idx = 2
                
                if (cond1 + cond2idx)<4:
                    SHASTAindex = 4
                else:
                    SHASTAindex = 1
                
                # update the water year type
                newwytdf.loc[newwytdf[wytidxcol]==yr, 'Shastaindex']= SHASTAindex
            
        dftabledict['sacvalleyindex.table'] = newsacdf
        dftabledict['sjvalleyindex.table'] = newsjrdf
        dftabledict['wytypes.table'] = newwytdf
        dftabledict['wytypesjr.table'] = newsjwytdf
        dftabledict['wytypesjrave5.table'] = newsj5wytdf
        
        return(dftabledict)


#%%    
    
#rffp = r'C:\Users\jmgilbert\02_Projects\03_MP\MP2019\04_CSHydro\CalSimHydro_v20190903\1a_RainfallRunoff\main.dat'
#idcfp = r'C:\Users\jmgilbert\02_Projects\03_MP\MP2019\04_CSHydro\CalSimHydro_v20190903\2a_IDC\Main.dat'
#idcfp = r'C:\Users\jmgilbert\02_Projects\03_MP\MP2019\03_CalSim3\03_Review\DWR_Review\CS3Merged20190319\CalSimHydro_L2015A_Merge_20190213\2a_IDC\Main.dat'
#
#thiscsh = calsimhydro()
#thiscsh.IDC = idc(idcfp)
#
#thisIDC = thiscsh.IDC
#thisIDC.init_DU_list()
#
#thisIDC.get_land_use()
#thisIDC.get_irr_eff()
#thisIDC.get_idc_params()
#thisIDC.get_et_ts()
#
#
##%%
#tt = thisIDC.Demand_Units['64_PA1'].ET_ts
#
#ttpv = patternizer(tt['AP'][0],freq='M', reference='WY')
#fig, ax = patternPlot(ttpv, indexCol='index')
##tt1 = pnd.DataFrame(data=tt['AP'][0].copy(), columns=['value'])
##tt1['Month'] = tt1.index.month
##tt1['Year'] = tt1.index.year
##tt1['WY'] = tt1.index.map(lambda x: af.addWY(x))
##ttpv = pnd.pivot(tt1, index='Month', columns='WY')['value']
#thisIDC.Demand_Units['90_PA1'].report_du_crop('AP')

#%%

#paramFP = r'C:\Users\jmgilbert\02_Projects\03_MP\MP2019\03_CalSim3\03_Review\DWR_Review\CS3Merged20190319\CalSimHydro_L2015A_Merge_20190213\2a_IDC\Parameter.dat'
#
#with open(paramFP,'r') as of:
#    rl = of.readlines()
#    
#    rlnc = [r for r in rl if (r[0].upper()!='C')]
#    rlnc = [r for r in rlnc if (r[0]!='*')]
#    rlnc = [r for r in rlnc if r.strip()!='']
#    nregn =thisIDC.IDC_Dict['NREGN']
#    ncrop = thisIDC.IDC_Dict['NCROP']
#    nlu = ncrop+2
#    grp = 0
#    # first group is field capacity table - cols = crops (1-22); rows=du/iregn
#    fcdf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn],columns=thisIDC.IDC_Dict['CropCodes'])
#    # then (ef)fective porosity
#    grp =1 
#    efdf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], columns = thisIDC.IDC_Dict['CropCodes'])
#    #then curve number - although this is not used in the idc implementation in calsimhydro
#    grp=2
#    cndf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], columns = thisIDC.IDC_Dict['CropCodes'])
#    
#    grp = 3
#    # no native vegetation in the return flow specification, adjust columns appropariately
#    noNVcols = [cr for cr in thisIDC.IDC_Dict['CropCodes'] if cr != 'NV']
#    icrfdf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], dtype='i',columns = noNVcols)
#    
#    grp=4
#    # water use parameters - indexes for where to get reuse data/info
#    noURNVcols = [cr for cr in thisIDC.IDC_Dict['CropCodes'] if cr not in ['UR', 'NV']]
#    wuparcols = ['PERV', 'ICPRECIP'] + noURNVcols + ['ICRUFURB', 'FURDPRF','FURDPSR']
#    wupardf = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], dtype='f',columns = wuparcols)
#    
#    grp=5
#    icinfilt = read_text_array(rlnc[grp*nregn:(grp+1)*nregn], dtype='i',columns = thisIDC.IDC_Dict['CropCodes'])
#    endidx5 = (grp+1)*nregn
#    
#    # rooting depths by crop
#    grp=6 
#    rootdepConvFact = float(rlnc[endidx5].split()[0].strip())
#    rootdepDF = read_text_array(rlnc[endidx5+1:endidx5+1+nlu], dtype='f',columns=['RootingDepth_ft'])
#    
#for idx,row in fcdf.iterrows():
#    du = thisIDC.IDC_Dict['DemandUnits'][idx-1]
#%%
#[FACTLN, DSSFL, fullDSSFL, lu_path_dict] = thisIDC.get_du_landuse_paths()
#
#luDSS = dssFile(fp=fullDSSFL)
#luDSS.get_cat_paths()
#
## test out getting regular time seires
#testvar = dssVar(luDSS, A='CALSIM',B='64_PA1_AL', C='LANDUSE',E='1MON',F='EXISTING')
#testvar.getRTS(dt.datetime(4000,1,31,0,0),ctime24=True, ntimes=12)
#
#luDSS.openDSS()
#luDSS.closeDSS()
#
#for (iregn, crtype) in lu_path_dict: # thisIDC.IDC_Dict['DemandUnits']:
#    du = thisIDC.IDC_Dict['DemandUnits'][iregn-1]
#    crtype = thisIDC.IDC_Dict['CropCodes'][crtype-1]
#    thisdu = 
#
##thiscsh.read_rfro(rffp)
#idcmain = thiscsh.read_MainIDC(idcfp)