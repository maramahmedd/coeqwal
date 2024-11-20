# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:25:50 2019

@author: jmgilbert
"""
#%%
import re, os, sys
import pandas as pnd

#sys.path.append(r'C:\Users\jgilbert\01_Programs\Excel DSS Add-in V3.3.jmg\libraries\64-bit')
sys.path.append('D:\\02_Projects\\CalSim\\util\\CalSim_Utilities\\Python_Functions\\Python_DSS')
import dss3_functions_reference as dss
import AuxFunctions as af

import datetime as dt

#%%

#def perToDTlist(periodidx):
#    dtlist=[]
#    for d in list(periodidx):
#        y = d.year
#        m = d.month
#        dy = d.day
#        odt = dt.date(y, m , dy)
#        dtlist.append(odt)
#    return(dtlist)

def interp_csh_dt(cshdt, sep='_'):
    ''' convert a calsimhydro datetime into an interpeted datetime object
        - assumes a m/d/year_24:00 format (24 hour max instead of 23:59)
    '''
    dat, tim = cshdt.split(sep)
    tmpdat = dt.datetime.strptime(dat,'%m/%d/%Y')
    hr, minut = tim.split(':')
    if int(hr.strip())==24:
        hr = 0
        minut = int(minut.strip())
        
        newdat = tmpdat + dt.timedelta(days=1)
    else:
        hr = int(hr.strip())
        minut = int(minut.strip())
        dayincr=0
        newdat = tmpdat 
        
    finnewdat = dt.datetime(newdat.year, newdat.month, newdat.day,hr, minut, 0)
        
    return(finnewdat)

        
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


class dssFile():
    
    def __init__(self, fp=None):
        self.filepath = fp
        self.IsOpen = False
        self.ifltab = None
        self.lgenca = False
        self.lopnca = False
        self.lcatlg = False
        self.lopncd = False
        self.lcatcd = False
        self.nrecs = None
        self.PathList = []
        self.CndPathList = []
        self.CndPathCatalog = None
        
    def openDSS(self):
        [self.ifltab, iostat] = dss.open_dss(self.filepath)
        if iostat==0:
            print("Opened file: %s"  %self.filepath)
            self.IsOpen = True
        else:
            print("something went wrong opening %s" %self.filepath)
            print("IOSTAT:  %s" %iostat)
            
    def closeDSS(self):
        dss.close_dss(self.ifltab)
        self.IsOpen=False
        
    def get_cat_paths(self):
#        [self.lgenca, self.lopnca, self.lcatlg, self.lgencd, 
#         self.lopncd, self.lcatcd, self.nrecs] = dss.open_catalog(self.filepath, 12)
#        
#        if self.nrecs>10000:
#            print("Retrieving catalog listing for %s paths - may take a moment" %self.nrecs)
        
        #self.PathList, self.lopnca = dss.read_catalog(self.lopnca)
        [self.PathList, nrecs, self.lopnca] = dss.get_catalog(self.filepath)

    def get_condensed_catalog(self, drop_parts=['D'], group_parts=['B']):
        if len(self.PathList)==0:
            print("Warning: catalog path list has not been created...doing that first\n")
            self.get_cat_paths()
        cndcatDF = af.condense_cat(self.PathList, drop_parts=drop_parts, group_parts=group_parts)
        self.CndPathList = list(cndcatDF['full_path'])
        self.CndPathCatalog = cndcatDF
        
class dssVar(dssFile):
    
    def __init__(self, dssFile, **kwargs):
        
        self.DSSFile = dssFile # connect the variable with the file it shoudl come from
        
        if 'cpath' in kwargs:
            [dum1, self.A, self.B, self.C, self.D, self.E, self.F, dum2] = kwargs['cpath'].split('/')
            if self.E == '' or self.E==None:
                print("cpath provided, but no E part to informt the time resolution\nCan't continue like this anymore....")
                self.TimeIncrNum = None
                self.TimeUnit = None
            else:
                time_unit_parts = re.split('(\d+)',self.E)[1:]
                self.TimeIncrNum = int(time_unit_parts[0])
                self.TimeUnit = time_unit_parts[1]
            
        else:
            if 'A' in kwargs:
                self.A = kwargs['A']
            else:
                self.A = ''
            if 'B' in kwargs:
                self.B = kwargs['B']
            else:
                self.B = ''
            if 'C' in kwargs:
                self.C = kwargs['C']
            else:
                self.C = ''
            if 'D' in kwargs:
                self.D = kwargs['D']
            else:
                self.D = ''    
            if 'E' in kwargs:
                self.E = kwargs['E']
                time_unit_parts = re.split('(\d+)',self.E)[1:]
                self.TimeIncrNum = int(time_unit_parts[0])
                self.TimeUnit = time_unit_parts[1]
            else:
                self.E = ''
                self.TimeIncrNum = None
                self.TimeUnit = None
            if 'F' in kwargs:
                self.F = kwargs['F']
            else:
                self.F = ''
        self.Units = None
        self.Type = None
        

        self.RecordType = None
        self.Precision = None

        self.Cpath = self.getCpath()
        self.RTS = None
        self.RecordStart = None
        self.RecordEnd = None
        self.SuppInfo = {}
        self.SuppInfo['coords'] = []
        self.SuppInfo['icdesc'] = ''
        self.SuppInfo['csupp'] = ''
        self.SuppInfo['ctzone'] = ''

        
    def timeUnitLookup(self):
        tudict= {
                'MON': 'M',
                'DAY': 'D',
                'YEAR': 'A',
                'HOUR': 'H'}
        if self.TimeIncrNum>1:
            retstr = f'{self.TimeIncrNum}{tudict[self.TimeUnit]}'
            return(retstr)
        else:
            return(tudict[self.TimeUnit])
        
    def getCpath(self):
        return('/'+'/'.join([self.A,self.B,self.C,'',self.E, self.F]) + '/')
        
        
    def getStartEnd(self, stDateTime,endDateTime, window=6, window_units='MON',
                    ctime24=True):
        '''
            function to check if any data exist beyond the prescribed
            start and end date/times - returns the datetimes rather 
            than the data
            - this is a hacky approximation of functionality available
              in the complete heclib library - this python code can be
              replaced once the linkage to the complete heclib library
              is sorted out on Windows 2020-03-19
              
        '''
        from dateutil.relativedelta import relativedelta
        
        if type(stDateTime) != dt.datetime:
            self.RTS = None
            print("couldn't parse start time - expected a datetime object")
        else:
            cdate = dt.datetime.strftime(stDateTime, '%d%b%Y') #'31Oct1921'
            if self.TimeUnit in ['HOUR','MINUTE']:
                psdate = dt.datetime.strftime(stDateTime, '%Y-%m-%d %H:%M') #format for creating a date list
            else:
                psdate = dt.datetime.strftime(stDateTime, '%Y-%m-%d') #format for creating a date list
            if ctime24:
                ctime = '2400'
            else:
                ctime = dt.datetime.strftime(stDateTime, '%H%M')
                
        # try to use an end date to determine number of values
        if type(endDateTime) != dt.datetime: 
            self.RTS = None
            print("couldn't parse end time - expected a datetime object")
        else:
            if self.TimeUnit in ['HOUR','MINUTE']:
                pedate = dt.datetime.strftime(endDateTime, '%Y-%m-%d %H:%M')
            else:
                pedate = dt.datetime.strftime(endDateTime, '%Y-%m-%d')
            # get the date list based on start date and end date
            pidx = pnd.period_range(start=psdate, end=pedate, freq=self.timeUnitLookup())
            ntimes= len(pidx)        
        # check that the DSS file is open, open if not
        if not self.DSSFile.IsOpen:
            self.DSSFile.openDSS()
        
        
        # check if values exist earlier than the given start datetime
        missing_data = False
        window_end_date = psdate
        window_end_date2 = stDateTime
        tmpwdw = window
        while missing_data==False:
                    
            if window_units.upper in ['MON', 'M', 'MONTH']:
                checkDate = stDateTime - relativedelta(months=tmpwdw)
            elif window_units.upper in ['YR', 'Y', 'YEAR']:
                checkDate = stDateTime - relativedelta(years=tmpwdw)
            elif window_units.upper in ['DY','D', 'DAY']:
                checkDate = stDateTime - relativedelta(days=tmpwdw)
            else:
                checkDate = stDateTime - relativedelta(months=tmpwdw)
            
            # get the date list based on start date and end date
            tmp_pidx = pnd.period_range(start=dt.datetime.strftime(checkDate, '%Y-%m-%d %H:%M'),
                                        end=window_end_date, freq=self.timeUnitLookup())
            ntimes= len(tmp_pidx) 
            tmpdate = dt.datetime.strftime(checkDate, '%d%b%Y')
            if ctime24:
                tmptime = '2400'
            else:
                tmptime = dt.datetime.strftime(stDateTime, '%H%M')
            
            [nvals, vals, cunits, ctype, iofset, istat] = dss.read_regts(self.DSSFile.ifltab, self.Cpath, tmpdate, tmptime, ntimes)
            
            for i,v in enumerate(vals):
                if v==-901. or v==-902.:
                    tmpdate = tmp_pidx[i]
                    print(str(tmpdate) + " --- " + str(v))
                    #break
                else:
                    tmpdate = tmp_pidx[i]  # found a non-missing record
                    break
                
            bdate = dt.datetime(tmpdate.year, tmpdate.month, tmpdate.day, 0,0)
        
            if bdate == checkDate: # we've found values back through the first window, try another window's length
                tmpwdw +=window
                window_end_date = dt.datetime.strftime(checkDate, '%Y-%m-%d')
                window_end_date2 = checkDate
            else:
                missing_data = True
        
        # check if values exist later than the given end datetime
        missing_data = False
        window_start_date = pedate
        window_start_date2 = endDateTime
        tmpwdw = window
        while missing_data==False:
                    
            if window_units.upper in ['MON', 'M', 'MONTH']:
                checkDate = window_start_date2 + relativedelta(months=tmpwdw)
            elif window_units.upper in ['YR', 'Y', 'YEAR']:
                checkDate = window_start_date2 + relativedelta(years=tmpwdw)
            elif window_units.upper in ['DY','D', 'DAY']:
                checkDate = window_start_date2 + relativedelta(days=tmpwdw)
            else:
                checkDate = window_start_date2 + relativedelta(months=tmpwdw)
            
            # get the date list based on start date and end date
            tmp_pidx = pnd.period_range(start=window_start_date,
                                        end=dt.datetime.strftime(checkDate, '%Y-%m-%d %H:%M'),
                                        freq=self.timeUnitLookup())
            #print(tmp_pidx)
            ntimes= len(tmp_pidx) 
            tmpdate = dt.datetime.strftime(window_start_date2, '%d%b%Y')
            
            if ctime24:
                tmptime = '2400'
            else:
                tmptime = dt.datetime.strftime(window_start_date2, '%H%M')
                
            #print(tmpdate, ntimes)
            [nvals, vals, cunits, ctype, iofset, istat] = dss.read_regts(self.DSSFile.ifltab, self.Cpath, tmpdate, tmptime, ntimes)
            
            for i,v in enumerate(vals):
                #print(v)
                if v==-901. or v==-902.:  # found missing data - assume that first missing value means the end of the record
                                          # at least as much as we care about it - CalSim will error out if there's a value missing 
                    missing_data = True   # that it needs
                    tmpdate = tmp_pidx[i-1]  # get the date right before the missing value
                    #print(tmpdate, v)
                    break
                else:
                    tmpdate = tmp_pidx[-1]
                
            edate = dt.datetime(tmpdate.year, tmpdate.month, tmpdate.day, 0,0)
        
            if edate == checkDate: # we've found values back through the first window, try another window's length
                tmpwdw +=window
                window_start_date2 = checkDate
                window_start_date = dt.datetime.strftime(checkDate, '%Y-%m-%d')
                
            else:
                missing_data = True      
        
        self.RecordStart = bdate
        self.RecordEnd = edate
        
        #return([bdate, edate]) #, tmp_pidx, vals])
        
    def getRTS(self, stDateTime, ntimes=None, endDateTime=None,
               ctime24=True):
        
        if type(stDateTime) != dt.datetime:
            self.RTS = None
            print("couldn't parse start time - expected a datetime object")
        else:
            cdate = dt.datetime.strftime(stDateTime, '%d%b%Y') #'31Oct1921'
            if self.TimeUnit in ['HOUR','MINUTE']:
                psdate = dt.datetime.strftime(stDateTime, '%Y-%m-%d %H:%M') #format for creating a date list
            else:
                psdate = dt.datetime.strftime(stDateTime, '%Y-%m-%d') #format for creating a date list
            if ctime24:
                ctime = '2400'
            else:
                ctime = dt.datetime.strftime(stDateTime, '%H%M')
        
        
        # try to use an end date to determine number of values
        if endDateTime==None:
            # if no end datetime window, then check for a number of times/records
            if ntimes==None or ntimes==0:
                # if not that either, for now, just return empty with a warning, i guess
                self.RTS = None
                print("Not enough info to get RTS\nProvide either a number of records or a complete time window")
                return
            else:
                # get the date list based on start date and number of times
                pidx = pnd.period_range(start=psdate, periods=ntimes, freq=self.timeUnitLookup())
        else:
            if self.TimeUnit in ['HOUR','MINUTE']:
                pedate = dt.datetime.strftime(endDateTime, '%Y-%m-%d %H:%M')
            else:
                pedate = dt.datetime.strftime(endDateTime, '%Y-%m-%d')
            # get the date list based on start date and end date
            pidx = pnd.period_range(start=psdate, end=pedate, freq=self.timeUnitLookup())
            ntimes= len(pidx)
        
        # check that the DSS file is open, open if not
        if not self.DSSFile.IsOpen:
            self.DSSFile.openDSS()
        
        [nvals, vals, cunits, ctype, iofset, istat] = dss.read_regts(self.DSSFile.ifltab, self.Cpath, cdate, ctime, ntimes)
        
        if istat!=0:
            print('istat returned [%s] was not zero - check data for %s' %(istat, self.Cpath)) #assert istat==0
        
        dtList = af.perToDTlist(pidx, include_time=True)
        
        self.Units = cunits
        self.RecordType = ctype
        self.RTS = pnd.Series(data=vals, index=dtList)
        self.istat = istat
        

    def setRTS(self, cdate, ctime, dbl=True):
        '''
            UPDATE: this function takes only a dssVar object and a specified
                    start date/time
                    these things are prepared by the cs3 function setSVts
                     - all the checking and wrangling
                    of data and headers happens there - this function just
                    does the heclib function callss
            
            
        '''
        
        cpath = self.Cpath
        vals = self.RTS
        cunits = self.Units
        ctype = self.Type
        
        coords = self.SuppInfo['coords']
        icdesc = self.SuppInfo['icdesc']
        csupp = self.SuppInfo['csupp']
        ctzone = self.SuppInfo['ctzone']
            
        # now, write to DSS file at the desired precision
        if dbl:
            istat = dss.write_regtsd(self.DSSFile.ifltab, cpath, cdate, ctime, 
                                     vals, cunits, ctype, coords=coords, 
                                     icdesc=icdesc, csupp=csupp, ctzone=ctzone)
        else:
            istat = dss.write_regts(self.DSSFile.ifltab, cpath, cdate, ctime, 
                                    len(vals), vals, cunits, ctype)
        
        if istat !=0:
            print("Error writing {0} to DSS file. istat = {1}".format(cpath, istat))
            return(istat)
        else:
            return(istat)
            

def dssify(df_in,B,C,units, A='CALSIM',E='1MON',F='2020D09E',
           datatype='PER-AVER'):
    
    col_mindex = pnd.MultiIndex(levels=[[]]*7,
                             codes=[[]]*7,
                             names=[u'A', u'B', u'C',u'E',u'F',u'Type',u'Units'])
    
    tmpDF = pnd.DataFrame(index=df_in.index, columns = col_mindex) 
    tmpDF[A, B, C, E, F, datatype,units] = df_in.values   
    
    return(tmpDF)
    
        
def CFS_TO_TAF(startDate, endDate, freq='M'):
    ts = pnd.date_range(start=startDate, end=endDate, freq=freq)
    cfs_taf_ts = ts.map(lambda x: (86400./43560.)*x.day/1000)
    return(cfs_taf_ts)



class wresl_util:

    def removeComments(string):
        # from http://stackoverflow.com/questions/2319019/using-regex-to-remove-comments-from-source-files 2017.03.16 jmg
        string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurance streamed comments (/*COMMENT */) from string
        string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurance singleline comments (//COMMENT\n ) from string
        return(string)
    
    def read_wresl_connectivity(fp):
        '''
            Reads wresl CalSim3 connectivity file, returns dictionary of connecvity
        '''
        with open(fp, 'r') as cf:
            lins = cf.readlines()
    
        # make a dictionary for all the nodal connectivity to live
        connectivity_ins = {}  # key: Node; Values = [Ins] and [Outs ]   
        connectivity_outs = {}
        l = lins[0]
        n = 0
        while n < len(lins):
            print("top loop n: %s" %n)
            l = lins[n]
            if l[0]=='!' or l.strip()=='':
                # comment line
                #print(l)
                advcn = 1
                next
            elif '/*' in l[0:5]:
                # start of a comment block
                while '*/' not in l:
                    n +=1
                    l = lins[n]
                    print(n)
                advcn = 2
                pass
            else:
                #print("shouldnt' be here")
                # remove comments here first before we do anything else
                l = removeComments(l)
                cstrt = l.find('continuity')
                cend = l.find('{')
                nodeName = l[cstrt+10:cend].strip()
                if l.find('}')<0:
                    l2 = lins[n+1]
                    advcn = 2
                    l = l + l2
                else:
                    advcn = 1
                eqend = l.find('}')
                # START - parse continuity equations for ins and outs
                #eqntmp = l[cend+1:eqend]
                #eqntmp = removeComments(eqntmp)
                eqn = l[cend+1:eqend].split()
                outs = []
                ins = []
                RHS = False
                for i in range(len(eqn)):
                    if not RHS:
                        if i==0:
                            if eqn[i]=='-':
                                outs.append(eqn[i+1])
                                advci = 2
                            else:
                                ins.append(eqn[i])
                                advci = 1
                        else:
                            if eqn[i]=='-':
                                outs.append(eqn[i+1])
                                advci = 2
                            elif eqn[i]=='+':
                                ins.append(eqn[i+1])
                                advci = 2
                            elif eqn[i]=='=':
                                RHS = True
                                # assume the first term on the RHS of the equals sign is a postive value
                                # and thus an 'out'
                                outs.append(eqn[i+1])
                                advci = 2
                            else:
                                pass                    
                        i = i + advci
                    else:
        
                        if eqn[i]=='-':
                            if eqn[i+1][0:3]!='Rsd':
                                ins.append(eqn[i+1])
                            advci = 2
                        elif eqn[i]=='*':  # assume multiplication is for storage term unit conversion, ignore
                            advci = 2
                        elif eqn[i]=='+':
                            if eqn[i+1][0:3]!='Rsd':
                                outs.append(eqn[i+1])
                            advci = 2
                        else:
                            pass
                        i = i+ advci
        
                # END - parse continuity equation for ins and outs  
                
                    # add ins and outs lists to the dictionary
                    connectivity_ins[nodeName] = ins
                    connectivity_outs[nodeName] = outs
        
            n = n + advcn # next continuity line
        return([connectivity_ins, connectivity_outs])
    
    
    def find_wresl_timeseries(fp):
        '''
            Reads wresl file, finds instances of timeseries, returns dict of info
        '''
        with open(fp, 'r') as cf:
            lins = cf.readlines()
    
        # make a dictionary for time series info to live
        ts_vars ={}  # key = var name; values=[kind, units, convert]
        l = lins[0]
        n = 0
        while n < len(lins):
            print("top loop n: %s" %n)
            l = lins[n]
            if l[0]=='!' or l.strip()=='':
                # comment line
                #print(l)
                advcn = 1
                next
            elif '/*' in l[0:5]:
                # start of a comment block
                while '*/' not in l:
                    n +=1
                    l = lins[n]
                    print(n)
                advcn = 2
                pass
            else:
                #print("shouldnt' be here")
                # remove comments here first before we do anything else
                l = removeComments(l)
                
                
                ts_ptrn = r"(define)\s*(?P<var>\w*)\s*\{timeseries\s*(?P<kind>(kind)?\s*'(\w*)')\s*(?P<units>(units)?\s*'(\w*)')\s*(?P<convert>(convert)?\s*'?(\w*)?)"
                
                tmp = re.search(ts_ptrn, l, re.I)  # make search case-insensitive
                
                if tmp != None: # if search string contains the pattern
                    td = tmp.groupdict()
                    var = td['var']
                    dupCntr = 1
                    if var in ts_vars.keys(): # checking to see if by some chance the variable has already been found elsewhere
                        print("Variable {0} already in dictionary".format(var))
                        newvar = var +'_{0}'.format(dupCntr)
                        while newvar in ts_vars.keys():
                            dupCntr+=1
                            newvar = var +'_{0}'.format(dupCntr)
                        var = newvar
                        
                    if td['kind']!='':
                        kind = td['kind'].split("'")[1]
                    else:
                        kind=''
                    if td['units']!='':
                        units = td['units'].split("'")[1]
                    else:
                        units=''
                    if td['convert']!='':
                        convert= td['convert'].split("'")[1]
                    else:
                        convert=''
                    
                    ts_vars[var] = [kind, units, convert]
                    

        return(ts_vars)


def read_playbook(yamlFP):
    import yaml
    '''
        input: full filepath to valid yaml file specifying one or more sets of
               actions to take, with the adequate information to accompany eac
               action; 
               
               example: for copying records from one file to another, you'd include
                       the paths to the source and destination files along with the
                       list of record paths to copy
    '''
    plybk = []
    with open(yamlFP,'r') as of:
        tmp = yaml.safe_load_all(of)  #, Loader=yaml.SafeLoader)
        for t in tmp:
            plybk.append(t)
    return(plybk)

def run_playbook(plybk):
    '''
        executes the actions in the playbook (or selected compnent thereof)
        currently supported actions:
            copyDSS - copy records in PathList from SourceFP to DestinationFP; 
                      TODO:  if no paths are provided in PathList, copies whole file(?)
                      
            study/setup - sets up CalSim objects and data retrieval for plotting/analysis
                          - this was moved to the csPlots.py file/namespace
    '''
    if type(plybk)==list:
        for ply in plybk: # iterate through each of multiple actions if provided
            if ply['action_type']=='copyDSS':
                if 'SourceFP' not in ply:
                    print("Error: 'SourceFP' not provided in playbook for %s" %ply['name'])
                    return
                if 'DestinationFP' not in ply:
                    print("Error: 'DestinationFP' not provided in playbook for %s" %ply['name'])
                    return
                if 'Paths' not in ply:
                    print("Warning: 'Paths' not provided in playbook for %s; assuming the whole file is to be copied" %ply['name'])
                srcFP= ply['SourceFP']
                destFP = ply['DestinationFP']
                pathList = ply['Paths']
                npaths = len(pathList)
                print("Copying %s records\n:\tfrom: %s \n\tto: %s" %(npaths, srcFP, destFP))
                copy_DSS_record(srcFP, destFP, pathList)
                

                    

                    
                        
                    
                    
            

def copy_DSS_record(srcFP, destFP, pathList, debug=False):
    ''' 
        wrapper around the DSS record copy function
        assumes that the paths will be named the same in destination file as they
        are in teh source file
        *NOTE: because we have to specify the time (D) parts of the paths, assumes
               a set of decadal intervals consistent with CalSim SV/DV paths
               TODO: modify if necessary to work with differently intervaled
               CalSimHydro and related data types
    '''    
    [ifltabOLD, istat] =dss.open_dss(srcFP)
    [ifltabNEW, istat] = dss.open_dss(destFP)

    times = ['01JAN1920',
             '01JAN1930',
             '01JAN1940',
             '01JAN1950',
             '01JAN1960',
             '01JAN1970',
             '01JAN1980',
             '01JAN1990',
             '01JAN2000',
             '01JAN2010']

    for p in pathList:
        if debug:
            print("copying %s from %s to %s" %(p, srcFP, destFP))
        [dum1, a, b, c, d, e, f, dum2] = p.split('/')
        for t in times:
    #        cpath='/CALSIM/I1/FLOW-INFLOW/%s/1MON/2020D09E/' %t
            cpath = '/'+'/'.join([a,b,c,t,e,f]) + '/'
            istat = dss.copyRecord(ifltabOLD, ifltabNEW, cpath, cpath)
            if istat != 0:
                print(istat)
    
    dss.close_dss(ifltabNEW)
    dss.close_dss(ifltabOLD)
    