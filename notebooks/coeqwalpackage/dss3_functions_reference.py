# -*- coding: utf-8 -*-
"""
Python wrapper for DSS HEC-LIB functions
    - Originally developed summer 2017
    - Functions added as needed, although not all are guaranteed to work as
      intended/needed

Updated 2018.04.05-2018.04.10 to include write capabilities, additional catalog
                              functions
                              
TODO:  Error catching with helpful return info (based on HECLIB documentation)
TODO:  Sort out catalog creation and write functionality  
TODO:  Add/retrieve more metadata with each record on read/write             
TODO:  Sort out single/double precision consistency on read/write operations                 

@author: jmgilbert
"""

import ctypes as C
import os, sys
import numpy as np
import struct
#import numba
#from numba import jit

global CATALOGUNIT
global CONDUNIT

CATALOGUNIT=12
CONDUNIT = 13

#excelversion = False 
excelversion = True


if not excelversion:
    mingwversion = True
else:
    mingwversion = False

if excelversion:
    #sys.path.append(r'C:\Users\jgilbert\01_Programs\exceldssadd-inpackage_feb2016\ExcelDSSAdd-inPackage_Feb2016\libraries')
    dll_path = r'../ExcelDSSAdd-inPackage_Feb2016/libraries/64-bit/heclib_x64.dll'
    #dll_path = r'C:\Users\jgilbert\01_Programs\HEC\hec-dss\heclib\VB_interface\x64\Release\heclib.dll'

if mingwversion:
    #sys.path.append(r'C:\msys64\mingw64\bin')
    sys.path.append(r'D:\02_Projects\CalSim\util\CalSim_Utilities\Python_Functions\HecLibSource\mingw64_build\heclib\release')
    dll_path = r"D:\02_Projects\CalSim\util\CalSim_Utilities\Python_Functions\HecLibSource\mingw64_build\heclib\release\heclib2.dll"

dsslib = C.cdll.LoadLibrary(dll_path)
#C.cdll.LoadLibrary(r"D:/02_Projects/CalSim/util/CalSim_Utilities/Python_Functions/HecLibSource/mingw64_build/heclib/release/libgfortran-5.dll")
#C.cdll.LoadLibrary('libquadmath-0.dll')
#%%
def open_dss(fpi):
    if excelversion:
        zopen = getattr(dsslib, 'ZOPEN_')
        
    if mingwversion:
        zopen = getattr(dsslib, 'zopen_')
        
    zopen.argparse = [C.c_long*600, C.c_char*800, C.c_long, C.c_long]
    zopen.restype = None
    ifltab = (C.c_long * 600)()    
    iostat = C.c_long()
    #fp = (C.c_char*800)(*fpi)
    #print(fp)
    zopen(C.byref(ifltab), fpi.encode('ascii'), C.byref(iostat), len(fpi.encode('ascii')))
    return([ifltab, iostat.value])
    
def close_dss(ifltab):
    if excelversion:
        zclose = getattr(dsslib, 'ZCLOSE_')
        
    if mingwversion:
        zclose = getattr(dsslib, 'zclose_')
        
    zclose.argparse = [C.c_long*600]
    stdout = zclose(ifltab)
    return(stdout)
    
def dss_exists(fp):
    if excelversion:
        zfname = getattr(dsslib, 'ZFNAME_')
    if mingwversion:
        zfname = getattr(dsslib, 'zfname_')
        
    zfname.argparse = [C.c_char*800, C.c_char*800, C.c_long, C.c_bool, C.c_long, C.c_long]
    # arguments: path in, path with extension if exists[out], length of returned name, 
    #            file exists logical, length of path in, length of path out
    zfname.restype = None
    plen = C.c_long()
    lexist = C.c_bool()
    new_fp = (C.c_char*800)()
    
    zfname(fp.encode('ascii'), new_fp, C.byref(plen),C.byref(lexist),len(fp.strip()),len(new_fp))
    
    #print(lexist.value)
    #print(new_fp)
    #print(plen)
    return([lexist.value, new_fp[0:].decode().strip()])
    

def read_regts(ifltab, cpath, cdate, ctime, nvalsi):
    if excelversion:
        zrrts = getattr(dsslib, 'ZRRTS_')
    if mingwversion:
        zrrts = getattr(dsslib, 'zrrts_')
# Args: (IFLTAB(600)  | CPATH*80   | CDATE   | CTIME   | NVALS   | VALUES   |  CUNITS   | CTYPE   | IOFSET   | ISTAT)
# DTYPE:(INTEGER(600) | CHAR*80    | CHAR*20 | CHAR*4  | INTEGER | REAL(NVALS)| CHAR*8  | CHAR*8  | INTEGER  | INTEGER)
#        INPUT        | INPUT      | INPUT   | INPUT   | INPUT   | OUTPUT   | OUTPUT    | OUTPUT | OUTPUT    | OUTPUT)
#    zrrts.argparse = [C.c_long*600, C.c_char*800, C.c_char*20, C.c_char*4, 
#                      C.c_long, C.c_float, C.c_char*8, C.c_char*8, C.c_long, C.c_long,
#                      C.c_long, C.c_long, C.c_long, C.c_long, C.c_long]  # lengths of: cpath, cdate, ctime, cunits, ctype
#    zrrts.restype = None
#    
#    nvals = C.c_long(nvalsi)  
#    vals = (C.c_float*nvalsi)()
#    cunits = (C.c_char*8)()
#    ctype = (C.c_char*8)()
#    iofset = C.c_long()
#    istat = C.c_long()
    zrrts.argparse = [C.c_int32*600, C.c_char*80, C.c_char*20, C.c_char*4, 
                      C.c_int32, C.c_float, C.c_char*8, C.c_char*8, C.c_int32, C.c_int32,
                      C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32]  # lengths of: cpath, cdate, ctime, cunits, ctype
    zrrts.restype = None
    
    ecpath = cpath.upper().encode('ascii')
    ecdate =cdate.upper().encode('ascii')
    ectime = ctime.upper().encode('ascii')
    nvals = C.c_int32(nvalsi)  
    vals = (C.c_float*nvalsi)()
    cunits = (C.c_char*8)()
    ctype = (C.c_char*8)()
    iofset = C.c_int32()
    istat = C.c_int32()
        
    l_ecpath = len(ecpath)
    l_ecdate = len(ecdate)
    l_ectime = len(ectime)
    l_cunits = 8
    l_ctype = 8
    
    zrrts(ifltab, cpath.encode('ascii'), cdate.upper().encode('ascii'), ctime.upper().encode('ascii'), 
          C.byref(nvals), C.byref(vals), cunits, ctype,C.byref(iofset), C.byref(istat), 
          len(cpath), len(cdate), len(ctime), len(cunits), len(ctype))
    
#    zrrts(C.byref(ifltab), ecpath, ecdate, ectime , C.byref(nvals), C.byref(vals), C.byref(cunits), C.byref(ctype),
#          C.byref(iofset), C.byref(istat), 
#          l_ecpath, l_ecdate, l_ectime, l_cunits, l_ctype)   
    
    return([nvals.value, np.float32(vals), cunits[0:].decode('utf-8').strip(), ctype[0:].decode('utf-8').strip(), iofset.value, istat.value])

def read_regtsd(ifltab, cpath, cdate, ctime, nvalsi, lgetdob_in=True):
    '''
    # Args: (IFLTAB(600)  | CPATH*80   | CDATE   | CTIME   | KVALS   | NVALS   | 
    #        INPUT        | INPUT      | INPUT   | INPUT   | INPUT   | OUTPUT   | OUTPUT    | OUTPUT | OUTPUT    | OUTPUT)
    # DTYPE:(INTEGER(600) | CHAR*80    | CHAR*20 | CHAR*4  | INTEGER | INTEGER | 
    
    # Args: LGETDOB | LFILDOB | sVALUES    | dVALUES       | JQUAL | LQUAL | LQREAD |  
    #       LOGICAL | LOGICAL | REAL(NVALS)| Double(NVALS) |INTEGER|LOGICAL| LOGICAL|

    #Args: CUNITS   | CTYPE   | CSUPP   | IOFSET   | JCOMP | ITZONE |CTZONE| COORDS |
    #       CHAR*8  | CHAR*8  | CHAR*80 | INTEGER  |INTEGER|INTEGER |CHAR*30 | DOUBLE |
    
    #Args: ICDESC |LCOORDS | ISTAT | L_CPATH | L_CDATE | L_CTIME | L_CUNITS | L_CTYPE | L_CSUPP | L_CTZONE)
    #      INTEGER|INTEGER |INTEGER| INTEGER | INTEGER | INTEGER | INTEGER  | INTEGER | INTEGER | INTEGER 
    '''
    if excelversion:
        zrrtsc = getattr(dsslib, 'ZRRTSC_')
    
    if mingwversion:
        zrrtsc = getattr(dsslib, 'zrrtsc_')
        
    zrrtsc.argparse = [
            C.c_long*600, C.c_char*800, C.c_char*20, C.c_char*4, C.c_long, C.c_long,  # IFLTAB, CPATH, CDATE, CTIME, KVALS, NVALS
            C.c_bool, C.c_bool, C.c_float, C.c_double, C.c_long, C.c_bool, C.c_bool,  # LGETDOB, LFILDOB, sVALUES, dVALUES, JQUAL, LQUAL, LQREAD
            C.c_char*8, C.c_char*8, C.c_char*800, C.c_long, C.c_long, C.c_long, C.c_char*30, C.c_double, # CUNITS, CTYPE, CSUPP, IOFSET, JCOMP, ITZONE, CTZONE, COORDS
            C.c_long, C.c_long, C.c_long, C.c_long, C.c_long, C.c_long, C.c_long, C.c_long, C.c_long, C.c_long ] # ICDESC, LCOORDS< ISTAT, L_CPATH, L_CDATE, L_CTIME, L_CUNITS, L_CTYPE, L_CSUPP L_CTZONE
    
    zrrtsc.restype = None
    
    nvals = kvals = C.c_long(nvalsi)
    sVals = (C.c_float*nvalsi)()
    dVals = (C.c_double*nvalsi)()
    lgetdob = C.c_bool(lgetdob_in)
    lfildob = C.c_bool()
    jqual = (C.c_long*nvalsi)()
    lqual = C.c_bool()
    lqread = C.c_bool()
    csupp = (C.c_char*800)()
    cunits = (C.c_char*8)()
    ctype = (C.c_char*8)()
    iofset = C.c_long()
    istat = C.c_long()
    jcomp = C.c_long()
    itzone = C.c_long()
    ctzone = (C.c_char*30)()
    coords = (C.c_double*3)()
    icdesc = (C.c_long*6)()
    lcoords = C.c_bool()
    
    ecpath = cpath.encode('ascii')
    ecdate = cdate.encode('ascii')
    ectime = ctime.encode('ascii')
    
    l_cpath = len(ecpath)
    l_cdate = len(ecdate)
    l_ctime = len(ectime)
    l_cunits = len(cunits)
    l_ctype = len(ctype)
    l_csupp = len(csupp)
    l_ctzone = len(ctzone)
    
    zrrtsc(ifltab, ecpath, ecdate, ectime, C.byref(kvals), C.byref(nvals), C.byref(lgetdob), C.byref(lfildob),   #8
           C.byref(sVals), C.byref(dVals), C.byref(jqual), C.byref(lqual), C.byref(lqread),    #+5 = 13
           cunits, ctype, csupp, C.byref(iofset), C.byref(jcomp), C.byref(itzone),            #+6 = 19
           ctzone, C.byref(coords), C.byref(icdesc), C.byref(lcoords), C.byref(istat),        #+5 = 24
           l_cpath, l_cdate, l_ctime, l_cunits, l_ctype, l_csupp, l_ctzone)                   #7 = 31
    
    # icdesc is a 6-item list with information on the coordinates
    # icdesc[0] = Coordiante system: (following values follow the DSSVue convention - as far as I can tell though, you can use whatever scheme you want)
    #               0 = No coordinates set
    #               1 = Lat/Long
    #               2 = State Plane/FIPS
    #               3 = State Plane/ADS
    #               4 = UTM
    #               5 = Local/other
    # icdesc[1] = coordinate ID (some integer)
    # icdesc[2] = Datum Units 
    #             0 = not specified
    #             1 = English (ft or miles i guess?)
    #             2 = SI   (m or km?)
    #             3 = Decimal degrees
    #             4 = degrees, minutes, seconds
    # icdesc[3] = horizontal datum
    #          0 = unset
    #          1 = NAD83
    #          2 = NAD27
    #          3 = WAS84
    #          4 = WAS72
    #          5 = local/other
    #
    
    # compile coordinate information into a single dictionary for clarity
    coords_info = icdesc_to_dict(coords, icdesc)
    
    return([nvals.value, dVals, cunits[0:].decode('utf-8').strip(), ctype[0:].decode('utf-8').strip(),
            iofset.value, istat.value, csupp[0:].decode('utf-8'), coords_info, ctzone[0:].decode('utf-8'), jqual, lfildob, itzone])
    
#            ByVal strCDate As String, ByVal strCTime As String, _
#            lngKVals As Long, lngNVals As Long, lngLGETDOB As Long, _
#            lngLFILDOB As Long, sngValues As Single, dblValues As Double, _
#            lngJQUAL As Long, lngLQUAL As Long, lngLQREAD As Long, _
#            ByVal strCUnits As String, ByVal strCType As String, _
#            ByVal strCSUPP As String, lngIOFSET As Long, _
#            lngJCOMP As Long, lngITZONE As Long, ByVal strCTZONE As String, _
#            dblCOORDS As Double, lngICDESC As Long, lngLCoords As Long, _
#            lngISTAT As Long, ByVal lngL_CPath As Long, ByVal lngL_CDate As Long, _
#            ByVal lngL_CTime As Long, ByVal lngL_CUnits As Long, _
#            ByVal lngL_CType As Long, ByVal lngL_CSUPP As Long, ByVal lngL_CTZONE As Long    

def icdesc_to_dict(coords, icdesc):
    '''
    Converts the ICDESC list as returned from a DSS file to a dictionary with 
    keys as identifiers, according to DSSVue scheme
    '''
    
    coords_info = {}
    
    coords_info['X_Long'] = coords[0]
    coords_info['Y_Lat'] = coords[1]
    
    coordsys = icdesc[0]
    if coordsys==0:
        coords_info['CoordSys'] = ['No coordinates set', 0]
    elif coordsys==1:
        coords_info['CoordSys'] = ['LatLong', 1]
    elif coordsys==2:
        coords_info['CoordSys'] = ['State Plane/FIPS', 2]
    elif coordsys==3:
        coords_info['CoordSys'] = ['State Plane/ADS', 3]
    elif coordsys==4:
        coords_info['CoordSys'] = ['UTM', 4]
    elif coordsys==5:
        coords_info['CoordSys'] = ['Local/other', 5]
    else:
        coords_info['CoordSys'] = ['Hard to say, really', coordsys]
    
    coordID = icdesc[1]
    coords_info['CoordID'] = coordID
    
    datumUnits = icdesc[2]
    if datumUnits==0:
        coords_info['DatumUnit'] = ['Not specified', 0]
    elif datumUnits==1:
        coords_info['DatumUnit'] = ['English', 1]
    elif datumUnits==2:
        coords_info['DatumUnit'] = ['SI', 2]
    elif datumUnits==3:
        coords_info['DatumUnit'] = ['Decimal Degrees', 3]
    elif datumUnits==4:
        coords_info['DatumUnit'] = ['degrees, minutes, seconds', 4]
    else:
        coords_info['DatumUnit'] = ['Unknown', None]
    
    datum = icdesc[3]
    #print("Datum from file is: %s" %datum)
    if datum==0:
        coords_info['Datum'] = ['Not specified', 0]
    elif datum==1:
        coords_info['Datum'] = ['NAD83', 1]
    elif datum==2:
        coords_info['Datum'] = ['NAD27', 2]
    elif datum==3:
        coords_info['Datum'] = ['WAS84', 3]
    elif datum==4:
        coords_info['Datum'] = ['WAS72', 4]
    elif datum==5:
        coords_info['Datum'] = ['local/other', 5]
    else:
        coords_info['Datum'] = ['Unknown', None]
    
    
    
    return(coords_info)
        
        
    
def open_catalog(fp, icunitin, lgenca_in=True, lgencd_in=False):   
    # need to open catalog before you can read it - this gets called from 'get_catalog'
    if excelversion:
        zopnca = getattr(dsslib, 'ZOPNCA_')
        
    if mingwversion:
        zopnca = getattr(dsslib, 'zopnca_')
    # Args:   CDSSFI  | ICUNIT   | LGENCA   |LOPNCA   | LCATLG   | ICDUNT   | LGENCD  |LOPNCD   | LCATCD  |NRECS  | *Len(CDSSFI)
    #         CHAR*64 | INT      | LOGICAL  | LOGICAL  | LOGICAL | INT      | LOGICAL |LOGICAL  |LOGICAL  | INT   | INT
    #          INPUT  | INPUT    |INPUT     | OUT      | OUT     | INPUT    | INPUT  | OUTPUT   | OUTPUT   | OUTPUT  | IN
    zopnca.argparse = [C.c_char,  C.c_long, C.c_bool, C.c_bool, C.c_bool, C.c_long, C.c_bool, C.c_bool, C.c_bool, C.c_long, C.c_long ]
    zopnca.restype = None
    icunit = C.c_long(icunitin)
    lgenca = C.c_bool(lgenca_in)
    lopnca = C.c_bool()
    lcatlg = C.c_bool()
    icdunt = C.c_long()  #@jmg 2017.11.30 - set this explicitly
    lgencd = C.c_bool(lgencd_in)
    lopncd = C.c_bool()
    lcatcd = C.c_bool()
    nrecs = C.c_long()
    #C.byref(icunit)
    zopnca(fp.encode('ascii'), C.byref(icunit) , C.byref(lgenca), C.byref(lopnca), C.byref(lcatlg), \
           C.byref(icdunt), C.byref(lgencd), C.byref(lopncd), C.byref(lcatcd), C.byref(nrecs), len(fp.encode('ascii')))
    return([lgenca.value, lopnca.value, lcatlg.value, lgencd.value,lopncd.value, lcatcd.value, nrecs.value])
   
def read_catalog(lopnca, icunitin=12):
    # zrdcat - read pathnames from catalog
    # first - check to see if catalog is open - if not, prompt to open it
    if not lopnca:
        print("Catalog is not open - do this first using open_catalog function")
        return([None, None])
    
    if excelversion:
        zrdcat = getattr(dsslib, 'ZRDCAT_')
    
    if mingwversion:
        zrdcat = getattr(dsslib, 'zrdcat_')
        
    # ARGS:  ICUNIT   |  LALL | IOUNIT  |CTAGS        | NDIM  | CPATHS | NPATHS  | NFOUND | *Len(ctags) | *Len(cpath)
    #DTYPES:  INT    | LOGICAL| INT     |CHAR(NDIM)*8 | INT   | 
    zrdcat.argparse= [C.c_long, C.c_bool, C.c_long, C.c_char, C.c_long, C.c_char, C.c_long, C.c_long, C.c_long, C.c_long ]
    icunit = C.c_long(icunitin)
    lall = C.c_bool(True)
    iounit = C.c_long(0)
    ndim = C.c_long(1) #nrecs  #C.c_long()
    ctags = (C.c_char*8)()
    cpaths = (C.c_char*80)()
    npaths = C.c_long()
    nfound = C.c_long(1)
    path_list = []
    

    while nfound.value ==1:
        zrdcat(C.byref(icunit), C.byref(lall), C.byref(iounit), ctags, C.byref(ndim),cpaths, C.byref(npaths), C.byref(nfound), len(ctags), len(cpaths) )
        #print(nfound.value)
        #print(cpaths.value)
        #print(ctags.value)
        nfoundv = nfound.value
        if nfoundv==1:
            path_list.append(cpaths.value.decode('utf-8').strip())
            #print(cpaths.value) 
    catstatus  = fortran_close_file(icunitin)  # close the catalog file
    if catstatus==0:
        lopnca= False
    else:
        lopnca= True
    return([path_list, lopnca])

def fortran_close_file(iunit_in):
    if excelversion:
        fortclose = getattr(dsslib,'FORTRANCLOSE_')
    if mingwversion:
        fortclose = getattr(dsslib, 'fortranclose_')
        
    # ARGES: IUNIT
    # DTYPES: INT
    fortclose.argparse = [C.c_long]
    iunit = C.c_long(iunit_in)
    status = fortclose(C.byref(iunit))
    return(status)
    
  
def write_regts(ifltab, cpath, cdate, ctime, nvalsi, valsi, cunits, ctype,_IPLAN=0):
    '''
    This function writes a regular time series, stored as a Python list, to a DSS file.
    The target DSS file to which the data should be written needs to be opened before 
    it can be written to. Opening the file (using `open_dss` function) will create
    a new file if the file doesn't already exist.
    INPUTS:
    -------
    ifltab: should be a long integer retrieved when opening a file
    cpath:  string of a properly formatted DSS pathname, should have 7 forward slashes /, 
            but no D-part (start date - this comes from the cdate argument)
    cdate:  a string representing the start date - should be of the form 
            "DDmonYYYY", e.g. "31Oct1999" for Oct 31, 1999
    ctime:  a string representing start time in hours since midnight on `cdate`,
            common practice is to use "2400" for monthly data
    nvalsi: number of values to be written, as an integer; the end date is defined
            by the start date, number of values, and the period length given in 
            the E-part of the variable path
    valsi:  a python list of length `nvalsi` to be written to the DSS file; missing values
            should be filled with -901.0 
    cunits: a string indicating units of the time series values
    ctype:  a string indicating the time period summary type that the values represent,
            commonly "PER-AVG" for period average
    _IPLAN:  (default = 0) - flag to indicate whether to write over existing data
            0 =  always write over existing data
            1 = only replace missing data flags in the record, as indicated by (-901)
            4 = if an input value is missing (-901), do not allow it to replace a
                non-missing value
    OUTPUTS:
    --------
    istat:   flag indicating if data was stored successfully (0), if all data was missing (-901),
             or other "fatal" errors(values >10, see HECLIB documentation for details for now...
             someday I'll include code to catch the errors and tell you something useful)
    ''' 
    if excelversion:
        zsrts = getattr(dsslib, 'ZSRTS_')
    if mingwversion:
        zsrts = getattr(dsslib, 'zsrts_')
    
# Args: (IFLTAB(600)  | CPATH*80   | CDATE   | CTIME   | NVALS   | VALUES   |  CUNITS   | CTYPE   | IPLAN   | ISTAT)
# DTYPE:(INTEGER(600) | CHAR*80    | CHAR*20 | CHAR*4  | INTEGER | REAL(NVALS)| CHAR*8  | CHAR*8  | INTEGER  | INTEGER)
#        INPUT        | INPUT      | INPUT   | INPUT   | INPUT   | OUTPUT   | OUTPUT    | OUTPUT | OUTPUT    | OUTPUT)
    zsrts.argparse = []


    zsrts.argparse = [C.c_long*600, C.c_char*800, C.c_char*20, C.c_char*4, C.c_long, (C.c_float*nvalsi)(), C.c_char*8, C.c_char*8, C.c_long, C.c_long, \
                      C.c_long, C.c_long, C.c_long, C.c_long, C.c_long]  # lengths of: cpath, cdate, ctime, cunits, ctype
    zsrts.restype = None
    valsi = np.float32(valsi) # enforce 32-bit single precision on the values
    
    nvals = C.c_long(nvalsi)  
    vals = (C.c_float*nvalsi)(*valsi)
    #print(vals)  # for debugging
    cunits = cunits  #(C.c_char*8)(cunits)
    ctype = ctype  #(C.c_char*8)(ctype)
    IPLAN = C.c_long(_IPLAN)
    istat = C.c_long()
    zsrts(C.byref(ifltab),
          cpath.encode('ascii'),
          cdate.encode('ascii'),
          ctime.encode('ascii'),
          C.byref(nvals),
          C.byref(vals), 
          cunits.encode('ascii'),
          ctype.encode('ascii'),
          C.byref(IPLAN), 
          C.byref(istat), 
          len(cpath), 
          len(cdate), 
          len(ctime), 
          len(cunits), 
          len(ctype))
    
    return([istat])
    
    
def create_catalog(ifltab, icunit, icdunt, inunit, cinstr, labrev, lsort): #, lcdcat, nrecs ):
    if excelversion:
        zcat = getattr(dsslib, 'ZCAT_')
    if mingwversion:
        zcat = getattr(dsslib, 'zcat_')
        
    zcat.argparse = []

    zcat.argparse = [C.c_long*600,C.c_long, C.c_long, C.c_long, C.c_char*800, C.c_bool, C.c_bool, C.c_bool,
                     C.c_long]  # length of cinstr input string

    zcat.restype = None
    
    icunit = C.c_long(icunit)
    icdunt = C.c_long(icdunt)
    inunit= C.c_long(inunit)
    cinstr = (C.c_char*800)(*cinstr.encode('ascii'))
    labrev = C.c_bool(labrev)
    lsort = C.c_bool(lsort)
    lcdcat = C.c_bool()
    nrecs = C.c_long()
    

    #C.byref(icunit)
    zcat(C.byref(ifltab), C.byref(icunit) , C.byref(icdunt), C.byref(inunit), cinstr, \
           C.byref(labrev), C.byref(lsort), C.byref(lcdcat), C.byref(nrecs), len(cinstr))
    return([lcdcat.value, nrecs.value])
    
def get_catalog(fp):
    
    # the DLL used in the Excel Add-in, which is the one we reference here for these
    # functions, creates a 'lock' file (*.dsk) when catalog operations are made;
    # once this lock file is created in a session, any *outside* modifications
    # to the *.dsc or *.dsd files will result in calls to open, create, or read
    # catalogs not working properly; this happens with the catalog functions in
    # the excel add-in as well; 
    # the 'get_catalog' function can be called (and should work correctly) multiple
    # times within a session *as long as the *.dsc and *.dsd files are not modified
    # by another process*
    # as a precaution, we will try to delete any existing *.dsk files associated
    # with 'fp' upon calling this function - if the *.dsk file cannot be deleted,
    # a message will be returned indicating this condition
    
    dskfp = os.path.join(os.path.dirname(fp), os.path.basename(fp)[:-1]+'k')
    try:
        os.remove(dskfp)
    except:
        print("\n-------------------------------------------------------------\n"+ \
              "Could not remove the *dsk file - it is locked\nfor use by a previous " +\
              "call of this function \nor another system process (Excel add-in, I'm looking at you...)\n\n" +\
              "---DO NOT MODIFY THE CATALOG FILES (*.dsc, *.dsd) \n---WHILE RUNNING THIS PYTHON SESSION!!!\n" +\
              "================================================================\n")
    
    [ifltab, iostat] = open_dss(fp)  #open the DSS file to start
    if iostat!=0:
        print("couldn't open the dss file - exiting now...")
        return([None, None, None])
    
    dscfp = os.path.join(os.path.dirname(fp), os.path.basename(fp)[:-1]+'c')
    if os.path.exists(dscfp):
        if os.stat(dscfp).st_size==0:
            fortran_close_file(CATALOGUNIT)
            os.remove(dscfp)    
    [lgenca, lopnca, lcatlg, lgencd, lopncd, lcatcd, nrecs] = open_catalog(fp, CATALOGUNIT, lgenca_in=True, lgencd_in=True)
    
    if not lopnca:
        print("Couldn't open catalog - something is not right...exiting..")
        return([None,None, None])
    else:
        NStdCatRecs = NCatalogedRecs = nrecs
        ValidCat = lcatlg
    
    # add some checks for number of records - ZINQR vs what's returned with open_catalog
    
    # check if a valid catalog was found
    if ValidCat:
        proceed=True
        print("Valid catalog found: %s \nNumber of records: %s" %(ValidCat, nrecs))
    else:
        print("Creating a catalog...")

        [lcatcd, nrecs] = create_catalog(ifltab, CATALOGUNIT, CONDUNIT, 0,'',False,True)
        #        Call vbaZCAT(CATALOGFILENUMBER, 0)
        #                          12,           0
        #                     TargetFile,  SourceFile
        #                     CatUnit,     SourceUnit
        print("Created condensed catalog: %s \nNumber of records: %s" %(lcatcd, nrecs))
        
        if nrecs>0:
            proceed=True
        else:
            proceed=False
        
    if proceed:
        
        if nrecs==0:
            print("no records found in file...something is wrong again...exiting")
            fortran_close_file(CATALOGUNIT)
            fortran_close_file(CONDUNIT)
            close_dss(ifltab)
            return([None, None, None])
        else:
            [pathlist, lopnca] = read_catalog(lopnca, icunitin=CATALOGUNIT)
            close_dss(ifltab)
            fortran_close_file(CATALOGUNIT)
            fortran_close_file(CONDUNIT)
            return([pathlist, nrecs, lopnca])
    else:
        print("Couldn't create a catalog for some reason...exiting")
        close_dss(ifltab)
        fortran_close_file(CATALOGUNIT)
        fortran_close_file(CONDUNIT)
        return([None, None, None])
    
def get_datatype(ifltab, cpath):
    
    if excelversion:
        zdtype = getattr(dsslib, 'ZDTYPE_')
    if mingwversion:
        zdtype = getattr(dsslib, 'zdtype_')
# Args: (IFLTAB(600)  | CPATH*80   | NSIZE  | LEXIST  | CDTYPE | IDTYPE)
# DTYPE:(INTEGER(600) | CHAR*80    | INT    | LOGICAL | CHAR*3 | INT )
#        INPUT        | INPUT      | OUTPUT | OUTPUT  | OUTPUT | OUTPUT)
    zdtype.argparse = [C.c_long*600, C.c_char*80, C.c_long, C.c_bool, C.c_char*3, C.c_long, 
                       C.c_long, C.c_long]  # lengths of: cpath and cdtype
    zdtype.restype = None
    
    ecpath = cpath.upper().encode('ascii')
    lexist = C.c_bool()
    nsize = C.c_long()
    lexist = C.c_bool()
    cdtype = (C.c_char*3)()
    idtype = C.c_long()
    
    l_cdtype = 3    # the code for data type is assumed to always be 3 characters
    l_cpath = len(ecpath)
    
    if excelversion:
        zdtype(C.byref(ifltab), ecpath, C.byref(nsize), 
               C.byref(lexist), cdtype, C.byref(idtype), l_cpath, l_cdtype)
    if mingwversion:
        zdtype(C.byref(ifltab), ecpath, C.byref(nsize), C.byref(lexist), cdtype,C.byref(idtype), l_cpath,l_cdtype) #, l_cpath, l_cdtype)       
    
    return([lexist.value, cdtype.value.decode('utf-8').strip(), idtype.value, nsize.value])

    

def read_other(ifltab,cpath,kdata,kheadi=75, kheadu=0, kheadc=0):
    '''
        wrapper around the zreadx function - to read datasets that don't conform
        to the 'standard DSS converntions'
        
    '''
    if excelversion:
        zreadx = getattr(dsslib, 'ZREADX_')
    if mingwversion:
        zreadx = getattr(dsslib, 'zreadx_')
# Args: (IFLTAB(600)  | CPATH*80   | HEADI        | KHEADI  | NHEADI)
# DTYPE:(INTEGER(600) | CHAR*80    | REAL(KHEADI) |  INT   | INT   )
#        INPUT        | INPUT      | OUTPUT      | INPUT  | OUTPUT)    
#
#Args(2): HEADC     | KHEADC    | NHEADC   | HEADU      | KHEADU  | NHEADU |
#DTYPE: REAL(KHEADC)| INT       | INT      |REAL(KHEADU)| INT    | INT
#In/Out:  OUTPUT | INPUT     | OUTPUT   | OUTPUT| INPUT   | OUTPUT
#
#args(3): DATA        | KDATA | NDATA | IPLAN  | LFOUND )
#DTYPE:   REAL(KDATA) | INT   |  INT  |  INT    | LOGICAL
#In/Out:  OUTPUT      | INPUT | OUTPUT| INPUT  | OUTPUT
    
    zreadx.argparse = [C.c_long*600, C.c_char*80, C.c_long, C.c_long, C.c_long, \
                       C.c_long, C.c_long, C.c_long, C.c_long, C.c_long, C.c_long,\
                       C.c_short, C.c_long, C.c_long, C.c_long, C.c_bool,\
                       C.c_long]  # length of cpath
    zreadx.restype = None
    
    ecpath = cpath.encode('ascii')
    l_ecpath = len(ecpath)
    
    # if number of data values (KDATA) not provided as argument, retrieve it
    # from a call to zdtype
    if kdata==0 or kdata=='' or kdata is None:
        dtyp = get_datatype(ifltab, cpath)
        kdata=dtyp[3]*2
    
    
#    nheadi = len(headi)
#    nheadc = 0 #len(headc)
#    nheadu = 0 #len(headu)
##    if sizeout==1:
##        ndata1 = 1
##    else:
##        ndata1 = int((sizeout+3)/4)*2 #max(round((sizeout+3)/4), int((sizeout+3)/4))
#    ndata2 = len(data)
#    
#
#    ndata = int((sizeout+3)/2) #ndata2
#    data = data[0:ndata]
#    if type(headi) is list:
#        headi = (C.c_long*nheadi)(*headi)
#    
#    if type(headc) is list:
#        headc = (C.c_long*nheadc)(*headc)
#    
#    if type(headu) is list:
#        headu = (C.c_long*nheadu)(*headu)
    
    # variable specs & types
    kihead = C.c_long(kheadi)  
    kchead = C.c_long(kheadc)
    kuhead = C.c_long(kheadu)
    headi = (C.c_long*kheadi)()   # internal header array containg kheadi (or fewer) elements
    nheadi = C.c_long()            # number of internal header array elements returned
    headc = (C.c_long*kheadc)()  # compression array containing kheadc (or fewer) elements
                                   # - looking at Debug logs in HEC-DSSVUe, it seems this value
                                   #   often is set at 0 internally, may benefit from some 
                                   # further testing and investigation to see if a different value
                                   # would be more appropriate
    nheadc = C.c_long()           # number of compression header array elements returned
    headu  = (C.c_float*kheadu)()  # user header array containing `kheadu` or fewer
                                   # elements; as with headc and DSSVue, needs more
                                   #  attention
    nheadu = C.c_long()           # number of user header array elements returned
    data = (C.c_short*kdata)()    # the array of data values
    kdatai = C.c_long(kdata)
    ndata = C.c_long()            # number of data values actually returned
    iplan = C.c_long(1)           # internal DSS var, set to 1 for consistency with HEC-DSSVue
    lfound = C.c_bool()           # logical - if FALSE - couldn't find record, nothing retrieved
    
    
    zreadx(ifltab, ecpath, C.byref(headi), C.byref(kihead),C.byref(nheadi), \
           C.byref(headc),C.byref(kchead), C.byref(nheadc), \
           C.byref(headu), C.byref(kuhead), C.byref(nheadu), \
           C.byref(data), C.byref(kdatai), C.byref(ndata), C.byref(iplan), C.byref(lfound), \
           l_ecpath)
    
    # for some reason the values are returned in reversed order by pairs
    # the following re-orders thse pairs
    for i in range(0, len(data), 2):
        data[i], data[i+1] = data[i+1], data[i] 
    
    results = [headi, nheadi.value, headc, nheadc.value,
               headu, nheadu.value,data, ndata.value, lfound.value]
    
    return(results)

def get_gridInfo(intHdr):

    gridInfo = {} # dictionary to hold grid information
    
    gridInfo['InfoFlatSize'] = intHdr[0]
    gridInfo['GridType'] = intHdr[1]
    gridInfo['InfoSize']= intHdr[2]
    gridInfo['GridTypeDesc'] = get_gridTypeDesc(intHdr[1])
    gridInfo['GridInfoSize'] = intHdr[3]
    if intHdr[3] ==124:
        gridInfo['Version']=1
    else:
        gridInfo['Version']=intHdr[3]  #TODO: this will be incorrect if it gets to here - add correct version look up condition
    gridInfo['StartTime'] = intHdr[4]
    gridInfo['EndTime'] = intHdr[5]
    gridInfo['dataUnits'] =  hol2char(intHdr[6:9],0,12,0)
#        [struct.unpack('4s',(intHdr[6]).to_bytes(4,byteorder='little'))[0].decode(),\
#         struct.unpack('4s',(intHdr[7]).to_bytes(4,byteorder='little'))[0].decode(),\
#         struct.unpack('2s',(intHdr[8]).to_bytes(2,byteorder='little'))[0].decode()]
    gridInfo['DataType'] = intHdr[9] #0 = period avg, 1=period cumulative, 2 = instantaneous value, 3=inst. cumulative, 4 = frequency, 5=invalid
    gridInfo['LLCellX'] = intHdr[10]
    gridInfo['LLCellY'] = intHdr[11]
    gridInfo['nCols'] = intHdr[12]
    gridInfo['nRows'] = intHdr[13]
    gridInfo['CellSize'] = struct.unpack('f',(intHdr[14]).to_bytes(4,byteorder='little'))[0]
    gridInfo['CompressionMethod']=intHdr[15]
    gridInfo['CompressedSize']=intHdr[16]
    gridInfo['CompScaleFactor'] = struct.unpack('f',(intHdr[17]).to_bytes(4,byteorder='little'))[0]
    gridInfo['CompBase'] = struct.unpack('f',(intHdr[18]).to_bytes(4,byteorder='little', signed=True))[0]
    gridInfo['MaxValue'] = struct.unpack('f',(intHdr[19]).to_bytes(4,byteorder='little',signed=True))[0]
    gridInfo['MinValue'] = struct.unpack('f',(intHdr[20]).to_bytes(4,byteorder='little', signed=True))[0]
    gridInfo['MeanValue'] = struct.unpack('f',(intHdr[21]).to_bytes(4,byteorder='little', signed=True))[0]
    gridInfo['NumRanges'] = intHdr[22]
    rangeLimit=[]
    cntr=23
    for n in range(20): #gridInfo['NumRanges']):
        rangeLimit.append(struct.unpack('f',(intHdr[cntr]).to_bytes(4,byteorder='little',signed=True))[0])
        cntr+=1
    numEqExceedRng =[]
    for n in range(20): #gridInfo['NumRanges']):
        numEqExceedRng.append(intHdr[cntr])
        cntr+=1
    gridInfo['RangeHist'] = [rangeLimit, numEqExceedRng]
    # extend the dictionary for spatial reference info by grid type
    if gridInfo['GridType']==420:
        # Albers SHG grid type
        (gridInfo['SpatialRef'], cntr) = get_AlbersInfo(intHdr,cntr)
    
    return(gridInfo)

def get_AlbersInfo(intHdr,idx):
    tmpDict = {}
    tmpDict['ProjDatum'] = intHdr[idx]
    idx+=1
# TODO : the ProjectionUnits entry appears to be a 3-word component (like data
#        units in GridInfo), but it only seems to work with sizes 4s, 2s, 2s -
#        consider updating with call to HecLib 'StringHol' to get precise conversion )
    tmpDict['ProjUnits'] = hol2char(intHdr[idx:idx+3],0,12,0) # \
#        [struct.unpack('4s',(intHdr[idx]).to_bytes(4,byteorder='little'))[0].decode(),\
#         struct.unpack('2s',(intHdr[idx+1]).to_bytes(2,byteorder='little'))[0].decode(),\
#         struct.unpack('2s',(intHdr[idx+2]).to_bytes(2,byteorder='little'))[0].decode()]
    idx+=3
    tmpDict['FirstStdParallel'] = struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    tmpDict['SecondStdParallel']=struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    tmpDict['CentralMeridian']=struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    tmpDict['LatitudeProjOrigin']=struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    tmpDict['FalseEasting']=struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    tmpDict['FalseNorthing']=struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    tmpDict['XCoordLowerLeft']=struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    tmpDict['YCoordLowerLeft']=struct.unpack('f',(intHdr[idx]).to_bytes(4,byteorder='little',signed=True))[0]
    idx+=1
    return([tmpDict, idx])
    
def get_gridTypeDesc(gridType):
    gridTypes = {
            400:'Unknown grid type',
            410:'HRAP grid type',
            420:'Albers grid type',
            430:'Specified grid type',
            411:'HRAP non-time-varying',
            421:'Albers non-time-varying',
            431:'Custom non-time-varying'
            }
    try:
        gridTypeDesc = gridTypes[gridType]
    except:
        gridTypeDesc = 'Unspecified/unknown grid type'
        
    return(gridTypeDesc)
    
#@jit  #(nopython=True)
def grid_decomp(data, dsize, scaleFactor, scaleBase, nrows, ncols, returnArray=True, debug=False):
    numZeros = 0
    numOnes = 0
    numValues = 0
    cntr = 0
    #newData = [None]*(nrows*ncols) #initialize a list to hold uncompressed data
    newData = np.empty(nrows*ncols, dtype=np.float)
    newIdx = 0
    i = 0
    intsize = int(dsize/2)
    # check the compressed size against the number of entries to read -
    #  if these aren't equal, something isn't correct and we shouldn't try
    #  to decompress the array
#    if intsize != int(len(data)-1):
#        print("Error - gridInfo['CompressedSize'] doesn't match size ofdata being provided")
#        return
    #try:
    #for i in range(len(data)): #range(intsize):
    while i < intsize:
        #print(i, intsize, cntr, newIdx)
        if (data[i] & 0x8000) == 0:   # bitwise comparison -> is this a real value?
            #print(newIdx)
            newData[newIdx] = data[i]
            numValues+=1
            newIdx +=1
            if debug:
                print("Value at base index %s, new index %s" %(i, newIdx-1))
        elif (data[i] & 0xC000) == 49152:  #bitwise comparision -> is this a repeating 1?
            numRepeats = (data[i] & 0x3FFF)
            numOnes += numRepeats
            for j in range(newIdx, newIdx+numRepeats):
                newData[j] = -1.0
            newIdx+=numRepeats
            if debug:
                print("%s repeating ones in base index %s start at new index %s, end at new index %s" %(numRepeats, i, newIdx-numRepeats, newIdx))
                print("Number of zeros so far: %s" %numZeros)
                print("Number of ones so far: %s" %numOnes)
                #if numRepeats>3:
                #    break

        elif (data[i] & 0xC000) == 32768:  #bitwise comparison -> is this a repeating 0?
            numRepeats= (data[i] & 0x3FFF)
            numZeros += numRepeats
            for j in range(newIdx, newIdx+numRepeats):
                newData[j] = 0.0
            newIdx+=numRepeats
            if debug:
                print("%s repeating zeroes in base index %s start at new index %s, end at new index %s" %(numRepeats, i, newIdx-numRepeats, newIdx))
                print("Number of zeros so far: %s" %numZeros)
                print("Number of ones so far: %s" %numOnes)
                #if numRepeats>1:
                #    break

        else:
            print("Unknown value encountered....hmmmm...")
            #return([-99999])
        i+=1
        #cntr+=1
        
    for i in range(newIdx):
        if newData[i] < 0.0:   # the -1 value is a flag for missing/empty data (as opposed to zeroes)
            newData[i] = np.nan
        else:
            newData[i] = newData[i]/scaleFactor + scaleBase
    
    if returnArray:
        # shape the list into a numpy array
        #newArray = np.asarray(newData, dtype=np.float32).reshape(nrows, ncols, order='F')
        #newArray = newData.reshape((nrows, ncols), order='F')
        newArray = newData.reshape((ncols, nrows), order='F')
    else:
        newArray = newData
    return(newArray)
        
#    except:
#        print("seomthing didn't work...")
#        print("base index, new index, repeats, zeros", i, newIdx, numRepeats, numZeros)
        
    
#@jit #(nopython=True)
def grid_comp(data2d, scaleFactor, scaleBase, nrows, ncols, flatten_order='C', NA_Value=-99999, debug=False):
    data2d[np.isnan(data2d)]=NA_Value
    data = data2d.flatten(order=flatten_order)
    dsize = len(data)
    if debug:
        print("Size found = %s" %dsize)
    numZeros = 0
    numOnes = 0
    numValues = 0
    cntr = 0
    newcntr = 0
    
    sizeXY = nrows*ncols
    
    shortval = [None]*dsize 
    
    if sizeXY % 2 == 0:
        arrayout = [0]*sizeXY
    else:
        arrayout = [0]*(sizeXY+1)

    # do an initial sweep to check for negative numbers and numbers with greater
    # preciision than can be accomodated under the assumptions of the format
    # in both cases, the values will be replaced with a -1 and subsequently (belwo)
    # converted to missing flags
    while cntr<dsize:
        if data[cntr] == NA_Value: #<0.0:   # no negative values allowed - assumed to be NaN/missing data
            shortval[cntr] = -1
        else:
            lngval = int((data[cntr]-scaleBase) * scaleFactor + 1e-4)
            if lngval >32768:
                print("Scaled/shifted value exceeds precision - value will be lost")
                shortval[cntr] = -1  # if value is beyond available precision (e.g. value can't be >327.68 for scaleFactor of 100)
            else:
                shortval[cntr]= lngval
        cntr+=1
    
    if debug:
        print("shortval list is %s long" %len(shortval))
    cntr=0
    #print("read all values once - going to compress...or something...")
    while cntr<dsize:
        #if cntr%1000==0:
            #print("compressed %s values of dsize" %cntr)
        if shortval[cntr]==0.:
            repeatvals = 0
            while (cntr < dsize) and (shortval[cntr]==0):
                 #print("found some zeros to possibly lump")
                #if cntr == (dsize-1):
                #print("Lumping zeros %s out of %s" %(cntr, dsize))
                repeatvals+=1
                cntr+=1
            if repeatvals <16000:
                numZeros +=repeatvals
                #print("setting number of zeros to: %s" %numZeros)
                #arrayout.append(repeatvals | 0x8000)
                arrayout[newcntr] = repeatvals | 0x8000
                newcntr+=1
            else:
                print("Wow...more than 16000 repeat values!")
                while repeatvals >=16000:
                    numZeros += 16000
                    #arrayout.append(48768)
                    arrayout[newcntr] = 48768
                    newcntr+=1
                    repeatvals -= 16000
                numZeros += repeatvals
                #arrayout.append(repeatvals | 0x8000)
                arrayout[newcntr] = repeatvals | 0x8000
                newcntr+=1
        elif shortval[cntr]==-1:
            repeatvals = 0
            while (cntr < dsize) and (shortval[cntr]==-1):
                repeatvals+=1
                cntr+=1
            if repeatvals <16000:
                numOnes +=repeatvals
                #arrayout.append(repeatvals | 0xC000)
                arrayout[newcntr] = repeatvals | 0xC000
                newcntr+=1
            else:
                while repeatvals >=16000:
                    numOnes += 16000
                    arrayout.append(65152)
                    newcntr+=1
                    repeatvals -= 16000
                numOnes += repeatvals
                #arrayout.append(repeatvals | 0xC000)
                arrayout[newcntr] = repeatvals | 0xC000
                newcntr+=1
        else:
            #arrayout.append(shortval[cntr])
            arrayout[newcntr] = shortval[cntr]
            cntr+=1
            numValues +=1
            newcntr+=1
            
    #if newcntr != len(arrayout):
    #    print("Warning: compressed array size (%s) not the same as counted length...better check this out." %newcntr)
   #     print("-------:   Setting size to be that of the compressed array: %s" %len(arrayout))        
        #newcntr = len(arrayout)
    sizeout = newcntr*2
    
    totalsize = numZeros+numOnes+numValues
    
    if debug:
        print("Num zeros, NumOnes, NumVals, Total, Sizeout: %s, %s, %s, %s, %s" %(numZeros, numOnes, numValues, totalsize, sizeout))
    
    return([arrayout, sizeout, totalsize])
                
            
def write_compressed(ifltab, cpath, headi, headc,headu, data, sizeout, itype, iplan=0):
    '''
        wrapper around the zwritx function - to read datasets that don't conform
        to the 'standard DSS converntions'; assumes input data is a compressed
        (short integer) array
        
        ifltab - table identifier for open dss file
        cpath - DSS pathname
        npath- lenght of cpath
        headi = flattened header array (returned from 'set_gridInfo')
        nheadi = length of headi
        headc = "data compression array" (assume it's an empty variable)
        nheadc = length of headc = 0
        headu = user header array (usually ZSTFH is called to prepare this, assume empty variable)
        nheadu = length of headu = 0
        data = the (compressed) data array
        sizeout = the integer size of compressed array returned from grid_comp()
        ndata = length of data
        itype = data type (for grids, this is the GridType (e.g. 420 for Albers))
        iplan = write over existing data or not (0 = always, 1 only write if record is new, 2 only write if the record already exists)
        istat (OUTPUT) status parameter indicteing success or not
        lfound (OUTPUT) logical status variable indicating if the record laready existed
        
    '''
    if excelversion:
        zwritx = getattr(dsslib, 'ZWRITX_')
    if mingwversion:
        zwritx = getattr(dsslib, 'zwritx_')
#     Argument     Dtype          Input/Output
# --------------------------------------------
#   IFLTAB(600)   INTEGER(600)     INPUT
#   CPATH*80      CHAR*80          INPUT
#   NPATH         INT              INPUT
#   HEADI        REAL(NHEADI)      INPUT
#   NHEADI        INT              INPUT
#   HEADC        REAL(NHEADC)      INPUT  possibly INT?
#   NHEADC        INT              INPUT
#   HEADU        REAL(NHEADU)      INPUT
#   NHEADU       INT               INPUT
#   DATA         REAL(NDATA)        INPUT
#   NDATA        INT               INPUT
#   ITYPE        INT               INPUT
#   IPLAN        INT               INPUT
#   ISTAT        INT               OUTPUT
#   LFOUND       LOGICAL           OUTPUT
#
#    if len(headi)%4 !=0:
#        print("integer header not the right length...exiting\n")
#        exit
    nheadi = len(headi)
    nheadc = 0 #len(headc)
    nheadu = 0 #len(headu)
#    if sizeout==1:
#        ndata1 = 1
#    else:
#        ndata1 = int((sizeout+3)/4)*2 #max(round((sizeout+3)/4), int((sizeout+3)/4))
    ndata2 = len(data)
    

    ndata = int((sizeout+3)/2) #ndata2
    data = data[0:ndata]
    if type(headi) is list:
        headi = (C.c_long*nheadi)(*headi)
    
    if type(headc) is list:
        headc = (C.c_long*nheadc)(*headc)
    
    if type(headu) is list:
        headu = (C.c_long*nheadu)(*headu)
    
    # do we need to do the ordered-pair flip here?
    # for some reason the values are returned in reversed order by pairs
    # the following re-orders thse pairs
#    if len(data) % 2 != 0:
#        print("expanding dataset by one")
#        datalen = len(data)+1
#        data.append(0)
#        ndata+=1
#    else:
#        datalen = len(data)
 

#    if ndata1 != ndata:
#        print("***Compressed data size not consistent (%s, %s) ---please check!***\n" %(ndata1, ndata))
#        return([None, None, None])
#    else:
#        ndata = ndata1

       
    for i in range(0, len(data)-1, 2):
       data[i], data[i+1] = data[i+1], data[i] 
    
    ifltab_orig = ifltab
    
    zwritx.argparse = [C.c_long*600,              #IFLTAB
                       C.c_char*80,               #CPATH
                       C.c_long,                  #NPATH
                       C.c_long*nheadi,      #HEADI
                       C.c_long,                  #NHEADI
                       C.c_long*nheadc, #)(),      #HEADC
                       C.c_long,                  #NHEADC
                       C.c_long*nheadu, #)(),       #HEADU
                       C.c_long,                  #NHEADU
                       C.c_short*ndata, #)(),         # DATA
                       C.c_long,                  #NDATA
                       C.c_long,                   #ITYPE
                       C.c_long,                   #IPLAN
                       C.c_long,                   #ISTAT
                       C.c_bool,                   #LFOUND
                       C.c_long]                   #length of CPATH

    zwritx.restype = None
    
    
    ecpath = cpath.upper().encode('ascii')
    l_ecpath = len(ecpath)
    

    # variable specs & types
    cpath_ = ecpath
    npath_ = C.c_long(l_ecpath)
    nheadi_ = C.c_long(nheadi)  
    nheadc_ = C.c_long(nheadc)
    nheadu_ = C.c_long(nheadu)
    ndata_ = C.c_long(ndata)
    ifltab_ = (C.c_long*600)(*ifltab)

    data_ = (C.c_short*ndata)(*data)    # the array of data values

    iplan_ = C.c_long(iplan)           # internal DSS var, set to 1 for consistency with HEC-DSSVue
    itype_ = C.c_long(itype)         #integer data type identifier (420 = albers grid)
    istat_ = C.c_long(-99)
    lfound_ = C.c_bool()           # logical - if FALSE - couldn't find record, nothing retrieved

       
    #print("IFLTAB: %s\n" %ifltab[0:4])
    try:
        zwritx(C.byref(ifltab),
               cpath_, 
               C.byref(npath_),
               C.byref(headi), 
               C.byref(nheadi_),
               C.byref(headc),
               C.byref(nheadc_), 
               C.byref(headu),
               C.byref(nheadu_),
               C.byref(data_),
               C.byref(ndata_),
               C.byref(itype_),
               C.byref(iplan_),
               C.byref(istat_),
               C.byref(lfound_),
               l_ecpath)
    except:
        print("Something went wrong with call to `zwritx_`")
    
    if ifltab != ifltab_orig:
        print("ifltab changed")
    
    istat_val = istat_.value
    if istat_val==0:
        pass #print("Data stored successfully\n")
    elif istat_val==-1:
        print("Error: The record does note exist and IPLAN\n"+\
              "was set to two (write over existing records only")
    elif istat_val==-2:
        print("Error: The record already exists and IPLAN\n"+\
              "was set to one (do not write over existing records")
    elif istat_val==-10:
        print("Error: An invalid pathname was given")
    elif istat_val==-11:
        print("Error: An invalid number of data values was given")
    elif istat_val==-12:
        print("Error: The DSS file has read access only")
    else:
        print("Error: An unknown error hath occurred forthwith")
        
    lfound = lfound_
    results = [istat_val, lfound, ifltab]
    
    return(results)  
    
    
    
    
def write_regtsd(ifltab, cpath, cdate, ctime, vals, cunits, ctype,
                 coords=[0.0, 0.0, 0.0], icdesc=[0, 0, 0, 0, 0, 0],
                 csupp='', ctzone='', iplan=0):
    """
    Summary
    -------
    Python function to extract regular time series and supplemental information
    from DSS file. This function was written with guidance from HECLIB
    documentation of 'ZSRTSX' [1]_ and source code in DSSExcel.xlam [2]_.

    Parameters
    ----------
    ifltab : c_long_Array_600
        Integer long returned when opening a DSS file with open_dss().
    cpath : str
        Pathname of regular time series to be stored.
    cdate : str
        Starting date of regular time series in "DDMMMYYYY" format. If the year
        is in the 1900s, date can be in "DDMMMYY" format.
    ctime : str
        Time since midnight of starting date.
    vals : list of float
        Sequential list of regular time series.
    cunits : str
        Units of time series data.
    ctype : str
        Type of time series data (e.g. 'PER-AVER', 'PER-CUM', 'INST-VAL')
    coords : list of float, default [0.0, 0.0, 0.0], optional
        Coordinates of time series data.

        - coords[0] : X-coordinate
        - coords[1] : Y-coordinate
        - coords[2] : Unknown; default set to 0.

    icdesc : list of int, default [0, 0, 0, 0, 0, 0], optional
        Metadata for coordinates of time series data.

        +------------+-----------------------------------------------------+
        | List Entry | Description                                         |
        +============+=====================================================+
        | icdesc[0]  | One of the following options for Coordinate System. |
        |            |                                                     |
        |            | - 0 = No Coordinates Set                            |
        |            | - 1 = Latitude/Longitude                            |
        |            | - 2 = State Plane/FIPS                              |
        |            | - 3 = State Plane/ADS                               |
        |            | - 4 = UTM                                           |
        |            | - 5 = Local/other                                   |
        +------------+-----------------------------------------------------+
        | icdesc[1]  | Coordinate ID Number.                               |
        +------------+-----------------------------------------------------+
        | icdesc[2]  | Horizontal Datum Units.                             |
        |            |                                                     |
        |            | - 0 = Not Specified                                 |
        |            | - 1 = English (ft or miles i guess?)                |
        |            | - 2 = SI   (m or km?)                               |
        |            | - 3 = Decimal degrees                               |
        |            | - 4 = degrees, minutes, seconds                     |
        +------------+-----------------------------------------------------+
        | icdesc[3]  | Horizontal Datum.                                   |
        |            |                                                     |
        |            | - 0 = Unset                                         |
        |            | - 1 = NAD83                                         |
        |            | - 2 = NAD27                                         |
        |            | - 3 = WAS84                                         |
        |            | - 4 = WAS72                                         |
        |            | - 5 = local/other                                   |
        +------------+-----------------------------------------------------+
        | icdesc[4]  | Unknown; default set to 0                           |
        +------------+-----------------------------------------------------+
        | icdesc[5]  | Unknown; default set to 0.                          |
        +------------+-----------------------------------------------------+

    csupp : str, default '', optional
        Description of time series data.
    ctzone : str, default '', optional
        Time Zone Identification.
    iplan : int, default 0, optional
        Argument for writing over existing data according to the following
        table from HECLIB documentation of 'ZSRTSX' [1]_.

        +-------+-------------------------------------------------------+
        | iplan | Description                                           |
        +=======+=======================================================+
        | 0     | Always write over existing data.                      |
        +-------+-------------------------------------------------------+
        | 1     | Only replace missing data flags in the record (-901). |
        +-------+-------------------------------------------------------+
        | 4     | If an input value is missing (-901), do not allow it  |
        |       | to replace a non-missing value.                       |
        +-------+-------------------------------------------------------+

    Returns
    -------
    istat : None
        Status indicator of writing operation with the following possible
        returns as stated in the HECLIB documentation of 'ZSRTSX' [1]_.

        +-------+---------------------------------------------------------+
        | istat | Description                                             |
        +=======+=========================================================+
        | 0     | The data was successfully stored.                       |
        +-------+---------------------------------------------------------+
        | 4     | All of the input data provided were missing data flags  |
        |       | (-901).                                                 |
        +-------+---------------------------------------------------------+
        | >10   | A "fatal" error occurred."                              |
        +-------+---------------------------------------------------------+
        | 11    | The number of values to store (nvals) is less than one. |
        +-------+---------------------------------------------------------+
        | 12    | Unrecognized time interval (E part).                    |
        +-------+---------------------------------------------------------+
        | 15    | The starting date or time is invalid.                   |
        +-------+---------------------------------------------------------+
        | 24    | The pathname given does not meet the regular-interval   |
        |       | time series conventions.                                |
        +-------+---------------------------------------------------------+
        | 51    | Unrecognized data compression scheme.                   |
        +-------+---------------------------------------------------------+
        | 53    | Invalid precision exponent specified for the delta      |
        |       | compression method. The precision exponent range is     |
        |       | -6 to +6.                                               |
        +-------+---------------------------------------------------------+

    Notes
    -----
    Variables without the ``_input`` suffix are used directly into the
    heclib_x64.dll ``'ZSRTSC_'`` function. Variables with ``_input`` suffix are
    transformed before input into the heclib_x64.dll ``ZSRTSC_`` function.

    Set missing values to -901.0.

    The following table summarizes arguments for ``'ZSRTSC_'`` based on
    information from HECLIB documentation of 'ZSRTSX' [1_] and source code in
    DSSExcel.xlam [2]_.

    +----------+----------+--------------+--------------+---------------------+
    | Sequence | Argument | Data Type    | Input/Output | Description         |
    +==========+==========+==============+==============+=====================+
    | 1        | IFLTAB   | INTEGER(600) | INPUT        | The DSS work space  |
    |          |          |              |              | used to manage the  |
    |          |          |              |              | DSS file.           |
    +----------+----------+--------------+--------------+---------------------+
    | 2        | CPATH    | CHARACTER*80 | INPUT        | The pathname of the |
    |          |          |              |              | data to store.      |
    +----------+----------+--------------+--------------+---------------------+
    | 3        | CDATE    | CHARACTER*20 | INPUT        | The beginning date  |
    |          |          |              |              | of the time window. |
    +----------+----------+--------------+--------------+---------------------+
    | 4        | CTIME    | CHARACTER*4  | INPUT        | The beginning time  |
    |          |          |              |              | of the time window. |
    +----------+----------+--------------+--------------+---------------------+
    | 5        | NVALS    | INTEGER      | INPUT        | The number of values|
    |          |          |              |              | to store for        |
    |          |          |              |              | SVALUES, defining   |
    |          |          |              |              | the end of the time |
    |          |          |              |              | window.             |
    +----------+----------+--------------+--------------+---------------------+
    | 6        | DOUBLE   | INTEGER      | INPUT        | The number of values|
    |          |          |              |              | to store for        |
    |          |          |              |              | DVALUES, defining   |
    |          |          |              |              | the end of the time |
    |          |          |              |              | window.             |
    +----------+----------+--------------+--------------+---------------------+
    | 7        | SVALUES  | REAL(NVALS)  | INPUT        | List of time series |
    |          |          |              |              | values in sequential|
    |          |          |              |              | order to be stored  |
    |          |          |              |              | in single precision.|
    +----------+----------+--------------+--------------+---------------------+
    | 8        | DVALUES  | REAL(DOUBLE) | INPUT        | List of time series |
    |          |          |              |              | values in sequential|
    |          |          |              |              | order to be stored  |
    |          |          |              |              | in double precision.|
    +----------+----------+--------------+--------------+---------------------+
    | 9        | JQUAL    | INTEGER      | INPUT        | Unknown, but likely |
    |          |          |              |              | an array containing |
    |          |          |              |              | thirty-two bit      |
    |          |          |              |              | flags. Not stored if|
    |          |          |              |              | LQUAL is false.     |
    +----------+----------+--------------+--------------+---------------------+
    | 10       | LQUAL    | LOGICAL      | INPUT        | Variable indicating |
    |          |          |              |              | whether or not to   |
    |          |          |              |              | store JQUAL.        |
    +----------+----------+--------------+--------------+---------------------+
    | 11       | CUNITS   | CHARACTER*8  | INPUT        | The units of the    |
    |          |          |              |              | data (e.g., 'FEET').|
    +----------+----------+--------------+--------------+---------------------+
    | 12       | CTYPE    | CHARACTER*8  | INPUT        | The type of the data|
    |          |          |              |              | (e.g., 'PER-AVER'). |
    +----------+----------+--------------+--------------+---------------------+
    | 13       | COORDS   | REAL         | INPUT        | Coordinates of the  |
    |          |          |              |              | time series data.   |
    +----------+----------+--------------+--------------+---------------------+
    | 14       | NCOORDS  | INTEGER      | INPUT        | Length of COORDS.   |
    +----------+----------+--------------+--------------+---------------------+
    | 15       | ICDESC   | INTEGER      | INPUT        | Metadata for COORDS.|
    +----------+----------+--------------+--------------+---------------------+
    | 16       | NCDESC   | INTEGER      | INPUT        | Length of ICDESC.   |
    +----------+----------+--------------+--------------+---------------------+
    | 17       | CSUPP    | CHARACTER*80 | INPUT        | Description of the  |
    |          |          |              |              | time series data.   |
    +----------+----------+--------------+--------------+---------------------+
    | 18       | ITZONE   | INTEGER      | INPUT        | Time offset in      |
    |          |          |              |              | minutes from UTC.   |
    +----------+----------+--------------+--------------+---------------------+
    | 19       | CTZONE   | CHARACTER*30 | INPUT        | Time Zone ID.       |
    +----------+----------+--------------+--------------+---------------------+
    | 20       | IPLAN    | INTEGER      | INPUT        | Variable indicating |
    |          |          |              |              | whether or not to   |
    |          |          |              |              | write over existing |
    |          |          |              |              | data.               |
    +----------+----------+--------------+--------------+---------------------+
    | 21       | JCOMP    | INTEGER      | INPUT        | Data compression    |
    |          |          |              |              | method.             |
    +----------+----------+--------------+--------------+---------------------+
    | 22       | BASEV    | REAL         | INPUT        | Baseline value for  |
    |          |          |              |              | data compression.   |
    +----------+----------+--------------+--------------+---------------------+
    | 23       | LBASEV   | LOGICAL      | INPUT        | Variable indicating |
    |          |          |              |              | whether or not to   |
    |          |          |              |              | store BASEV.        |
    +----------+----------+--------------+--------------+---------------------+
    | 24       | LDHIGH   | LOGICAL      | INPUT        | Setting for         |
    |          |          |              |              | preallocating       |
    |          |          |              |              | for data            |
    |          |          |              |              | compression.        |
    +----------+----------+--------------+--------------+---------------------+
    | 25       | NPREC    | INTEGER      | INPUT        | Precision exponent  |
    |          |          |              |              | for data            |
    |          |          |              |              | compression.        |
    +----------+----------+--------------+--------------+---------------------+
    | 26       | ISTAT    | INTEGER      | OUTPUT       | Status parameter    |
    |          |          |              |              | indicating success  |
    |          |          |              |              | of storage.         |
    +----------+----------+--------------+--------------+---------------------+
    | 27       | L_Cpath  | INTEGER      | INPUT        | Length of CPATH.    |
    +----------+----------+--------------+--------------+---------------------+
    | 28       | L_CDate  | INTEGER      | INPUT        | Length of CDATE.    |
    +----------+----------+--------------+--------------+---------------------+
    | 29       | L_CTime  | INTEGER      | INPUT        | Length of CTIME.    |
    +----------+----------+--------------+--------------+---------------------+
    | 30       | L_CUnits | INTEGER      | INPUT        | Length of CUNITS.   |
    +----------+----------+--------------+--------------+---------------------+
    | 31       | L_CType  | INTEGER      | INPUT        | Length of CTYPE.    |
    +----------+----------+--------------+--------------+---------------------+
    | 32       | L_CSUPP  | INTEGER      | INPUT        | Length of CSUPP.    |
    +----------+----------+--------------+--------------+---------------------+
    | 33       | L_CTZONE | INTEGER      | INPUT        | Length of CTZONE.   |
    +----------+----------+--------------+--------------+---------------------+

    Stored data sets are without flags. For more information on flagging, see
    HECLIB documentation, Appendix C [1]_.

    Stored data is not compressed. For more information on data compression,
    see HECLIB documentation, Chapter 10 [1]_.

    References
    ----------

    The references below are formatted according to Chicago Manual of Style,
    16th Edition.

    .. [1] CEWRC-IWR-HEC. CPD-57.pdf. PDF. Davis, CA: US Army Corps of
       Engineers Institute for Water Resources Hydrologic Engineering Center,
       May 1991.
       Title: "HECLIB Volume 2: HECDSS Subroutines, Programmer's Manual"
       URL: http://www.hec.usace.army.mil/publications/ComputerProgramDocumentation/CPD-57.pdf
       Accessed: 2018-10-04

    .. [2] Steissberg, Todd. DSSExcel.xlam. Microsoft Excel XLAM. Davis, CA: US
       Army Corps of Engineers Institute for Water Resources Hydrologic
       Engineering Center, February 11, 2016. Title: "HEC-DSS MS Excel Data
       Exchange"

    """
    # Get data length from 'vals' list.
    nvals = len(vals)
    # Get DLL function for Storing Regular-Interval Time Series Data
    # (Extended Version).
    if excelversion:
        zsrtsc = getattr(dsslib, 'ZSRTSC_')
    if mingwversion:
        zsrtsc = getattr(dsslib, 'zsrtsc_')
    # Initialize input declarations for function.
    # ???: Is initialization required?
    # <JAS 2018-10-02>
    zsrtsc.argparse = []
    # Set input declarations for function, mapped to DLL function variables.
    # ???: Why is CPATH multiplied by 800 instead of 80?
    # <JAS 2018-10-03>
    zsrtsc.argparse = [C.c_long*600,  # IFLTAB
                       C.c_char*800,  # CPATH
                       C.c_char*20,   # CDATE
                       C.c_char*4,    # CTIME
                       C.c_long,      # NVALS
                       C.c_long,      # DOUBLE
                       (C.c_float*nvals)(),   # SVALUES
                       (C.c_double*nvals)(),  # DVALUES
                       C.c_long,      # JQUAL
                       C.c_bool,      # LQUAL
                       C.c_char*8,    # CUNITS
                       C.c_char*8,    # CTYPE
                       C.c_double,    # COORDS
                       C.c_long,      # NCOORDS
                       C.c_long,      # ICDESC
                       C.c_long,      # NCDESC
                       C.c_char*80,   # CSUPP
                       C.c_long,      # ITZONE
                       C.c_char*30,   # CTZONE
                       C.c_long,      # IPLAN
                       C.c_long,      # JCOMP
                       C.c_float,     # BASEV
                       C.c_bool,      # LBASEV
                       C.c_bool,      # LDHIGH
                       C.c_long,      # NPREC
                       C.c_long,      # ISTAT
                       C.c_long,      # L_CPATH  25
                       C.c_long,      # L_CDATE  26
                       C.c_long,      # L_CTIME  27
                       C.c_long,      # L_CUNITS 28
                       C.c_long,      # L_CTYPE  29
                       C.c_long,      # L_CSUPP  30
                       C.c_long]      # L_CTZONE 31
    # Set return type of function to NoneType.
    zsrtsc.restype = None
    # Indicate that there are no flags in the data set.
    jqual = [0 for i in range(nvals)]
    lqual = False
    # Set coordinates.
    ncoords = len(coords)
    # Set length of Coordinate System Info.
    ncdesc = len(icdesc)
    # Set Time Zone Offset.
    # NOTE: Offset seems to be in minutes whereas milliseconds in HEC-DSS Vue.
    # TODO: Set itzone given ctzone.
    # <JAS 2018-10-05>
    if ctzone == 'PST':
        itzone = -420
    else:
        itzone = 0
    # Set parameters for no data compression.
    jcomp = 0
    basev = 0
    lbasev = False
    ldhigh = False
    nprec = 0
    # Prepare function inputs. Distiguish function inputs with "_" suffix.
    ifltab_ = ifltab
    cpath_ = cpath.encode('ascii')
    cdate_ = cdate.encode('ascii')
    ctime_ = ctime.encode('ascii')
    nvals_ = C.c_long(nvals)
    double_ = C.c_long(nvals)
    svalues_ = (C.c_float*nvals)(*vals)
    dvalues_ = (C.c_double*nvals)(*vals)
    jqual_ = (C.c_long*nvals)(*jqual)
    lqual_ = C.c_bool(lqual)
    cunits_ = cunits.encode('ascii')
    ctype_ = ctype.encode('ascii')
    coords_ = (C.c_double*3)(*coords)
    ncoords_ = C.c_long(ncoords)
    icdesc_ = (C.c_long*6)(*icdesc)
    ncdesc_ = C.c_long(ncdesc)
    csupp_ = csupp.encode('ascii')  # FUNCTION INPUT
    itzone_ = C.c_long(itzone)
    ctzone_ = ctzone.encode('ascii')
    iplan_ = C.c_long(iplan)
    jcomp_ = C.c_long(jcomp)
    basev_ = C.c_float(basev)
    lbasev_ = C.c_bool(lbasev)
    ldhigh_ = C.c_bool(ldhigh)
    nprec_ = C.c_long(nprec)
    istat_ = C.c_long()
    l_cpath_ = len(cpath_)
    l_cdate_ = len(cdate_)
    l_ctime_ = len(ctime_)
    l_cunits_ = len(cunits_)
    l_ctype_ = len(ctype_)
    l_csupp_ = len(csupp_)
    l_ctzone_ = len(ctzone_)
    # Pass variables to DLL function.
    zsrtsc(C.byref(ifltab_),
           cpath_,
           cdate_,
           ctime_,
           C.byref(nvals_),
           C.byref(double_),
           C.byref(svalues_),
           C.byref(dvalues_),
           C.byref(jqual_),
           C.byref(lqual_),
           cunits_,
           ctype_,
           C.byref(coords_),
           C.byref(ncoords_),
           C.byref(icdesc_),
           C.byref(ncdesc_),
           csupp_,
           C.byref(itzone_),
           ctzone_,
           C.byref(iplan_),
           C.byref(jcomp_),
           C.byref(basev_),
           C.byref(lbasev_),
           C.byref(ldhigh_),
           C.byref(nprec_),
           C.byref(istat_),
           l_cpath_,
           l_cdate_,
           l_ctime_,
           l_cunits_,
           l_ctype_,
           l_csupp_,
           l_ctzone_)
    return istat_.value



def char2hol(cstr, ibeg, ilen, nbeg):
    '''
        converts a character string to Hollerith (integer arra) on byte boundaries
    '''
    if excelversion:
        chrhol = getattr(dsslib, 'CHRHOL_')
    if mingwversion:
        chrhol = getattr(dsslib, 'chrhol_')
    
    # CSTR - string | IBEG(int) - beginning position | ILEN(int) - length to convert |
    # IHOL - integer array out | NBEG (int) -beggining byte position in IHOL in which to place conveted chars
    
    chrhol.argparse = [C.c_char*10, C.c_long, C.c_long, C.c_long*10, C.c_long]
    chrhol.restype = None
    
    cstr = cstr.strip()
    ecstr = cstr.encode('ascii')
    ibeg = C.c_long(ibeg)
    ilen = C.c_long(ilen)
    ihol = (C.c_long*10)()
    nbeg = C.c_long(nbeg)
    
    l_cstr = len(ecstr)
    
    chrhol(ecstr, C.byref(ibeg), 
           C.byref(ilen), ihol, C.byref(nbeg), \
           l_cstr)
    
    #chrhol(ecstr, ibeg, ilen, C.byref(ihol), nbeg, l_cstr)
            
    return(ihol)

def hol2char(ihol, ibeg, ilen, nbeg):
    '''
        converts a Hollerith (integer array) to characters on byte boundaries
    '''
    if excelversion:
        holchr = getattr(dsslib, 'HOLCHR_')
    if mingwversion:
        holchr = getattr(dsslib, 'holchr_')
    
    # CSTR - string | IBEG(int) - beginning position | ILEN(int) - length to convert |
    # IHOL - integer array INPUT | NBEG (int) -beggining byte position in IHOL in which to place conveted chars
    
    ihol_len = len(ihol)
    holchr.argparse = [C.c_long*ihol_len, C.c_long, C.c_long, C.c_char*100, C.c_long]
    holchr.restype = None
    
    #ecstr = cstr.encode('ascii')
    cstr = (C.c_char*100)()
    ibeg_ = C.c_long(ibeg)
    ilen_ = C.c_long(ilen)
    if type(ihol) is list:
        ihol = (C.c_long*ihol_len)(*ihol)
    nbeg_ = C.c_long(nbeg)
    
    l_cstr = len(cstr)
    
    holchr(ihol, C.byref(ibeg_), 
           C.byref(ilen_), cstr, C.byref(nbeg_), \
           l_cstr)
    
    # remove all non-ascii characters
    cstr1 =b''.join([i if ord(i) <128 else b'\x00' for i in cstr])
    cstr2 = cstr1.decode('ascii').strip('\x00')#[0:3]
    cstrFin = cstr2.split('\\')[0]
#    if b'\x0b' in cstr[0:12]:
#        cstr1 = cstr[0:12].split(b'\x0b')[0]
#        cstrFin = cstr1.decode('ascii').strip('\x00')
##    cstr2 = b''.join(list(cstr)).strip('\x00')  #.decode('ascii')
#    else:
#        cstrFin = b''.join(cstr).decode().strip('\x00')
    #chrhol(ecstr, ibeg, ilen, C.byref(ihol), nbeg, l_cstr)
            
    return(cstrFin)    
    
def set_gridInfo(gridInfo):
    
    # converts a dictionary of grid header info to an encoded integer array for saving to DSS
    # it is assumed that the gridInfo dictionary has all key-value pairs filled in
    if gridInfo['InfoFlatSize']%4 != 0:
        print("Error: Header info not the correct size - grid cannot be written to DSS")
        return(-1)
    else:
        intHdrLen = int(gridInfo['InfoFlatSize']/4)
    
    intHdr = [None for n in range(intHdrLen)]  # a list to be filled with integers; will be converted to a ctypes c_long_Array after being populated
    
    intHdr[0] = gridInfo['InfoFlatSize']  # assume for Albers array that InfoFlatSize is 300 (252 general + 48 Albers info)
    intHdr[1] = gridInfo['GridType']
    intHdr[2] = gridInfo['InfoSize']
    intHdr[3] = gridInfo['GridInfoSize']

    if gridInfo['GridInfoSize'] ==124:
        thisVersion=1
    else:
        thisVersion=intHdr[3]  #TODO: this will be incorrect if it gets to here - add correct version look up condition
    intHdr[4] = gridInfo['StartTime'] 
    intHdr[5] = gridInfo['EndTime']
    charLen = max(12, 2*len(gridInfo['dataUnits']))
    intHdr[6:9] = char2hol(gridInfo['dataUnits'], 0, charLen, 0)[0:3]
    intHdr[9] = gridInfo['DataType']  #0 = period avg, 1=period cumulative, 2 = instantaneous value, 3=inst. cumulative, 4 = frequency, 5=invalid
    intHdr[10] = gridInfo['LLCellX']
    intHdr[11] = gridInfo['LLCellY'] 
    intHdr[12] = gridInfo['nCols']
    intHdr[13] = gridInfo['nRows']
    intHdr[14] = floatToIntBitsLE(gridInfo['CellSize'])
    intHdr[15] = gridInfo['CompressionMethod']
    intHdr[16] = gridInfo['CompressedSize']
    intHdr[17] = floatToIntBitsLE(gridInfo['CompScaleFactor'] )
    intHdr[18] = floatToIntBitsLE(gridInfo['CompBase']) 
    intHdr[19] = floatToIntBitsLE(gridInfo['MaxValue']) 
    intHdr[20] = floatToIntBitsLE(gridInfo['MinValue'])
    intHdr[21] = floatToIntBitsLE(gridInfo['MeanValue']) 
    intHdr[22] = gridInfo['NumRanges']

    cntr=23
    
    [rangeLimit, numEqExceedRng] = gridInfo['RangeHist']
    for n in rangeLimit: 
        intHdr[cntr] = floatToIntBitsLE(n)  # this is the histogram range/bin values
        cntr+=1

    for n in numEqExceedRng: 
        intHdr[cntr] = n  #this should be an integer (count)
        cntr+=1

    # extend the dictionary for spatial reference info by grid type
    if gridInfo['GridType']==420:
        # Albers SHG grid type
        albInfo = set_AlbersInfo(gridInfo)
        
    for a in albInfo:
        intHdr[cntr] = a
        cntr+=1
    
    return(intHdr)

def set_AlbersInfo(gridInfo):
    # takes dictionary of grid information, picks out the Albers spatial ref 
    # info and converts it to an integer array
    albInfo = [None for n in range(12)]
    spref = gridInfo['SpatialRef']
    albInfo[0] = spref['ProjDatum']
    albInfo[1:4] = char2hol(spref['ProjUnits'],0,12,0)[0:3]
    albInfo[4] = floatToIntBitsLE(spref['FirstStdParallel'])
    albInfo[5] = floatToIntBitsLE(spref['SecondStdParallel'])
    albInfo[6] = floatToIntBitsLE(spref['CentralMeridian'])
    albInfo[7] = floatToIntBitsLE(spref['LatitudeProjOrigin'])
    albInfo[8] = floatToIntBitsLE(spref['FalseEasting'])
    albInfo[9] = floatToIntBitsLE(spref['FalseNorthing'])
    albInfo[10] = floatToIntBitsLE(spref['XCoordLowerLeft'])
    albInfo[11] = floatToIntBitsLE(spref['YCoordLowerLeft'])
    return(albInfo)
    
def floatToIntBitsLE(f):
    # from https://stackoverflow.com/questions/14431170/get-the-bits-of-a-float-in-python
    s = struct.pack('<f', f)
    return(struct.unpack('i', s)[0])
    
def setAlbersProj():
    proj={}
    proj['CentralMeridian'] =  -96.0
    proj['FalseEasting'] = 0.0
    proj['FalseNorthing'] = 0.0
    proj['FirstStdParallel'] = 29.5
    proj['LatitudeProjOrigin'] = 23.0
    proj['ProjDatum'] = 2
    proj['ProjUnits'] = 'METERS'
    proj['SecondStdParallel'] = 45.5
    proj['XCoordLowerLeft'] = 0.0
    proj['YCoordLowerLeft'] = 0.0
    return(proj)


def copyRecord(ifltabOLD, ifltabNEW, pathOLD, pathNEW):
    if excelversion:
        zcorec = getattr(dsslib, 'ZCOREC_')
    if mingwversion:
        zcorec = getattr(dsslib, 'zcorec_')
    zcorec.restype = None
#     Argument     Dtype          Input/Output
# --------------------------------------------
#   IFTOLD(600)   INTEGER(600)     INPUT
#   IFTNEW(600)   INTEGER(600)     INPUT
#   CPOLD         CHAR*80          INPUT
#   CPNEW         CHAR*80 
#   CPATH*80      CHAR*80          INPUT
    KBUFF1=750
    KBUFF2=100
    zcorec.argparse = [
              C.c_long*600,  # old ifltab
              C.c_long*600,  # new ifltab
              C.c_char*800,  # old CPATH
              C.c_char*800,  # new CPATH
              (C.c_float*KBUFF1)(), #first buffer
              C.c_long,       #KBUFF1 - length of first buffer
              (C.c_float*KBUFF2)(),  #second buffer
              C.c_long,        #ISTAT            
              C.c_long,        #length of old path
              C.c_long         # lenght of new path
            ]
    
    istat = C.c_long()
    zcorec(C.byref(ifltabOLD),
          C.byref(ifltabNEW),
          pathOLD.encode('ascii'),
          pathNEW.encode('ascii'),
          (C.c_float*KBUFF1)(),
          C.byref(C.c_long(KBUFF1)),
          (C.c_float*KBUFF2)(),
          C.byref(C.c_long(KBUFF2)),
          C.byref(istat),
          len(pathOLD),
          len(pathNEW))
    
    if istat.value == 1:
        print("Error: The record to be copied (%s) does not exist." %pathOLD)
    elif istat.value == 2:
        print("Error: the new record (%s) already esists and write protection was set for this file" %pathNEW)
    elif istat.value == -1:
        print("Error: buffer lengths (KBUFF1 or KBUFF2) were set to zero. The dude will not abide")
    elif istat.value ==-2:
        print("Error: the bufferes supplied are too small for this record.\n" + \
              "The sizerequired will be printed if the message level is two or greater")
    elif istat.value==-12:
        print("Error: the file being copied to is in read access only mode")
    elif istat.value==0:
        #success!
        pass
    else:
        print("Error: this error (istat: %s) is uknown to us..." %(istat.value))
        
    return(istat.value)
    
    
def getTSrange(ifltab, cpath, searchWindow=10):
    
    if excelversion:
        print("Function for finding full length of recrods 'ztsends' not included in Excel DLL")
        return()
    if mingwversion:
        ztsends = getattr(dsslib, 'ztsends_')
    ztsends.restype = None
#     Argument     Dtype          Input/Output
# --------------------------------------------
#   IFLTAB(600)   INTEGER(600)     INPUT
#   CPATH*80      CHAR*80          INPUT
#   searchWindow  INTEGER          INPUT

    ztsends.argparse = [
              C.c_long*600,  # ifltab
              C.c_char*800,  #  CPATH
              C.c_long,  # search window lenght
              C.c_long, #start date
              C.c_long,  #start time in minutes
              C.c_long,  #end date
              C.c_long,  #end date, in minutes
              C.c_bool,    # lfound
              C.c_long    # length of path
            ]
    
    bdate = C.c_long()
    btime  = C.c_long()
    edate = C.c_long()
    etime = C.c_long()
    lfound = C.c_bool()
    #istat = C.c_long()
    ztsends(C.byref(ifltab),
          cpath.encode('ascii'),
          C.c_long(searchWindow),
          C.byref(bdate),
          C.byref(btime),
          C.byref(edate),
          C.byref(etime),
          C.byref(lfound),
          len(cpath))
    
    return([bdate, btime, edate, etime])