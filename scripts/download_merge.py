"""Script utilizado para baixar dados do merge cptec e recortá-los
na área definida pelo shapefile"""
import shutil
import urllib.request as request
from contextlib import closing
import pandas as pd
import datetime as dtt
import os
import os.path as osp
import xarray as xr

# -- paths --#
fsave = '../../project/data/treated'
ftpstr = 'http://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY'
tag = 'RdI'

# -- time period -- #
tperiod = pd.date_range(start='2022-01-01', end='2022-04-01', freq='1D')

bbox = {"longitude": slice(-50,-30), "latitude": slice(-30,-20)}

# --  configure ftp paths -- #
str_tmplt=f'{ftpstr}/%Y/%m/MERGE_CPTEC_%Y%m%d.grib2' # str template
merge_list = [dtt.datetime.strftime(i, str_tmplt) for i in tperiod] # ftp paths

for ffile in merge_list:
    print(ffile)
    fout = ffile.split('/')[-1]  # str: get the grib2 file name
    fgrib = osp.join(fsave, fout)  # str: saving path + grib2 filename
    fnc = osp.join(fsave, fout.split('_')[-1].split('.')[0] + 'd.nc')

    # -- ftp download -- #
    with closing(request.urlopen(ffile)) as r:
        with open(fgrib, 'wb') as f:
            shutil.copyfileobj(r, f)

    # xr.dataset: read grib2 with xarray and cfgrib engine
    nc = xr.open_dataset(fgrib, engine='cfgrib')
    nc['longitude'] = nc['longitude'] - 360
    nc = nc.sel(**bbox)
    
    nc.to_netcdf(fnc)
    print(f'{fnc} saved')

    os.system(f'rm {fsave}/*grib2 {fsave}/*idx')