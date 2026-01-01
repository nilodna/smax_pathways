import subprocess
import pandas as pd
import numpy as np

baseurl = 'https://opendap.jpl.nasa.gov/opendap/OceanTemperature/modis/L3/aqua/4um/v2019.0/4km/daily'

# 2022/001/AQUA_MODIS.20220101.L3m.DAY.SST4.sst4.4km.NRT.nc

year = 2022
dates = pd.date_range('20220101', '20220331', freq='1D')
running_days = np.arange(1, len(dates)+1, 1)


for day,date in zip(running_days, dates):
    date = date.strftime('%Y%m%d')
    url = f"{baseurl}/{str(year)}/{str(day).zfill(3)}/AQUA_MODIS.{date}.L3m.DAY.SST4.sst4.4km.NRT.nc"
    
    cmd = f"wget {url} ."
    
    subprocess.call(cmd, shell=True)