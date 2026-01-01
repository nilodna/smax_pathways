# scraps available URLs from NCEI NOAA website then download GHRSTT files [global], slicing regionally and merging it into a single monthly file
import warnings
warnings.filterwarnings("ignore")

import subprocess
import requests
from bs4 import BeautifulSoup
from urllib import request
import xarray as xr
import pandas as pd

# 0. general settings
savedir = '../data/sanagu2022/AVHRR/'
year = '2022'
idate = '2022-03-01'
edate = '2022-04-01'

bbox = {'lon': slice(-50, -30), 'lat': slice(-15, -30)}

# 1. scrapping URLs
url = f"https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/"

list_urls = []
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')

urls = []
for link in soup.find_all('a'):
    list_urls.append(link.get('href'))

print(f'{len(list_urls)} files available in {url}')

# 2. Download only a selected period
date_range = pd.date_range(idate, edate, freq='1D')

for date in date_range:
    print(f'Downloading {date}...')
    date = date.strftime('%Y%m%d')
    
    # find files with date in it
    fnames = [i for i in list_urls if date in i]
    fin = [f"{url}{i}" for i in fnames]
    fout = [f"{savedir}{i}" for i in fnames]
    
    # download
    for f,fsave in zip(fin, fout):
        response = request.urlretrieve(f, fsave)

    # 3. Slice and resample

    # read files in to slice and merge
    ds = xr.open_mfdataset(fout)

    ds_out = ds.sel(**bbox)
    #ds_out = ds_out.resample(time='1D').mean()
    #ds_out = ds_out.where(ds_out.pathfinder_quality_level > 2, drop=True)

    ds_out.to_netcdf(f"{savedir}/{date}d.nc")

    # cleaning downloaded files
    _ = [subprocess.call(f"rm {i}", shell=True) for i in fout]
