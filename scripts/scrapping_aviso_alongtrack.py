import os
from ftplib import FTP
from tqdm import tqdm
import numpy as np

import dask
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=5) #number of cores
client = Client(cluster) # go check dask dashboard

# put your credentials here
credentials = dict(user='user', passwd='passwd')

def get_paths(base,satellite,years,months):
    years = np.array(years)
    months = np.array(months)
    ftp.cwd(os.path.join(base,satellite))
    files = []
    for year in ftp.nlst():
        if np.any(int(year)==years):
            ftp.cwd(os.path.join(base,satellite,year))
            for month in ftp.nlst():
                if np.any(int(month)==months):
                    ftp.cwd(os.path.join(base,satellite,year,month))
                    fnames = ftp.nlst()
                    for fname in fnames:
                        files.append(os.path.join(base,satellite,year,month,fname))
    return files

@dask.delayed
def download_file(remote_path,credentials=credentials):
    ftp = FTP('my.cmems-du.eu',**credentials)
    fname = os.path.basename(remote_path)
    local_path = os.path.join("data/",fname)
    if not os.path.exists(local_path):
        handle = open(local_path, 'wb')
        ftp.retrbinary('RETR %s' % remote_path, handle.write)
                

base = "/Core/SEALEVEL_GLO_PHY_L3_MY_008_062/"
ftp = FTP('my.cmems-du.eu',**credentials)
ftp.cwd(base)

satellites = ftp.nlst()
fnames = []
for satellite in tqdm(satellites):
    fnames.append(get_paths(base,satellite,[2022],[1,2,3,4]))
fnames = np.hstack(fnames)

delayed_objs = [download_file(fname) for fname in fnames]
       
results = dask.compute(*delayed_objs)