# conda activate parcels
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4 , plotTrajectoriesFile, Field, ErrorCode

import xarray as xr
import numpy as np
import math
from datetime import timedelta,datetime


def DeleteParticle(particle, fieldset, time):
    particle.delete()

def set_aviso_field(fname):
    print("Building fieldset from AVISO geostrophic field")
    ds = xr.open_dataset(fname)
    
    # based on glorys
    dimensions = {'lon': 'longitude', 
                  'lat': 'latitude', 
                  'time': 'time'}
    variables = {'U': 'ugos', 
                 'V': 'vgos'}

    # instantiate fieldset
    fieldset = FieldSet.from_xarray_dataset(ds, 
                                            variables, 
                                            dimensions)
    
    return fieldset, ds

def set_cmems_field(fname, run3D=True):
    # if * exist in fname, then we are dealing with several fields
    if "*" in fname:
        print('mfdataset')
        ds = xr.open_mfdataset(fname)
    else:
        print('dataset')
        ds = xr.open_dataset(fname)
    
    dimensions = {'lon': 'longitude', 
                  'lat': 'latitude', 
                  'time': 'time'}
    
    if run3D:
        dimensions['depth'] = 'depth'
    
    variables = {'U': 'uo', 'V': 'vo'}
    
    fieldset = FieldSet.from_xarray_dataset(ds, 
                                            variables, 
                                            dimensions)
    
    return fieldset, ds
    
def get_particleSet(fieldset, center_launch, launch_depth, time, poolSize=None, poolN=None, ds=None):
        
    if poolSize and poolN:
        print('launch pool')
        lons = np.linspace(center_launch[0]-poolSize, center_launch[0]+poolSize, poolN)
        lats = np.linspace(center_launch[1]-poolSize, center_launch[1]+poolSize, poolN)
        x,y = np.meshgrid(lons,lats)
        lon_launch = x.ravel()
        lat_launch = y.ravel()
    else:
        print('single launch')
        lon_launch = center_launch[0]
        lat_launch = center_launch[1]
    
    if launch_depth != None:
        print('3D run')
        nz = (np.abs(ds.depth.values - launch_depth)).argmin()
        pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle,
                             lon=lons, 
                             lat=lats,
                             time=time,
                             depth=np.ones([poolN])*ds.depth.values[nz])
    else:
        print('2D run')
        pset = ParticleSet.from_list(fieldset=fieldset,
                                     pclass=JITParticle,
                                     lon=lon_launch, 
                                     lat=lat_launch,
                                     time=time)
        
    return pset
    
    
def set_running(outfile, fname, center_launch, time, launch_depth=None, run3D=False, 
                poolSize=None, poolN=None, aviso=True, cmems=False):
    """
        outfile: output filename
        fname: absolute path to hydrodynamic field
        center_launch (lon,lat): center coordinates of launch
        poolSize: set a float to create a launchin pool around center_launch
        poolN: number of particles
    """
    
    if aviso:
        fieldset, ds = set_aviso_field(fname)
    elif cmems:
        fieldset, ds = set_cmems_field(fname, run3D=run3D)
    else:
        raise ValueError('Set a hydrodynamic field aviso or cmems')
    
    pset = get_particleSet(fieldset, center_launch, launch_depth, time,
                           poolSize=poolSize, poolN=poolN, ds=ds)
        
    return fieldset, pset

def plot_track(fout, dict_casts):
    # load output
    p = xr.open_zarr(fout)

    # plotting with my functions
    zoomed_extent= [-50, -39, -30, -22]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(zoomed_extent, crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(
            category='physical',name='coastline',scale='10m',
            facecolor='lightgray')
    ax.add_feature(states_provinces, edgecolor='black',linewidth=.1)

    ax.scatter(p.lon, p.lat, s=15)
    
    for cast,values in dict_casts.items():
        prof = values['prof']
        str_time = values['time'].strftime("%m/%d")
        xlon = values['xlon']
        ylat = values['ylat']

        ax.scatter(xlon, ylat, c='k', s=10)
    return ax


## MAIN CODE ##

# define experiments
experiments_cmems = {
    "smax_from_north_cmems": {
                'fname': '../data/glorys/glorys_current_*.nc',
                'outfile': '../data/lagrangean/smax_from_north_cmems.zarr',
                'center_launch': [-40, -22],
                'launch_depth': 50,
                'time': datetime(2022,1,1),
                'poolSize': 0.5,
                'poolN': 100,
                'aviso': False,
                'cmems': True,
                'run3D': True
                },
    "smax_from_east_cmems": {
                'fname': '../data/glorys/glorys_current_*.nc',
                'outfile': '../data/lagrangean/smax_from_east_cmems.zarr',
                'center_launch': [-39, -26.5],
                'launch_depth': 50,
                'time': datetime(2022,1,1),
                'poolSize': 0.5,
                'poolN': 100,
                'aviso': False,
                'cmems': True,
                'run3D': True
                },
    "smax_from_east_cmems_isopycnal": {
                'fname': '../data/glorys/glorys_isopycnal25.5.nc',
                'outfile': '../data/lagrangean/smax_from_east_cmems_isopycnal.zarr',
                'center_launch': [-39, -26.5],
                'launch_depth': None,
                'time': datetime(2022,1,2),
                'poolSize': 0.5,
                'poolN': 100,
                'aviso': False,
                'cmems': True,
                'run3D': False
                },
    "smax_from_north_cmems_isopycnal": {
                'fname': '../data/glorys/glorys_isopycnal25.5.nc',
                'outfile': '../data/lagrangean/smax_from_northt_cmems_isopycnal.zarr',
                'center_launch': [-40, -22],
                'launch_depth': None,
                'time': datetime(2022,1,2),
                'poolSize': 0.5,
                'poolN': 100,
                'aviso': False,
                'cmems': True,
                'run3D': False
                },
}
experiments_aviso = {
    "smax_from_north_aviso": {
                'fname': '../data/dataset-duacs-nrt-global-merged-allsat-phy-l4_20220101_20220401.nc',
                'outfile': '../data/lagrangean/smax_from_north_aviso.zarr',
                'center_launch': [-40, -22],
                'launch_depth': None,
                'time': datetime(2022,1,1),
                'poolSize': 0.5,
                'poolN': 100,
                'aviso': True,
                'cmems': False
                },
    "smax_from_east_aviso": {
                'fname': '../data/dataset-duacs-nrt-global-merged-allsat-phy-l4_20220101_20220401.nc',
                'outfile': '../data/lagrangean/smax_from_east_aviso.zarr',
                'center_launch': [-39, -26.5],
                'launch_depth': None,
                'time': datetime(2022,1,1),
                'poolSize': 0.5,
                'poolN': 100,
                'aviso': True,
                'cmems': False
                },
}

# combining dictionaries
experiments = {**experiments_cmems, **experiments_aviso}


# running 2D experiments
exp2D = ['smax_from_north_aviso', 'smax_from_east_aviso']

for exp in exp2D:
    days_to_run = (datetime(2022,4,1) - experiments[exp]['time']).days
    
    fieldset, pset = set_running(**experiments[exp])
    
    output_file = pset.ParticleFile(name=experiments[exp]['outfile'], 
                                    outputdt=timedelta(hours=6))
    
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=days_to_run),
                 dt=timedelta(minutes=5),
                 output_file=output_file,
                 verbose_progress=False,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

# running 3D experiments
exp3D = ['smax_from_north_cmems', 'smax_from_east_cmems']

for exp in exp3D:
    days_to_run = (datetime(2022,4,1) - experiments[exp]['time']).days
    fieldset, pset = set_running(**experiments[exp])
    
    output_file = pset.ParticleFile(name=experiments[exp]['outfile'], 
                                    outputdt=timedelta(hours=6))
    
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=days_to_run),
                 dt=timedelta(minutes=5),
                 output_file=output_file,
                 verbose_progress=False,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

