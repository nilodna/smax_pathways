import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#############################################################################################v

# create South Brazil Bight map, returning an ax object
def make_map(extent=[-50,-39,-30,-22], ax=None, facecolor=None, deltax=3, deltay=3,
            drawlines=True, drawlinebottom=True, drawlineleft=True, drawlinetop=False, drawlineright=False):
    """Make map using the box send in extent variable, with a minimap on the right
    lower corner.

    Parameters
    ----------
    extent : list
        Coordinates of each corner. [lower left long, upper right long, lower
        left latitude, upper right latitude].

    Usage
    -----
    >>> make_map(extent=[-60,-50,-30,-40])

        HOW TO USE

        #limites
        upper_lat = -22.
        upper_lon = -39.
        lower_lat = -30.
        lower_lon = -49.

        # creating plot
        fig = plt.figure(figsize=(15/2.54,15/2.54))
        ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())

        extent = [lower_lon, upper_lon, lower_lat, upper_lat]
        ax = make_map(ax,extent=extent)

        ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax = make_map(ax,extent=extent)

    """

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        
    if facecolor == None:
        facecolor = cfeature.COLORS['land']
        
    # Create a feature for regions at 1:10m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='physical',name='coastline',scale='10m',
        facecolor=facecolor)

    ### ADD FEATURES
    # adding coastline from NaturalEarth dataset
    ax.add_feature(states_provinces, edgecolor='black',linewidth=.1)


    ## CONFIGURING MERIDIANS AND PARALLELS
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=drawlines,linewidth=.4,
                        color='gray', alpha=0.5, linestyle='--',
                        xlocs=list(np.arange(extent[0], extent[1], deltax)),
                        ylocs=list(np.arange(extent[2], extent[3], deltay)),)

    gl.xlabels_top = drawlinetop
    gl.xlabels_bottom = drawlinebottom

    gl.ylabels_right = drawlineright
    gl.ylabels_left = drawlineleft

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'color': 'black', 'fontsize': '14'}
    gl.xlabel_style = {'color': 'black', 'fontsize': '14'}

    return ax

#############################################################################################


def get_shelf_edge(depfile='../data/gebco_2019.nc', extent=[-47.5, -37.5, -28, -20],
                   rename={'elevation': 'depth'}):
    bathy = xr.open_dataset(depfile)

    # rename variables for a padronization
    bathy = bathy.rename(rename)
    bathy = bathy.where(bathy['depth'] < 0, drop=True)


    # regional selection (for both dataset loaded)
    lon_lower, lon_upper = extent[0], extent[1] #-50, -40
    lat_lower, lat_upper = extent[2], extent[3] #-29, -23

    # bathymetry
    bathy = bathy.sel(lon=slice(lon_lower, lon_upper),
                    lat=slice(lat_lower, lat_upper))
    isobath = -200

    plt.ioff()
    CR = bathy.depth.plot.contour(levels=[isobath])
    # here we extract the coordinates
    fit = CR.allsegs[0][0]

    # convert locations into a xarray.Dataset
    ds_locs = xr.Dataset()
    ds_locs['lon'] = (('locations'), fit[:,0])
    ds_locs['lat'] = (('locations'), fit[:,1])
    
    plt.close('all')
    plt.ion()
    
    return ds_locs

#############################################################################################


def subplots_structure(ncols, nrows, figsize, n):
    '''
    ncols : number of columns
    nrows : number of rows
    figsize: tuple with figure size
    n : how much subplots must be created
    '''
    # plotting composites
    fig = plt.figure(figsize=figsize)
    
    axes = []
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i+1, projection=ccrs.PlateCarree())
        
        axes.append(ax)
        
    return fig,axes

#############################################################################################

# filtering eddies in space and time
def filtering_eddies(atlas, lon_range, lat_range, time_range):
    """
    
    lon_range : list
    
    lat_range : list
    
    time_range : pandas.DatetimeIndex
    
    """    
    # since it is a massive dataset, we should first filter by 
    # space, to reduce the size of the dataset and improve the 
    # identification performance in the temporal dimension
    lonatlas= atlas['longitude']
    lonn = lonatlas+180
    lonatlas2 = np.remainder(lonn,360)-180
    latatlas = atlas['latitude']

    idxs = np.where((lonatlas2 > lon_range[0]) & (lonatlas2 < lon_range[1]) & 
                    (latatlas > lat_range[0]) & (latatlas < lat_range[1]))
    
    ds_loc = atlas.isel(obs=idxs[0])

    ds_list = []    
    for n,date in enumerate(time_range):
        subset = ds_loc.where(ds_loc['time'] == date, drop=True)
        # fix longitude
        subset['longitude'] = np.remainder((subset['longitude']+180), 360) - 180
        subset['longitude'].attrs = atlas['longitude'].attrs

        subset['effective_contour_longitude'] = np.remainder((subset['effective_contour_longitude']+180), 360) - 180
        subset['effective_contour_longitude'].attrs = atlas['effective_contour_longitude'].attrs

        # inserting anticyclonic eddies contour
        for i in range(subset.dims['obs']):
            print(i)
            
            _ = subset.isel(obs=i)
            ds_list.append(_)
            
    return xr.concat(ds_list, dim='obs')

#############################################################################################

def loading_eddies(datadir, lon_range, lat_range, time_range):
    atlas_anticyclonic = xr.open_dataset(f"{datadir}/../eddies_atlas/Eddy_trajectory_nrt_3.2exp_anticyclonic_20180101_20230224.nc")
    atlas_cyclonic = xr.open_dataset(f"{datadir}/../eddies_atlas/Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20230224.nc")
    
    # eddies detection
    eddies_anticyclonic = filtering_eddies(atlas_anticyclonic, lon_range, lat_range, time_range)
    eddies_cyclonic = filtering_eddies(atlas_cyclonic, lon_range, lat_range, time_range)

    # creating label to identify anti and cyclonic eddies
    eddies_anticyclonic['eddy'] = (('obs'), np.tile(['anticyclonic'], (eddies_anticyclonic.dims['obs'],1))[:,0])
    eddies_cyclonic['eddy'] = (('obs'), np.tile(['cyclonic'], (eddies_cyclonic.dims['obs'],1))[:,0])

    # put it all together
    ds_locs = xr.concat([eddies_anticyclonic, eddies_cyclonic], dim='obs')
    
    return ds_locs


#############################################################################################
def detect_JPtechnique(atlas, lon_range, lat_range, time_range):
    # TODO: acho que to deixando passar eventos ainda. Preciso rever o código
    tempoatlas = atlas['time']
    lonatlas= atlas['longitude']
    lonn = lonatlas+180
    lonatlas2 = np.remainder(lonn,360)-180
    latatlas = atlas['latitude']

    # identifying tracks that pass by the South Brazil Bight margin in a specific
    # temporal coverage
    idxs = np.where((lonatlas2 > lon_range[0]) & (lonatlas2 < lon_range[1]) & 
                    (latatlas > lat_range[0]) & (latatlas < lat_range[1]) & 
                    (tempoatlas >= time_range[0]) & (tempoatlas <= time_range[1]))
        
    # found obs with track within item list
    subset = atlas.isel(obs=idxs[0])
    
    # fix longitude
    subset['longitude'] = np.remainder((subset['longitude']+180), 360) - 180
    subset['longitude'].attrs = atlas['longitude'].attrs
    
    subset['effective_contour_longitude'] = np.remainder((subset['effective_contour_longitude']+180), 360) - 180
    subset['effective_contour_longitude'].attrs = atlas['effective_contour_longitude'].attrs
    
    # identifiying OBS for each eddy passing in the region
    ax = make_map()
    ds.sel(time=_['time'], method='nearest').sla.plot.pcolormesh(ax=ax, x='longitude', y='latitude', cmap='RdBu')
    for i in range(subset.dims['obs']):
        print(i)
        
        _ = subset.isel(obs=i)
        
        
        ax.plot(_['effective_contour_longitude'], _['effective_contour_latitude'], 'k')
        ax.text(_['longitude'].data.item(), _['latitude'].data.item(), _['track'].data.item())
        
        
        
    #### segundo método
    # TODO: acho que to deixando passar eventos ainda. Preciso rever o código

            
            
#############################################################################################


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

#############################################################################################


def prepare_eddies2plot(subset):
    # fix longitude
    subset['longitude'] = np.remainder((subset['longitude']+180), 360) - 180
    subset['effective_contour_longitude'] = np.remainder((subset['effective_contour_longitude']+180), 360) - 180

    return subset

#############################################################################################

def set_figure(analysis=True):
    """
    Options are:
    - analyis == True    -- larger figures
    - analysis == 'plot' -- article quality figures
    - analysis == None   -- default
    """
    # configuring matplotlib
    from matplotlib import rcParams
    
    if analysis:
        rcParams['figure.figsize']    = [30,15] # width for 1.5column large figure
        rcParams['font.size']         = 12
        rcParams['axes.linewidth']    = 0.5
        rcParams['xtick.major.width'] = 0.4
        rcParams['ytick.major.width'] = 0.4
    elif analysis == 'plot':
        rcParams['figure.figsize']    = [14/2.54,12/2.54] # width for 1.5column large figure
        rcParams['font.size']         = 8
        rcParams['axes.linewidth']    = 0.1
        rcParams['xtick.major.width'] = 0.4
        rcParams['ytick.major.width'] = 0.4
    else:
        plt.rcdefaults()
        
#############################################################################################

def add_observation_locations(ax, timerange, sanagu=False, argo=False, glider=False):
    color = 'tab:orange'
    
    if sanagu:
        locs = pd.read_csv(sanagu, index_col=0)
        ax.scatter(locs['lon'], locs['lat'], c=color, s=5, zorder=5)
        
    if glider:
        # loading glider track
        ds = xr.open_dataset(glider)

        ds = ds.sel(time=slice(timerange[0], timerange[1]))
        
        ax.scatter(ds['longitude'], ds['latitude'], color=color, zorder=5, s=5)
        
    if argo:
        # loading ARGO profiles
        lista = []
        for f in argo:
            ds_ = xr.open_dataset(f)
            ds_ = ds_.where(ds_['TIME'] <= timerange[1], drop=True)
            lista.append(ds_)
        
        ds_argo = xr.concat(lista, dim='N_PROF')
        ax.scatter(ds_argo['LONGITUDE'], ds_argo['LATITUDE'], color=color, zorder=3, s=5)
        
    return ax