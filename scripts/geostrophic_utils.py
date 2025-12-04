import xarray as xr
from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import gsw
import cmocean.cm as cmo
from matplotlib.gridspec import GridSpec

from utils import get_shelf_edge, nearest, prepare_eddies2plot, subplots_structure, make_map

# import sys
# sys.path.insert(0, '/home/danilo/phd/repos/pytools')
# from pytools.lhico import make_map
# from pytools.sandbox.sandbox import get_uv2

def get_uv2(u,v,int,dir):
    ur = +u*np.cos(np.deg2rad(dir)) + v*np.sin(np.deg2rad(dir))
    vr = -u*np.sin(np.deg2rad(dir)) + v*np.cos(np.deg2rad(dir))
    return ur,vr


################################################################################

def load_satellites(days, datadir, box=[360-47.5, 360-37.5, -28, -20]):

    fnames = glob(f"{datadir}/*")
    satellite = [fname.split("_")[3] for fname in fnames]
    usatellites = list(np.unique(satellite))
    
    # loading tracks for these days
    ds_list = {}
    for t,day in enumerate(days):
        # effectively loading satellite's alongtracks, filtering for time and space during the process
        kw = dict(concat_dim="time", combine="nested",chunks=dict(time=1000))
        ds = {}
        for usatellite in tqdm(usatellites):
            dsi = xr.open_mfdataset(f"{datadir}/*{usatellite}*",**kw).sortby("time")

            # getting only the period of interest
            dsi = dsi.sel(time=slice(day, day + pd.Timedelta(days=1)))

            ind = np.argwhere((
                (dsi.longitude>box[0]) & (dsi.longitude<box[1]) &
                (dsi.latitude>box[2]) & (dsi.latitude<box[3])
            ).values).ravel()

            if len(ind)>0:
                ds[usatellite] = dsi.isel(time=ind).compute()

        # saving each ds for a specific day
        ds_list[str(t)] = ds
        
    return ds_list

################################################################################

def plot_4days_tracks(days, ds_list, extent):
    colors = ['g', 'tab:orange', 'k', 'red', 'tab:blue', 'magenta', 'pink', 'brown']
    
    fig,axes = subplots_structure(2, 2, (15,12), 4)

    for t,day in enumerate(days):
        ax = axes[t]

        ax = make_map(ax=ax)
        ax.set_extent(extent)

        cont = 0 # controller for color
        ds = ds_list[str(t)]

        for k in ds:
            dsi = ds[k]

            # if exist data for this satellite in this day, then we plot the track
            if dsi.time.size > 0:
                for trackid in set(dsi.track.values):
                    dsplot = dsi.where(dsi['track'] == trackid, drop=True)

                    dsplot.plot.scatter(
                        ax=ax,
                        x="longitude", y="latitude", hue="sla_filtered",
                        edgecolor=None, s=10, vmin=-0.2, vmax=0.2,
                        cmap="RdBu", add_guide=False
                    )

                    _ = ax.scatter(dsplot.longitude.values[0] - 360, 
                                   dsplot.latitude.values[0],
                                   marker='s', color=colors[cont],
                                   s=50,
                                   label=f"{trackid}/{int(list(set(dsplot.cycle.values))[0])}/{k}")

                    cont += 1

        ax.set_title(day)
        ax.legend(loc='upper right')

################################################################################

def get_angle(lons,lats):
    
    # identify the most south/west location
    p2 = np.array([
                    np.min(lons),
                    np.min(lats)
                 ])
    
    p1 = np.array([
                    np.max(lons),
                    np.max(lats)
                 ])
    
    # Compute the difference vector between the two points
    diff = p2 - p1

    # Compute the angle in radians from the y-axis (geographical north)
    angle = np.arctan2(diff[1], diff[0])

    # Convert the angle to degrees
    angle_deg = np.degrees(angle)

    return angle_deg

################################################################

def _plot_one_track(ax, dsplot, theta, fix=False, ds_aviso=None, ds_locs=None,
                     cbar_kwargs={'orientation': 'horizontal', 'location': 'bottom','pad': 0.03},
                     color=None):
    # plot background variable if available
    if ds_aviso:
        # Brazil Current isoline of sea level anomaly track
        # ds_aviso.sel(time=day).sla.plot.contour(ax=ax, x='longitude', y='latitude', levels=[0.6], colors='r', linestyles=('--'), linewidths=(2))

        # _ = ds_aviso.sel(time=day).adt.plot.contourf(ax=ax, x='longitude', y='latitude', cmap='RdBu', levels=np.arange(0.1, 1., 0.01),
                                                    # cbar_kwargs=cbar_kwargs)
        CS = ds_aviso.adt.plot.contour(ax=ax, 
                                                     x='longitude', 
                                                     y='latitude', 
                                                     levels=np.arange(0.3, 1., 0.05), 
                                                     colors='gray', 
                                                     linewidths=(.5), 
                                                     kwargs=dict(inline=True))

        # CS = ds_aviso.sel(time=day).sla.plot.contour(ax=ax, x='longitude', y='latitude', levels=[-0.2, -0.1, 0.1, 0.2], colors='k', linewidths=(1), kwargs=dict(inline=True))
        _ = ds_aviso.sla.plot.contourf(ax=ax, x='longitude', y='latitude', cmap='RdBu_r', levels=np.arange(-.5, .5, 0.01),
                                                    cbar_kwargs=cbar_kwargs)

        ax.clabel(CS)
               
    # plotting gridded geostrophic velocity
    if ds_aviso:
        ds_aviso.plot.quiver(ax=ax, x='longitude', y='latitude', u='ugos', v='vgos', scale=10, width=.003, color='gray', )

    lon, lat, ug, vg = compute_geostrophic_velocity(dsplot, theta, fix)

    ax.quiver(lon, lat, ug, vg, scale=5000, width=.003, color='k')

    dsplot.plot.scatter(
        ax=ax,
        x="longitude", y="latitude", 
        edgecolor=None, s=5,
        facecolor=color
    )
    
    # plot first and last observation
    dsplot.isel(time=0).plot.scatter(
        ax=ax,
        x="longitude", y="latitude", 
        edgecolor=None, s=100,
        facecolor='green',
        marker="*"
    )
    
    dsplot.isel(time=-1).plot.scatter(
        ax=ax,
        x="longitude", y="latitude", 
        edgecolor=None, s=100,
        facecolor='red',
        marker="*"
    )
    
################################################################################

def compute_geostrophic_velocity(dsplot, theta, fix, adt=False):  
    """
    this function considers that at the surface the pressure is given by the sea surface anomaly (eta), so the geostrophic
    equation used is:
    
        vg = (g/f) * (del_eta/del_x)
        
    however, we replace del_x by the average distance between each observation within a track, in meters.
    
    Parameters
    ----------
    
    dsplot: xarray.Dataset
        Dataset containing, at least, longitude, latitude and sla_filtered for the calculations
    theta: float
        Track angle relative to the geometrical zero (East), to rotate the geostrophic vectors
    fix: boolean
        Boolean for fixing the direction of the track, as better describe in the Note below.
    
    Note
    ----
    the direction of the track matters when computing the derivative. So, for ascending tracks, we
    must invert all vectors before computing the geostrophic current and then return to the original
    ordenation after.
    
    this way we make sure the geostrophic currents points to the right direction.
    """
    g = 9.8
    
    # reverting the temporal dimension to compute the derivative
    if fix:
        dsplot_ = dsplot.isel(time=slice(None, None, -1))
    else:
        dsplot_ = dsplot.copy(deep=True)

    lat,lon = dsplot_['latitude'].values, dsplot_['longitude'].values
    
    if adt:
        # if adt is activated, then use the sum of SLA and MDT to get the absolute dynamic topography
        eta = (dsplot_['sla_filtered'] + dsplot_['mdt']).values
    else:
        eta = dsplot_['sla_filtered'].values

    # average distance between each observation, in meters
    dist = np.nanmean(gsw.distance(lon, lat))
    del_eta = np.gradient(eta)

    vg = (g/gsw.f(lat)) * (del_eta / dist)
    ug = np.zeros(vg.shape)

    ug,vg = get_uv2(ug, vg, 0, theta)

    return lon, lat, ug, vg

################################################################################

def plot_eddies(ds_locs, day, ax):

    # first eddy filter: by day
    subset = ds_locs.where(ds_locs['time'] == pd.to_datetime(day), drop=True)   

    # second eddy filter: selecting specific eddies by track
    eddies = [118593, 114945]
    subset_list = []
    for eddy in eddies:
        _s = subset.where(subset['track'] == eddy, drop=True)
        subset_list.append(_s)

    subset = xr.concat(subset_list, dim='obs')
    subset = prepare_eddies2plot(subset)

    # inserting anticyclonic eddies contour
    for i in range(subset.dims['obs']):        
        _ = subset.isel(obs=i)
        _anti = _.where(_['eddy'] == 'anticyclonic', drop=True)
        _cycl = _.where(_['eddy'] == 'cyclonic', drop=True)

        ax.plot(_anti['effective_contour_longitude'], _anti['effective_contour_latitude'], 'r')
        ax.text(_anti['longitude'].data.item(), _anti['latitude'].data.item(), _anti['track'].data.item(), fontsize=8)

        ax.plot(_cycl['effective_contour_longitude'], _cycl['effective_contour_latitude'], 'b')
        ax.text(_cycl['longitude'].data.item(), _cycl['latitude'].data.item(), _cycl['track'].data.item(), fontsize=8)
        
################################################################################

# def _plot_single_day(ax, day, tracks, ds_aviso=None, ds_locs=None, 
#                      cbar_kwargs={'orientation': 'horizontal', 'location': 'bottom','pad': 0.03},
#                      contourf_kwargs={"levels": np.arange(-0.5, 0.5, 0.01), "cmap": "RdBu_r"},
#                      colors=None):
#     # load shelf edge
#     # shelf = get_shelf_edge()
#     # ax.plot(shelf['lon'], shelf['lat'], c='gray', ls='--')

#     # plot background variable if available
#     if ds_aviso:
#         # Brazil Current isoline of sea level anomaly track
#         # ds_aviso.sel(time=day).sla.plot.contour(ax=ax, x='longitude', y='latitude', levels=[0.6], colors='r', linestyles=('--'), linewidths=(2))

#         CS = ds_aviso.sel(time=day).adt.plot.contour(ax=ax, x='longitude', y='latitude', levels=np.arange(0.3, 1., 0.05), colors='k', linewidths=(.5), kwargs=dict(inline=True))

#         _ = ds_aviso.sel(time=day).sla.plot.contourf(ax=ax, x='longitude', y='latitude', 
#                                                      cbar_kwargs=cbar_kwargs,
#                                                      **contourf_kwargs)

#         ax.clabel(CS)
               
#     # plotting gridded geostrophic velocity
#     if ds_aviso:
#         ds_aviso.sel(time=day).plot.quiver(ax=ax, x='longitude', y='latitude', u='ugos', v='vgos', scale=10, width=.003, color='gray', )
        
#     # plotting along track geostrophic
#     for sat in tracks[day]:
#         if colors:
#             c = colors[sat]
#         else:
#             c = None
            
#         dsplot = tracks[day][sat]['data']

#         lon, lat, ug, vg = compute_geostrophic_velocity(dsplot, tracks[day][sat]['theta'], tracks[day][sat]['fix'])

#         ax.quiver(lon, lat, ug, vg, scale=5000, width=.003)

#         dsplot.plot.scatter(
#             ax=ax,
#             x="longitude", y="latitude", 
#             edgecolor=None, s=20,
#             facecolor=c
#         )

################################################################################

def daily_geostrophic_current(day, tracks, ds_aviso=None, ds_locs=None, savefig=None):
    """ 
    plot a single figure with a single day selected, with:
        - background color: absolute dynamic topography [m]
        - background contour: absolute dynamic topography [m]
        - gray quiver: gridded aviso geostrophic velocity [m/s]
        - black quiver: geostrophic velocity derived from alongtrack altimeter [m/s]
    """
    
    # load shelf edge
    shelf = get_shelf_edge()
    
    # build figure structure
    fig,axes = subplots_structure(1,1, (15,10), 1)
    ax = axes[0]

    ax = make_map(ax=ax)
    ax.set_extent([-46, -40, -27, -22])
    ax.plot(shelf['lon'], shelf['lat'], c='gray', ls='--')
    
    _plot_single_day(ax, day, tracks, ds_aviso=ds_aviso, ds_locs=None, cbar_kwargs={})

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

################################################################################

def _plot_single_day(ax, day, tracks, adt=False, ds_aviso=None, ds_locs=None, 
                     cbar_kwargs={'orientation': 'horizontal', 'location': 'bottom','pad': 0.03},
                     contourf_kwargs={"levels": np.arange(-0.5, 0.5, 0.01), "cmap": "RdBu_r"},
                     colors=None, transect=None, shelf=None):
    # load shelf edge
    if shelf != None:
        ax.plot(shelf['lon'], shelf['lat'], c='white', ls='--')

    # plot background variable if available
    if ds_aviso:
        _ = add_aviso(ax, ds_aviso.sel(time=day), 
                      cbar_kwargs=cbar_kwargs, 
                      contourf_kwargs=contourf_kwargs,
                      quiver_kwargs={"scale": 10, "width": .003, "color": "gray", "add_guide": False})
               
    # plotting along track geostrophic
    for sat in tracks[day]:
        if colors:
            c = colors[sat]
        else:
            c = 'tab:blue'
            
        dsplot = tracks[day][sat]['data']
        theta = tracks[day][sat]['theta']
        fix = tracks[day][sat]['fix']

        # add geostrophic current over the map
        _ = add_geostrophic_velocity(ax, dsplot, theta, fix, adt=adt,
                                     quiverkey=[-44.5, -22.2, 0.5, '0.5 m/s'],
                                     quiver_kwargs={"scale": 3.6, "units": "width", 
                                                    "width": .003, "color": "k"})

        # add track scatter
        _ = add_track_scatter(ax, dsplot, color=c)
        
        if transect:
            _ = add_transect(ax, transect['lons'], transect['lats'])
        
################################################################################

def add_aviso(ax, ds_aviso, 
              quiver_kwargs={"scale": 10, "width": .003, "color": "gray"},
              contourf_kwargs={"levels": np.arange(-0.5, 0.5, 0.01), "cmap": "RdBu_r"},
              cbar_kwargs={'orientation': 'horizontal', 'location': 'bottom','pad': 0.03}):
    
    # add contour/contourf
    CS = ds_aviso.sla.plot.contour(ax=ax, 
                                   x='longitude', 
                                   y='latitude', 
                                   levels=np.arange(0.3, 1., 0.05), 
                                   colors='gray', 
                                   linewidths=(.5), 
                                   kwargs=dict(inline=True))
    _ = ds_aviso.sla.plot.contourf(ax=ax, 
                                   x='longitude', 
                                   y='latitude',
                                   cbar_kwargs=cbar_kwargs,
                                   **contourf_kwargs)
    _.colorbar.set_label('sea level anomaly [m]', fontsize=12)
    
    ax.clabel(CS)
    
    # add quiver
    ds_aviso.plot.quiver(ax=ax, 
                         x='longitude', 
                         y='latitude', 
                         u='ugos', 
                         v='vgos', **quiver_kwargs)
    
    return ax

################################################################################

def add_geostrophic_velocity(ax, dsplot, theta, fix, adt=False,
                             quiverkey=[-45, -22.5, 0.5, '0.5 m/s'],
                             quiverkey_kws=dict(labelpos='S', color='k', coordinates='data'),
                             quiver_kwargs=dict(scale=5000, width=.003, color='k')):
    
    # calculations
    lon, lat, ug, vg = compute_geostrophic_velocity(dsplot, theta, fix, adt)

    # plot
    q = ax.quiver(lon, lat, ug, vg, **quiver_kwargs)
    
    # add quiverkey
    if quiverkey != None:
        qk = ax.quiverkey(q,
                          quiverkey[0], # x position
                          quiverkey[1], # y position
                          quiverkey[2], # unit 
                          quiverkey[3], # label
                          **quiverkey_kws).set_zorder(11)

    return ax

################################################################################

def add_track_scatter(ax, dsplot, color='tab:blue'):
    dsplot.plot.scatter(
        ax=ax,
        x="longitude", y="latitude", 
        edgecolor=None, s=5,
        facecolor=color
    )
    
#     # plot first and last observation
#     dsplot.isel(time=0).plot.scatter(
#         ax=ax,
#         x="longitude", y="latitude", 
#         edgecolor=None, s=100,
#         facecolor='green',
#         marker="*"
#     )
    
#     dsplot.isel(time=-1).plot.scatter(
#         ax=ax,
#         x="longitude", y="latitude", 
#         edgecolor=None, s=100,
#         facecolor='red',
#         marker="*"
#     )

################################################################################
    
def add_transect(ax, lons, lats):
    ax.plot(lons, lats, color='r', lw=3)
    
    return ax
    
################################################################################

def ultimate_plot_one_track(ax, dsplot, theta, 
                            fix=False, ds_aviso=None, ds_locs=None,
                            cbar_kwargs={'orientation': 'horizontal', 
                                         'location': 'bottom','pad': 0.03},
                            quiver_kws_aviso={"scale": 10, "width": .003, "color": "gray", "add_guide": False},
                            quiver_kws_geo=dict(scale=5000, width=.003, color='k'),
                            color=None,
                            transect=None):
   
    # background variable if available
    if ds_aviso:
        _ = add_aviso(ax, ds_aviso, cbar_kwargs=cbar_kwargs, quiver_kwargs=quiver_kws_aviso)
        
    # add geostrophic current over the map
    _ = add_geostrophic_velocity(ax, dsplot, theta, fix, quiver_kwargs=quiver_kws_geo)
    
    # add track scatter
    _ = add_track_scatter(ax, dsplot, color=color)
    
    # add transect location
    if transect:
        _ = add_transect(ax, transect['lons'], transect['lats'])
    
################################################################################


def plot_timeseries(dsplot, theta, day, sat, fix=False, axes=None, distance=False,
                    quiver_kws_plot=dict(scale=3.6, units="width", width=.002, color='k')):
    
    
    # creating and plotting sla plot and stickplot
    if len(axes) == 0:
        fig,axes = plt.subplots(nrows=2, figsize=(15,10), sharex=True)
        
    # extracting variables based on arguments
    sla = dsplot.sla_filtered.values
    
    if distance:
        x = np.cumsum(gsw.distance(dsplot['longitude'].values, dsplot['latitude'].values)) / 1000 # km
        x = np.insert(x, [0], [0]) 
        xlabel = 'distance [km]'
    else:
        x = dsplot['time'].values
        xlabel = 'time'
        
    # plot line for sea level anomaly
    _ = axes[0].plot(x, sla)
    _ = axes[0].scatter(x[0], sla[0], marker='*', c='green')
    _ = axes[0].scatter(x[-1], sla[-1], marker='*', c='red')

    # plot stickplot for geostrophic velocity derived from the sla
    # angle is zero because we don't want to rotate
    lon, lat, ug, vg = compute_geostrophic_velocity(dsplot, 
                                                    0, 
                                                    fix) 

    q = axes[1].quiver(x, 
                       [[0]*len(x)], 
                       ug, vg, 
                       **quiver_kws_plot)

    # tweak to fix the magnitude of the vector
    _ = axes[1].plot(x, vg, zorder=0, c='w')

    _ = [ax.grid(True, ls='--', c='gray', alpha=.5) for ax in axes]
    _ = axes[0].set_title('')
    _ = axes[0].set_title(day, loc='left')
    _ = axes[0].set_title(sat, loc='right')

    _ = axes[0].set_ylabel('sea level anomaly [m]')
    _ = axes[1].set_ylabel('geostrophic curr [m s-1]')
    _ = axes[1].set_xlabel(xlabel)

    if fix:
        axes[1].invert_xaxis()
        
        
################################################################################

def build_transect(days_keys, fpath=None):
    
    if not fpath:
        fpath = '/home/danilo/phd/thesis/chapter_colab/scripts/section-ctd_dalton/data/treated/SANAGU2022/shelf/20220516_v0.2_nc'
    
    stations_dict = {
        'G1': {
            'before':'SANAGU14_G1_CTD_2_OT',
            'after': 'SANAGU15_G1_CTD_2_OT',
        },
        'G2': {
            'before':'SANAGU14_G2_CTD_1_OT',
            'after': 'SANAGU15_G2_CTD_2_OT',
        },
        'G3': {
            'before':'SANAGU14_G3_CTD_1_OT',
            'after': 'SANAGU15_G3_CTD_2_OT',
        }
    }
    
    transect = {
        days_keys[0]: {"lons": [],
                       "lats": []
            },
        days_keys[1]: {"lons": [],
                       "lats": []
            },
    }
    
    for station in stations_dict.keys():
        ds1 = xr.open_dataset(f"{fpath}/{stations_dict[station]['before']}.nc")
        ds2 = xr.open_dataset(f"{fpath}/{stations_dict[station]['after']}.nc")

        transect[days_keys[0]]['lons'].append(ds1.attrs['lon'])
        transect[days_keys[0]]['lats'].append(ds1.attrs['lat'])

        transect[days_keys[1]]['lons'].append(ds2.attrs['lon'])
        transect[days_keys[1]]['lats'].append(ds2.attrs['lat'])
        
    return transect

################################################################################

def compute_distance(lons, lats):
    if lats[0] < lats[-1]:
        # descending, needs to invert
        lats = np.flip(lats)
        lons = np.flip(lons)
        
    x = np.cumsum(gsw.distance(lons,lats)) / 1000 # km
    x = np.insert(x, [0], [0])
    
    return x


################################################################################

def plot_timeseries2(dsplot, theta, day, sat, fix=False, axes=None, distance=False,
                    quiver_kws_plot=dict(scale=3.6, units="width", width=.002, color='k')):
           
    # creating and plotting sla plot and stickplot
    if len(axes) == 0:
        fig,axes = plt.subplots(nrows=2, figsize=(15,10), sharex=True)
        
    # extracting variables based on arguments
    sla = dsplot.sla_filtered.values
    
    if distance:
        x = compute_distance(dsplot['longitude'].values, dsplot['latitude'].values)
        xlabel = 'distance [km]'
    else:
        x = dsplot['time'].values
        xlabel = 'time'
        
    # plot line for sea level anomaly
    _ = axes[0].plot(x, sla)
    _ = axes[0].scatter(x[0], sla[0], marker='*', c='green')
    _ = axes[0].scatter(x[-1], sla[-1], marker='*', c='red')

    # plot stickplot for geostrophic velocity derived from the sla
    # angle is zero because we don't want to rotate
    lon, lat, ug, vg = compute_geostrophic_velocity(dsplot, 
                                                    0, 
                                                    fix) 

    q = axes[1].quiver(x, 
                       [[0]*len(x)], 
                       ug, vg, 
                       **quiver_kws_plot)

    # tweak to fix the magnitude of the vector
    _ = axes[1].plot(x, vg, zorder=0, c='w')

    _ = [ax.grid(True, ls='--', c='gray', alpha=.5) for ax in axes]
    _ = [ax.margins(x=0) for ax in axes]
    
    _ = axes[0].set_title('')
    _ = axes[0].set_title(day, loc='left')
    _ = axes[0].set_title(sat, loc='right')

    _ = axes[0].set_ylabel('sea level anomaly [m]')
    _ = axes[1].set_ylabel('geostrophic curr [m s-1]')
    _ = axes[1].set_xlabel(xlabel)