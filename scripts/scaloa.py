from rtree import index
import numpy as np
import pandas as pd
import glob
import os.path as osp
import xarray as xr
import logging
import gsw
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import ticker
from scipy.interpolate import griddata

from utils import insert_stations_marker

def find_nearest(lon,lat,ilon,ilat,k=1):
    """ Find nearest coordinates to a given coordinate, given by ilon and ilat,
    in a grid defined by lon and lat arguments.

    Parameters
    ----------
    lon     : np.ndarray
        2D array with gridded longitudes.
    lat     : np.ndarray
        2D array with gridded latitudes.
    ilon    : float
        Specific longitude of the coordinate of interest.
    ilat    : float
        Specific latitude of the coordinate of interest.

    Returns
    -------
    iss,jss : list
        Indexes of the i,j pair of the nearest point found.
    """
    from scipy import spatial

    # generating coords pairs of longitude and latitude
    lo,la = lon.ravel(), lat.ravel()
    coords = []
    for o,a in zip(lo,la):
         coords.append((o,a))

    # define tuple with the interest coordinates
    x = [(ilon,ilat)]

    # using cKDTree to find the nearest points
    dists,indexes = spatial.cKDTree(coords).query(x, k=k)

    # indexes are given considering the flatten 2D arrays for lon and lat. We must
    # now transform this single index in a i,j form, so we can use in the 2D arrays
    # of the model
    points = []

    for index in indexes:
        points.append(coords[index])

    # converting list to array
    points = np.asarray(points)

    # findind indexes from lat and lon
    ind = []

    for p in points:
        tmp = np.where(np.logical_and(lon == p[0], lat == p[1]))
        ind.append(tmp)

    ind = np.asarray(ind)

    # separating i and j in two lists
    iss=[]
    jss=[]

    for i,j in ind:
        iss.append(int(i))
        jss.append(int(j))

    return iss,jss

def prepare_reference_table(fpath, ctdpath=None, ctdformat=None, decimal='.'):
    """Reads the csv reference table with the metadata on each station at
    each section. The function also includes a column with  ctd 'paths'
    in the dataframe. The csv must contain a column 'station', with the
    ctd file name (WITHOUT extension)


    Args:
        fpath (str):
            reference table file name (.csv)

        ctdpath (str, optional):
            ctd directory path with nc files produced by cnv2nc
                If ctdformat is None, this argument is ignored
        ctdformat (str, optional):
            file extension of the ctd files.
                If None, the original table is returned
                If 'nc', the files outputtedby cnv2nc will be read.

    Returns (pandas dataframe):
        Dataframe with nformation of the csv table
        and correspoding ctd paths
    """    
    ctdf = ctdformat
    df = pd.read_csv(fpath, decimal=decimal)  # read table 
    
    if ctdformat is not None:
        ctdpaths = [osp.join(ctdpath, f'{i}.{ctdf}') for i in df['station']] 
        ctdpaths1 = []
        for i in ctdpaths:
            aux = glob.glob(i)
            ctdpaths1.append(aux) if len(aux)>0 else print(i)
        logging.info(f"include 'paths' as a dataframe column with  the paths of ctd files with {ctdformat} format")
        df['paths'] = ctdpaths

    return df

def read_section(df, section, reference_table=[]):
    """Creates a ctd Dataframe based on the section and
    df (reference metadata dataframe created with prepare_reference_table)

    Args:
        df (pandas dataframe):
            dataframe created by prepare_reference_table method
        section (str):
            A given section ID in df
        reference_table (list, optional):
            Strings contained in this list are referenced to column keys in 
            df. The respective values are added in respective columns in the
            dataframe output . Defaults to [].

    Returns (pandas dataframe):
        A dataframe with the information of a given section with
        information updated from df[reference_table]
    """    
    # -- section -- #
    groups = df.groupby('section')
    dfg = groups.get_group(section)

    # -- read using netcdf -- #
    # netcdf of a given ctd data MUST contain lon, lat, time as 
    # attributes
    dfaux1 = []
    id = {}
    for j,i in dfg.iterrows():
        id[j] = i['station']
        nc = xr.open_dataset(i['paths'])
        dfaux = nc.to_dataframe()
        dfaux['station'] = i['station']  # sharing station id to dfaux
        
        # including position information into dfaux
        try:
            for k in ['lon', 'lat', 'time']:
                dfaux[k] = nc.attrs[k]
        except KeyError:
            raise KeyError(f"{i['station']} netcdf file does not contain {k} as an attribute")

        # including reference table information into dfaux
        for k in reference_table:
            if ~np.isnan(float(i[k])):
                dfaux[k] = float(i[k])
            else:
                pass
        
        dfaux1.append(dfaux)

    dfout = pd.concat(dfaux1, axis=0) 
    return dfout, dfg

def scaloa_old(xr,yr,zmr, xg,yg, corrlenx, corrleny, err):
    # grid
    xi = xg.copy()
    yi = yg.copy()

    # observation
    x  = xr.copy()
    y  = yr.copy()
    t = zmr.copy()

    # squared distance matrix between observations
    dx = (x[None,:] - x[:,None])**2
    dy = (y[None,:] - y[:,None])**2
    # dphi = abs(phi[:,None] - phi[None,:])/(phi[:,None]**2+phi[None,:]**2)**0.5 * (0.5*(dx/dx.std()+dy/dy.std()))

    # squared distance matrix between observation and grid
    if len(xi.shape) != 1:
        xc = xi.ravel()
        yc = yi.ravel()
    else:
        xc = xi[:]
        yc = yi[:]

    dxc = (xc[:,None] - x[None,:])**2
    dyc = (yc[:,None] - y[None,:])**2
    # dphic = abs(phic[:,None] - phi[None,:])/(phic[:,None]**2+phi[None,:]**2)**0.5 *(0.5*(dxc/dxc.std()+dyc/dyc.std()))
    # print(np.nanmin(dphic))
    # print(np.nanmax(dphic))
    ## dc2 = (xc[:,None] - x[None,:])**2 + (yc[:,None] - y[None,:])**2


    A = (1 - err) * np.exp(-(dx/corrlenx**2 + dy/corrleny**2))# +  dphi/corrlenphi**2))
    C = (1 - err) * np.exp(-(dxc/corrlenx**2 + dyc/corrleny**2))# + dphic/corrlenphi**2))

    del(dxc, dyc, dx, dy)

    A = A + err * np.eye(len(A))

    # tp = None
    ep = 1 - np.sum(C.T * np.linalg.solve(A, C.T), axis=0) / (1 - err)
    # if any(t)==True: ##### was t!=None:
    t = t.ravel()
    tp = np.dot(C, np.linalg.solve(A, t))

    tp.reshape(xi.shape)
    return tp, ep

def scaloa(xxr, yr, zmr, xg, yg, corrlenx, corrleny, err, max_dist=1.5):
    # build R-tree for observations
    obs_tree = index.Index()
    for i in range(len(xxr)):
        obs_tree.insert(i, (xxr[i], yr[i]))

    # grid
    xi = xg.copy()
    yi = yg.copy()

    # observation
    x = xxr.copy()
    y = yr.copy()
    t = zmr.copy()
    n = len(x)

    # squared distance matrix between observation and grid
    if len(xi.shape) != 1:
        xc = xi.ravel()
        yc = yi.ravel()
    else:
        xc = xi[:]
        yc = yi[:]

    # initialize output arrays
    tp = np.zeros(xi.shape)
    ep = np.zeros_like(tp)

    # loop over grid points
    for i in range(len(xc)):
        # find nearby observations using R-tree
        bbox = (xc[i]-corrlenx, yc[i]-corrleny, xc[i]+corrlenx, yc[i]+corrleny)
        nearby_obs = list(obs_tree.intersection(bbox))

        # exclude observations beyond max_dist if specified
        if max_dist is not None:
            nearby_obs = [j for j in nearby_obs if ((x[j]-xc[i])**2 + (y[j]-yc[i])**2)**0.5 <= max_dist]

        # compute squared distance matrix between nearby observations
        scale = (x[nearby_obs][:, None] - x[nearby_obs][None, :])**2/corrlenx**2
        scale += (y[nearby_obs][:, None] - y[nearby_obs][None, :])**2/corrleny**2
        A = (1 - err) * np.exp(-scale)

        # compute squared distance matrix between nearby observations and grid point
        scalec = (xc[i]-x[nearby_obs])**2/corrlenx**2
        scalec += (yc[i]-y[nearby_obs])**2/corrleny**2
        C = (1 - err) * np.exp(-scalec)

        # add error to diagonal of A
        np.fill_diagonal(A, np.diag(A) + err)

        # compute predicted z-value and estimation error
        if len(nearby_obs) > 0:
            tp[i] = np.dot(C, np.linalg.solve(A, t[nearby_obs]))
            ep[i] = 1 - np.sum(C * np.linalg.solve(A, C.T), axis=0) / (1 - err)

    return tp, ep

def objective_analysis_varb(df, var, mask=False, corrlenx=160000, corrleny=20, err_thresh=0.1):
    # preparing input data to ao.scaloa
    dist = df['dist'].values
    n = len(set(df['station']))  # number of profiles in a section
    dept = df.index.values # vertical distances
    varib = df[var].values.ravel()  # variable
    maskVal = ~np.isnan(varib)


    x = np.linspace(np.nanmin(dist), np.nanmax(dist),100)
    y = np.array(sorted(list(set(dept))))

    # input data to scaloa
    xm, ym = np.meshgrid(x, y)

    # objective analysis use a 2d gaussian surface to interpolate the value at
    # a given point.
    # if corrlenx=100000, scaloa will use 100km in the horizontal direction
    # if corrleny=10, scaloa 10m will use 10m in the vertical direction
    z_ao , err = scaloa_old( dist[maskVal],
                         dept[maskVal],
                         varib[maskVal],
                         xm.ravel(),
                         ym.ravel(),
                         corrlenx=corrlenx,
                         corrleny=corrleny,
                         err=err_thresh)

    z_ao_m = z_ao[:].reshape(y.size, x.size)
    errm = err.reshape(y.size, x.size)

    z_ao_m_mask = z_ao_m.copy()
    if mask:
            z_ao_m_mask[errm>err_thresh] = np.nan

    return xm,ym,  z_ao_m_mask, errm

def lonlatdepth_extend(nc_etopo, lon, lat, dist, n=20, dataset='gebco2019',varname='elevation'):
    """Extracts bathymetry values over a section defined by lon, lat from a netcdf file

    Args:
        nc_etopo (xr.Dataset): netcdf of a bathymetry dataset
        lon (np.ndarray): 1D array (size=N)
        lat ([np.ndarray): 1D array (size=N)
        dist (np.ndarray): 1D array (size=N)
        n (int, optional): lon lat will be interpolated in between the points.
                           this is the number of points used in the interpolation
        dataset (str, optional): Database used as reference. Defaults to 'gebco2019',
                                 we have 'rezende' and 'etopo' as other options.
        varname (str, optional): name of the bathymetry variable. Defaults to 'elevation'.
    """
    def multiple_linspace(varb, n=n):

        # determine interval of varb
        dvarb = []
        for l in zip(varb[:-1], varb[1:]):
            dvarb.append(list(np.linspace(l[0], l[1], n)))

        # flatten everything
        dvarb1 = [subitem for item in dvarb for subitem in item]
        return dvarb1

    dlon = multiple_linspace(lon)
    dlat = multiple_linspace(lat)
    ddst = multiple_linspace(dist)

    if dataset == 'rezende':
        # get the value of depth using the nearest value
        glon = nc_etopo.lon.values
        glat = nc_etopo.lat.values
        gdep = nc_etopo.depth.values

        depth = []
        for ilon,ilat in zip(dlon,dlat):
            indexes = find_nearest(glon,glat,ilon,ilat)
            depth.append(gdep[indexes[0], indexes[1]][0])

        depth = [x*(-1) for x in depth] # convert depth from positive to negative
    else:
        # get the value of depth using the nearest value
        depth = []
        for i in zip(dlon, dlat):
            d = nc_etopo[varname].sel(lon=i[0], lat=i[1], method='nearest')
            depth.append(float(d.values))

    return dlon, dlat, ddst, depth

def get_lonlat_cruises(df, sectionslist):
    """get lon and lat from the reference table

    Args:
        df (pd.DataFrame): reference table
        sectionslist ([type]): list with the sections ID (['A', 'B'], for instance)

    Returns:
        [tuple]: (lon, lat)
    """ 
    lons = []
    lats = []
    for i in sectionslist:
        df1,_ = read_section(df, section=i, reference_table=reference_table)   
        df1 = df1[['lon', 'lat']]
        lon =df1['lon'].values
        lat =df1['lat'].values
        ll = np.array(list(set(zip(lon,lat))))
        lons.append(list(ll[:,0]))
        lats.append(list(ll[:,1]))

    lons = [item for sublist in lons for item in sublist]
    lats = [item for sublist in lats for item in sublist]
    return lons, lats

def subroutine_TSsection(cdicts1, l, ylimdict, fig=None, ax=None, ax_cbar=None, secids=None, add_colorbar=True):
    if (fig is None) and (ax is None):
        fig = plt.figure(figsize=[7, 3.5])
        ax = fig.add_subplot()
        ax_cbar = ax
    else:
        pass
    # -- section plot -- #
    kwcf = {'levels':np.arange(cdicts1['vmin'],cdicts1['vmax'],cdicts1['deltav']),
            'zorder':1,
            'cmap':cdicts1['cmap'],
            'extend': 'both',}

    kwcr = {'levels':cdicts1['contour'],
            'zorder':1,
            'colors': 'k',
            'linewidths':1}

    kwsc = {
        'vmin':cdicts1['vmin'],
        'vmax':cdicts1['vmax'],
        'cmap':cdicts1['cmap'],
        'marker':'.',
        'zorder':10,
        's':1
    }
    
    xm = cdicts1['xm']
    ym = cdicts1['ym']
    depth = cdicts1['depth']
    ddst = cdicts1['ddst']
    
    err = cdicts1['err']

    ax = insert_stations_marker(ax, np.array(l)[:,-1], ylimdict[1]-1 , {})

    cf = ax.contourf(xm/1e3, ym, cdicts1['oa'], **kwcf)  # section
    
    if add_colorbar:
        cb = plt.colorbar(cf, ax=ax_cbar)
        cb.set_label(cdicts1['colorbar']['en'])

    cr = ax.contour(xm/1e3, ym, cdicts1['oa'], **kwcr)  # section
    ax.clabel(cr, fmt='%0.1f')

    # bathymetry mask
    ax.fill_between(np.array(ddst)/1e3,
                    -np.array(depth)+10, y2=10000,
                    color='0.5', zorder=2)

    ax.set_ylim([100,0])  # set ylims
    ax.set_ylabel('Depth [m]')

    cr = ax.contourf(xm/1e3, ym, err, [0.001, 1], hatches=['///'],
                    colors='None', zorder=1)
    colors = ['red']
    for i, collection in enumerate(cr.collections):
        collection.set_edgecolor(colors[i % len(colors)])
        collection.set_linewidths(0.5)

    return fig, ax, cf


def get_trend_correction_oa(df1, varb, corrlenx=None, corrleny=None, err_thresh=None):
    df2 = df1[[varb,'dist','station']].dropna()

    df2[varb + '_trend']   = df2[varb]*0
    df2[varb + '_detrend'] = df2[varb]*0
  
    f = lambda x: slice(x[0], x[1])
    ilim = np.where(df2[varb].index.diff()<0)
    ilim = [0] + list(ilim[0]) + [df2.shape[0]]
    ilim = list(zip(ilim, ilim[1:]))

    for i in ilim:
        aux = df2[varb].iloc[f(i)]
        
        coeff = np.polyfit(aux.index, aux.values, 1)
        trend = np.polyval(coeff, aux.index)
        
        df2[varb + '_trend'].values[f(i)] = trend
        df2[varb + '_detrend'].values[f(i)] = aux - trend

    xm, ym, oa ,erroa = objective_analysis_varb(df2,varb + '_detrend',
        corrlenx=corrlenx,
        corrleny=corrleny,
        err_thresh=err_thresh)
    trend = griddata((df2['dist'], df2.index), df2[varb + '_trend'], (xm,ym), method='linear')

    oa += trend

    return xm, ym, oa, erroa, df2