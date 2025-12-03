import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import gsw

# import sys
# sys.path.insert(0, './section-ctd_dalton/section-ctd/utils')

# from utils import gradient_thresh_index


def centered_diff(x, y, equally_spaced=False):
    nx = len(set(np.diff(x)))

    if (nx != 1) and (not equally_spaced):
        raise IOError('xnew must present equally spaced values')
    if len(x) != len(y):
        raise ValueError('x and y must present the same length')


    dx = np.diff(x)[0]

    f = np.zeros(y.size) * np.nan
    f[1:-1] = (y[2:] - y[:-2])/(2*dx)
    f[0] = (y[1]- y[0])/dx
    f[-1] = (y[-1]-y[-2])/dx
    return f


def centered_diff_2D(y, x):
    print('Attention this 2d derivative is applied along the first axis only')
    dx = np.gradient(x)[0]
    f = np.zeros(y.shape) * np.nan 
    f[1:-1] = (y[2:,:] - y[:-2,:])/(2*dx[1:-1])
    f[0] = (y[1,:]- y[0,:])/dx[0]
    f[-1] = (y[-1,:]-y[-2,:])/dx[-1]
    return f

def centered_diff_general(x, y,maxoffset=1):
    xi = np.arange(x.min(), x.max()+maxoffset)
    yi = np.interp(xi, x, y)
    gradient = centered_diff(xi, yi, equally_spaced=True)
    return xi, gradient


def gradient_thresh_index(x, y, scale, gradlim=0.05, rank=1, print=False):
    """Finds the first not nan index
    in a given matrix (along the first column),
    considering a scale factor
    """
    # - -vertical gradients (well behaved!) -- #
    xi, dydx = centered_diff_general(x, y)

    if rank == 2:
        xii, dydx = centered_diff_general(xi, dydx)
        xi = xii
    
    aux = np.zeros([dydx.shape[0]+2])
    xaux = aux.copy()
    aux[1:-1] = dydx
    xaux[1:-1] = xi
    xaux[0] = xi[0]
    xaux[-1] = xi[-1]

    # scale factors and threshold
    i0 = aux  > gradlim*scale
    i1 = aux  < -gradlim*scale
    
    # create a matrix of indexes in the vertical direction
    jmatrix = np.arange(aux.shape[0]).astype(float)

    # select all values within scale factor
    jmatrix[(i1 | i0)] = np.nan

    inan = np.isnan(jmatrix)
    
    if inan.sum()>0:
        iaux = np.where(inan)[0]
        iout = iaux[1] if len(iaux)>1 else iaux[0]
    else:
        iout = -1
    xout = xaux[int(iout)]  # actual depth

    if print:
        fig, ax = plt.subplots(nrows=1, ncols=2,sharey=True)
        ax[0].plot(y, x, marker='x')
        ax[1].plot(aux[1:-1], xaux[1:-1], marker='x')
        ax[1].vlines(+ gradlim*scale, xaux[1], xaux[-2], colors='r', linewidth=0.5)
        ax[1].vlines(-gradlim*scale, xaux[1], xaux[-2], colors='r', linewidth=0.5)
        ax[1].vlines(0, xaux[1], xaux[-2], colors='r', linewidth=0.5)
        ax[0].hlines(xout, y.min(), y.max(), colors='r', linewidth=1)
        ax[1].hlines(xout, dydx.min(), dydx.max(), colors='r', linewidth=0.5)
        ax[0].grid(True, linestyle='--')
        ax[1].grid(True, linestyle='--')
        xlim = ax[0].get_xlim()
        ax[0].set_xlim(xlim[0]-0.5, xlim[1]+0.5)
        plt.title(f"mld: {round(xout)}", loc='right', fontsize=8)
        ax[0].invert_yaxis()
        return xout, fig, ax
    else:
        return xout, None, None

def compute_mld(x, sa, ct, rho):
    mld_sa, figsa, axsa    = gradient_thresh_index(x, sa, 0.00625, rank=2, print=False)
    mld_ct, figct, axct    = gradient_thresh_index(x, ct, 0.225, rank=2, print=False)
    mld_rho, figrho, axrho = gradient_thresh_index(x, rho,0.025, rank=2, print=False)
    mld = np.array([mld_sa, mld_ct, mld_rho])
    mld = mld.max()
    
    return mld

def compute_mld_density(x, rho, threshold=0.03):
    mld_rho, figrho, axrho = gradient_thresh_index(x, rho, threshold, rank=2, print=False)
    
    return mld_rho


def load_glider(datadir_glider):
    ds = xr.open_dataset(f"{datadir_glider}/sg576_Azul_19")
    
    ds['pressure'] = gsw.p_from_z((-1)*ds['depth'], ds['latitude'])
    ds['SA'] = gsw.SA_from_SP(ds['salinity'], ds['pressure'], ds['longitude'], ds['latitude'])
    ds['CT'] = gsw.CT_from_t(ds['SA'], ds['temperature'], ds['pressure'])
    ds['dens'] = gsw.sigma0(ds['SA'], ds['CT'])

    # fixing attributes
    ds['SA'].attrs['long_name'] = 'absolute salinity'
    ds['SA'].attrs['standard_name'] = 'salt'
    ds['SA'].attrs['units'] = 'g/kg'

    ds['CT'].attrs['long_name'] = 'temperature'
    ds['CT'].attrs['standard_name'] = 'temp'
    ds['CT'].attrs['units'] = 'oC'
    
    return ds

def cast_process(profile):

    cast = profile.to_dataframe()

    lon,lat,time = cast['LONGITUDE'].values[0], cast['LATITUDE'].values[0],  cast['TIME'].values[0]

    # renaming        
    p = cast['PRES']

    cast['depth'] = gsw.z_from_p(cast["PRES"].values, cast["LATITUDE"].values) * (-1)

    # compute hydrographic data
    ### NOTE: eu ja converti quando salvei o arquivo netcdf com os perfis
    SA = cast['PSAL'] #gsw.SA_from_SP(cast['PSAL'], cast["PRES"].values, 
         #               cast["LONGITUDE"].values, cast["LATITUDE"].values)
    CT = cast['TEMP'] # gsw.CT_from_t(SA, cast['TEMP'].values, cast["PRES"].values)

    dens = gsw.sigma0(SA, CT)

    cast["SP"] = cast["PSAL"]
    cast['salt'] = SA
    cast['temp'] = CT
    cast['dens'] = dens
    
    cast = cast.to_xarray()
    cast.attrs = profile.attrs

    return cast

# standardizing datasets
def preparing_dataset(prof, dataset, capdepth=None):
    """
       prof:: xarray.Dataset
    dataset:: string (argo, ctd, glider)
    capdeth:: integer (set a value to select 
    """

    if dataset == 'argo':
        # ARGO
        y = prof.depth.values * (-1)
        indy = np.where(y >= -200)[0]
        
        salt = prof.PSAL.values[indy]
        temp = prof.TEMP.values[indy]
        dens = prof.dens.values[indy]
        dept = y[indy]
        
        # metadata
        time = datetime.strptime(str(prof['TIME'].values[0]).split('.')[0], "%Y-%m-%dT%H:%M:%S")
        xlon = prof['lon'].values[0]
        ylat = prof['lat'].values[0]
        
    elif dataset == 'ctd':
        # CTD
        df = prof.to_dataframe()
        y = df.interpolate()['depth'].values * (-1)
        indy = np.where(y >= -200)[0]
        salt = df.interpolate()['sa'].values[indy]
        temp = df.interpolate()['ct'].values[indy]
        dens = df.interpolate()['rho'].values[indy] - 1000
        dept = y[indy]

        # metadata
        time = datetime.strptime(prof.attrs['time'], "%Y-%m-%d %H:%M:%S")
        xlon = prof.attrs['lon']
        ylat = prof.attrs['lat']
        
        
    elif dataset == 'ctd_lado':
        # CTD
        y = prof['depth'].values * (-1)
        indy = np.where(y >= -200)[0]
        salt = prof['sa'].values[indy]
        temp = prof['ct'].values[indy]
        dens = prof['psigma0'].values[indy]
        dept = y[indy]

        # metadata
        time = datetime.strptime(prof.attrs['time'], "%Y-%m-%d")
        xlon = prof.attrs['lon']
        ylat = prof.attrs['lat']
    
    elif dataset == 'glider':
        # GLIDER
        df = prof[['depth', 'SA', 'CT', 'dens', 'time']].to_dataframe()
        salt = df['SA'].values
        temp = df['CT'].values
        dens = df['dens'].values
        dept = df.index.values * (-1)

        time = datetime.strptime(str(prof['time'].values).split('.')[0], "%Y-%m-%dT%H:%M:%S")
        xlon = prof['lon'].values.item()
        ylat = prof['lat'].values.item()
   
    # computing MLD
    # mld = compute_mld(dept, salt, temp, dens)
    mld = compute_mld_density(dept, dens)

    # build Dataset with hydrographic variables
    dsprof = xr.Dataset(data_vars=dict(
                            temp=(["y"], temp),
                            salt=(["y"], salt),
                            dens=(["y"], dens),
                            dept=(["y"], dept),
                            ),
                       coords={"y": np.arange(0, dept.shape[0])}, # arbitrary coordinate
                       attrs=dict(time=time, lon=xlon, lat=ylat, mld=mld)
                  )
    
    return dsprof

def load_vertical_profiles(dict_casts=False, 
                          datadir_ctd='/home/danilo/phd/thesis/chapter_colab/scripts/section-ctd_dalton/data/umassd2/treated',
                          datadir_argo='/home/danilo/phd/thesis/chapter_colab/data/argo',
                          datadir_glider='/home/danilo/phd/thesis/chapter_colab/data/glider'
                          ):
    if not dict_casts:
        dict_casts = {
                'cast1': {
                    "dataset": "argo",
                    "fname": f"{datadir_argo}/profiles_all_3901549.nc",
                    "selected": 4,
                    "rename": {'LONGITUDE': 'lon', 'LATITUDE': 'lat'},
                    'fixS': None,
                    'fixT': None,
                },
                'cast2': {
                    "dataset": "argo",
                    "fname": f"{datadir_argo}/profiles_all_3901225.nc",
                    "selected": 7,
                    "rename": {'LONGITUDE': 'lon', 'LATITUDE': 'lat'},
                    'fixS': None,
                    'fixT': None,
                },
                # 'cast3': {
                #     "dataset": "ctd",
                #     "fname": f"{datadir_ctd}/SANAGU14_G7_CTD_1_OT.nc",
                #     "selected": 'SANAGU14_G7_CTD_1_OT',
                #     "rename": {'depSM': 'depth'},
                #     'fixS': None,
                #     'fixT': None,
                # },
                'cast3.2': {
                    "dataset": "ctd_lado",
                    "fname": f"{datadir_ctd}/SANAGU14_G7_CTD_1_OT_testenew.nc",
                    "selected": 'SANAGU14_G7_CTD_1_OT_testenew', # SANAGU14_G7_CTD_1_OTtotable
                    "rename": {'depSM': 'depth'},
                    'fixS': None, #1.035,
                    'fixT': None,
                },
                # 'cast4': {
                #     "dataset": "glider",
                #     "fname": datadir_glider,
                #     "selected": '2022-03-12 17:33',
                #     "rename": {'longitude': 'lon', 'latitude': 'lat', 'depth': 'depth'}
                # },
                # 'cast5': {
                #     "dataset": "glider",
                #     "fname": datadir_glider,
                #     "selected": '2022-03-18 13:24',
                #     "rename": {'longitude': 'lon', 'latitude': 'lat', 'depth': 'depth'}
                # },
                'cast6': {
                    "dataset": "ctd",
                    "fname": f"{datadir_ctd}/SANAGU15_G4_CTD_2_OT.nc",
                    "selected": 'SANAGU15_G4_CTD_2_OT',
                    "rename": {'depSM': 'depth'},
                    'fixS': None,
                    'fixT': None,
                },
                'cast7': {
                    "dataset": "ctd",
                    "fname": f"{datadir_ctd}/SANAGU15_G3_CTD_2_OT.nc",
                    "selected": 'SANAGU15_G3_CTD_2_OT',
                    "rename": {'depSM': 'depth'},
                    'fixS': None,
                    'fixT': None,
                },
                # 'cast7.2': {
                #     "dataset": "ctd_lado",
                #     "fname": f"{datadir_ctd}/SANAGU15_G3_CTD_2_OTtotable.nc",
                #     "selected": 'SANAGU15_G3_CTD_2_OTtotable',
                #     "rename": {'depSM': 'depth'},
                #     'fixS': None,
                #     'fixT': None,
                # },
            }


    for cast,values in dict_casts.items():
        print(f"Processing {cast}...")
        if values['dataset'] == 'argo':
            ds1 = xr.open_dataset(values['fname'])
            prof = ds1.isel(N_PROF=values['selected'])
            prof = cast_process(prof)
            prof = prof.rename(values['rename'])
            time = datetime.strptime(str(prof['TIME'].values[0]).split('.')[0], "%Y-%m-%dT%H:%M:%S")
            xlon = prof['lon'].values[0]
            ylat = prof['lat'].values[0]

        elif values['dataset'] == 'ctd':
            prof = xr.open_dataset(values['fname'])
            prof = prof.rename(values['rename'])
            # renaming attributes before extracting them
            if 'Data_estacao' in prof.attrs.keys():
                prof.attrs['time'] = prof.attrs['Data_estacao']

            time = datetime.strptime(prof.attrs['time'], "%Y-%m-%d %H:%M:%S")
            xlon = prof.attrs['lon']
            ylat = prof.attrs['lat']
            
        elif values['dataset'] == 'ctd_lado':
            prof = xr.open_dataset(values['fname'])
            prof = prof.rename(values['rename'])
            # renaming attributes before extracting them
            if 'Data_estacao' in prof.attrs.keys():
                prof.attrs['time'] = prof.attrs['Data_estacao']
                prof.attrs['lon'] = prof['lon'].values[0]
                prof.attrs['lat'] = prof['lat'].values[0]

            time = datetime.strptime(prof.attrs['time'], "%Y-%m-%d")
            xlon = prof.attrs['lon']
            ylat = prof.attrs['lat']

        elif values['dataset'] == 'glider':
            ds = load_glider(datadir_glider)
            prof = ds.sel(time=values['selected'], method='nearest')
            prof = prof.rename(values['rename'])
            prof = prof.squeeze()        
            time = datetime.strptime(str(prof['time'].values).split('.')[0], "%Y-%m-%dT%H:%M:%S")
            xlon = prof['lon'].values.item()
            ylat = prof['lat'].values.item()
               
        # saving metadata and data for future plots
        dict_casts[cast]['prof'] = preparing_dataset(prof, dict_casts[cast]['dataset']) # standardize vertical profiles
               
        dict_casts[cast]['time'] = time
        dict_casts[cast]['xlon'] = xlon
        dict_casts[cast]['ylat'] = ylat

        if values['fixS']:
            dict_casts[cast]['prof']['salt'] = dict_casts[cast]['prof']['salt'] + values['fixS']
            
        if values['fixT']:
            dict_casts[cast]['prof']['temp'] = dict_casts[cast]['prof']['temp'] + values['fixT']
            
    return dict_casts