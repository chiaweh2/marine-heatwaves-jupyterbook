---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: python
  language: python
  name: python
---

# Detecting marine heatwave based on observation

## What is marine heatwaves
The marine heatwaves are anomalous warm water over the ocean. 
To detect the warm ocean water, sea surface temperature (SST) is usually used to define if there is any marine heatwave event.

## Observational data
In this notebook, we demonstrate how to use the [NOAA OISST v2 High resolution dataset](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html) to show the observed marine heatwaves.
The dataset is currently hosted on [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html).

````{margin}
```{tip}
To explore more gridded datasets that are hosted at NOAA PSL, here is a useful [search tool](https://psl.noaa.gov/data/gridded/index.html)
```
````

```{note}
The following example is following the paper [Jacox et al., 2022](http://doi.org/10.1038/s41586-022-04573-9). 
```

## Python modules
The module imported to the python kernel is shown below
```{code-cell} ipython3
import warnings
import datetime
import intake
import xarray as xr
import numpy as np
```
But for the binder/jupyternotebook to work, these are the list of required packages
```{note}
- numpy
- xarray
- dask
- netcdf4
- h5netcdf
- pydap
- scipy
- matplotlib
- intake
- intake-thredds
```
Some of the packages are used in the backend of the imported packaged shown above.


## Utilize intake for thredds
The thredds catalog can be viewed with the intake and intake-thredds module. 
This is not actually needed for downloading the data since PSL data server also provide the [OPeNDAP](https://www.earthdata.nasa.gov/engage/open-data-services-and-software/api/opendap) approach. 
Here, we are just demostrating a great way to generate a list of file names that can be seen in the thredds catalog.
```{code-cell} ipython3
cat_url = 'https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.xml'
catalog = intake.open_thredds_cat(cat_url, name='noaa-oisst')
catalog
```

```{code-cell} ipython3
list(catalog)[-10:]
```
````{margin}
```{tip}
For more information, [intake-thredds readthedoc](https://intake-thredds.readthedocs.io/en/latest/tutorials/index.html) shows how to access data using dask from intake
```
````

## Lazy loading the dataset through OPeNDAP
With the power of [Xarray](https://docs.xarray.dev/en/stable/) and [Dask](https://www.dask.org), we are able to load the data lazily (only loading the meta data and coordinates information) and peek at data's dimension and availability on our local machine.
The actual data (SST values at each grid point in this case) will only be downloaded from the PSL server when further data manipulation (subsetting and aggregation like calculating mean) is needed.
The lazy loading approach provides the oppurtunity to reduce the memory usage in the calculation, the possibility of parallizing the processes, and side stepping the data download limit set by OPeNDAP server (PSL server has a 500MB limit).
```{code-cell} ipython3
opendap_mon_url = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"

ds_mon = xr.open_dataset(opendap_mon_url, engine='pydap', chunks={'time':12,'lon':-1,'lat':-1})
```
```{important}
The `chunks` keyward argument in the `open_dataset` is the key to your processing speed and how one avoid the download limit of OPeNDAP server. 

The `engine` keyward argument is set to `'pydap'` to utilize the pydap backend to grab the data on a OPeNDAP server.
```
### What is chunks?
Dask has a great [documentation](https://docs.dask.org/en/latest/array-chunks.html) of chunks. The basic idea is that a single netCDF file can be seperated into multiple chunks (e.g. 20(lon)x20(lat) into four chunks of 10(lon)x10(lat)). 
By reading each chunk at a time as needed, we do not have to load the entire dataset into the memory.
```{code-cell} ipython3
ds_mon
```
In our example, we set the size of each chunk to be 12(time)x1440(lon)x720(lat) which is equal to 47.46 MB of data while the entire dataset is 1.39 GB. 
This allow us to get data in 47.46 MB chunk per download request.

## Climatology


```{important}
For a more accurate and scientificly valid estimate of marine heatwaves, one should usually consider a climatology period of at least 30 years.
Here we set the climatology period from 2010 to 2020 (10years) to speed up the processing time and for demostration only. 
The shorter period (smaller memory consumption) also make the binder launch on this page available for user to manipulate and play with the dataset. 
```
```{code-cell} ipython3
climo_start_yr = 2010             # determine the climatology/linear trend start year
climo_end_yr = 2020               # determine the climatology/linear trend end year

ds_mon_crop = ds_mon.where((ds_mon['time.year']>=climo_start_yr)&
                           (ds_mon['time.year']<=climo_end_yr),drop=True)
```

```{code-cell} ipython3
ds_mon_crop
```

```{code-cell} ipython3
ds_mon_climo = ds_mon_crop.groupby('time.month').mean()
ds_mon_anom = (ds_mon_crop.groupby('time.month')-ds_mon_climo).compute()
```

```{code-cell} ipython3
ds_mon_anom.sst
```

```{code-cell} ipython3
########## Functions ######### 
# Function to calculate the 3 month rolling Quantile
def mj_3mon_quantile(da_data, mhw_threshold=90.):
    
    da_data_quantile = xr.DataArray(coords={'lon':da_data.lon,
                                            'lat':da_data.lat,
                                            'month':np.arange(1,13)},
                                    dims = ['month','lat','lon'])

    for i in range(1,13):
        if i == 1:
            mon_range = [12,1,2]
        elif i == 12 :
            mon_range = [11,12,1]
        else:
            mon_range = [i-1,i,i+1]

        da_data_quantile[i-1,:,:] = (da_data
                                 .where((da_data['time.month'] == mon_range[0])|
                                        (da_data['time.month'] == mon_range[1])|
                                        (da_data['time.month'] == mon_range[2]),drop=True)
                                 .quantile(mhw_threshold*0.01, dim = 'time', skipna = True))

    return da_data_quantile
```

```{code-cell} ipython3
%time da_mon_quantile = mj_3mon_quantile(ds_mon_anom.sst, mhw_threshold=90)
```

```{code-cell} ipython3
da_mon_quantile.plot(col='month',vmin=0,vmax=3)
```

```{code-cell} ipython3
ds_mon_anom.sst.isel(time=slice(0,12)).plot(col='time',vmin=0,vmax=3)
```

```{code-cell} ipython3
da_mhw = ds_mon_anom.sst.where(ds_mon_anom.sst.groupby('time.month')>da_mon_quantile)
```

```{code-cell} ipython3
da_mhw.isel(time=slice(0,12)).plot(col='time',vmin=0,vmax=3)
```
