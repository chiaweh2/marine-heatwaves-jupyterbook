---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Marine Heatwaves forecasting using NMME (North American Multi-Model Ensemble)

## Predicting marine heatwaves
Using the NMME (North American Multi-Model Ensemble), [Jacox et al., 2022](http://doi.org/10.1038/s41586-022-04573-9) demonstrate the possibility of predicting the marine heatwaves under a monthly time scale with the lead time up to a year. 
The [marine heatwaves portal](https://psl.noaa.gov/marine-heatwaves/) forecast hosted at NOAA/PSL website is based on the calculation show in this notebook.

### Goals in the notes 
- [Lazy loading](lazyLoading) the NMME model data from [IRI/LDEO Climate Data Library](http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/)
- Calculate the ensemble mean climatology for each model based on hindcast
- Calculate the SST anomaly in the forecast
- Calculate the threshold based on the SST anomaly
- Calculate the marine heatwave in the forecast
- Show result

+++

```{note}
The following example is based on the paper [Jacox et al., 2022](http://doi.org/10.1038/s41586-022-04573-9). 
```

## Extract the data from the IRI/LDEO Climate Data Library OPeNDAP server
In this notebook, we demonstrate how to use the [NMME model](https://www.cpc.ncep.noaa.gov/products/NMME/) to generate the marine heatwaves prediction.
The dataset is currently hosted on [IRI/LDEO Climate Data Library](http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/).

```{tip}
The OPeNDAP server on the [IRI/LDEO Climate Data Library](http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/) also has data extraction limit. 
For solving some of the limit issue, there are some great discussion on this [GitHub issue](https://github.com/pangeo-data/pangeo/issues/767).
To summarized, user sometime will need to play a bit on the chunk size to find the optimal download scheme.
```



### Import needed python package
```{code-cell} ipython3
import warnings
import cftime
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
```

```{code-cell} ipython3
warnings.simplefilter("ignore")
```
```{tip}
This line of code is not affecting the execution but just removing some of the warning output that might clutter your notebook. 
However, do pay attention to some of the warning since it will indicate some deprecation of function and arg/kwarg for future use.
```

## Lazy loading the NMME model data from IRI/LDEO Climate Data Library
Like in the previous notebook, we request the meta data from the OPeNDAP server to quickly check the data structure.
There are four models that provide forecast till current and hindcast at the moment of generating this notes.
Here, we only request one model `GFDL-SPEAR` for demostration. 
However, for a better prediction, it is always better to have an ensemble of models with each model has its multiple runs. 

```{code-cell} ipython3
#### The opendap access
model_list = ['GFDL-SPEAR']
forecast_list = ['http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.%s/.FORECAST/.MONTHLY/.sst/dods'%model for model in model_list] 

hindcast_list = ['http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.%s/.HINDCAST/.MONTHLY/.sst/dods'%model for model in model_list] 
```
The code generate a list of URL to query with the OPeNDAP server for both hindcast and forecast.
With the URL generated, we can now send the query with `xr.open_dataset()`

```{code-cell} ipython3
# hindcast opendap
dict_model_hindcast = {}
for nmodel,link in enumerate(hindcast_list):
    dict_model_hindcast[model_list[nmodel]] = xr.open_dataset(link, engine='pydap',chunks={'M':1,'L':1,'S':1},decode_times=False)
```

```{code-cell} ipython3
# forecast opendap
dict_model_forecast = {}
for nmodel,link in enumerate(forecast_list):
    dict_model_forecast[model_list[nmodel]] = xr.open_dataset(link, engine='pydap',chunks={'M':1,'L':1,'S':1},decode_times=False)
```
Notice the `decode_times` is set to `False` due to the time unit is 'months since 1960-01-01' in the NMME dataset which is not following the [cf-convention](https://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate) of month and year should not be used in the time unit.
Therefore, `xr.open_dataset()` cannot decode the time coordinate correctly when reading the data.

```{important}
The `chunks` keyward argument in the `open_dataset` is the key to your processing speed and how one avoid the download limit of OPeNDAP server. 

The `engine` keyward argument is set to `'pydap'` to utilize the pydap backend to grab the data on a OPeNDAP server.
```
```{code-cell} ipython3
dict_model_hindcast['GFDL-SPEAR'].S
```
Without the decoding, time would show up as number of months (shown above) which is not easy to understand.

### Decoding the time axis manually
Since the initial time axis (coordinate = `S`) is not decoded, we need to decode it manually with the help of `cftime` package. 
```{code-cell} ipython3
for nmodel,model in enumerate(model_list):
    print(model)
    dict_model_hindcast[model]['S'] = cftime.num2date(
        dict_model_hindcast[model].S.values,
        'months since 1960-01-01',
        calendar='360_day'
        )
    dict_model_forecast[model]['S'] = cftime.num2date(
        dict_model_forecast[model].S.values,
        'months since 1960-01-01',
        calendar='360_day'
        )
```

```{code-cell} ipython3
dict_model_hindcast['GFDL-SPEAR'].S
```
Now after decoding, we can see the time in a more friendly format (shown above).

To quickly test if we are getting the right dataset, it is easy to generate a quick plot with the `.plot()` method which is using the `matplotlib` backend with `pcolormesh` method.
```{code-cell} ipython3
dict_model_hindcast['GFDL-SPEAR'].sst.isel(S=0,M=0,L=0).plot()
```
```{tip}
`pcolormesh` method generates map plot which reflects the true value at each grid point. `contour` or `contourf` methods, on the other hand, generate interpolated map plot. It is a personal choice and sometime depend on the purpose of the plots. However, `pcolormesh` method would be the best way to represent the actual value in the dataset and not the interpolated result.
```


## Calculate the ensemble mean climatology 
The climatology of ensemble mean is determined based on two steps.

- Calculate the ensemble mean of all ensemble members in each specific model (in our example `GFDL-SPEAR`)
- Calculate the monthly climatology from the ensemble mean.  

The calculation is based on the hindcast.
If there are more than one model, each model will have its own ensemble mean climatology. 
```{important}
For demostration in this notebook, the climatology period is only based on one year which does not make sense scientifically but this allow user to quickly play around the data.
For scientific purpose, the hindcast should at least has 30 years of data to calculate a more meaningful climatology.
```
```{code-cell} ipython3
# one year of data for climatology (demo purpose only)
start_year = 2020
end_year = 2020
```
The code loop throught different models (in this case `GFDL-SPEAR` only), download the desired period, calculate the ensemble mean, and calculate the climatology of the ensemble mean.
```{code-cell} ipython3
for nmodel,model in enumerate(model_list):
    print('-------------')
    print(model)
    print('-------------')

    # crop data to the desired period
    with ProgressBar():
        da_model = dict_model_hindcast[model].sst.where(
            (dict_model_hindcast[model]['S.year']>=start_year)&
            (dict_model_hindcast[model]['S.year']<=end_year),
            drop=True).compute()
    
    # calculate ensemble mean
    print('calculate ensemble mean')
    da_ensmean = da_model.mean(dim='M')

    # calculate ensemble mean climatology
    print('calculate climatology')
    da_ensmean_climo = da_ensmean.groupby('S.month').mean(dim='S')
```
```{tip}
Only till this point of the code with `.compute()`, the data is actually downloaded to the local memory and cropped to our desired time period. 
The `ProgressBar()` method provided by the `dask` package give us a quick way to monitor the progress of the data query in the notebook. 
``` 
To do a quick peak on the calculated `da_ensmean` and `da_ensmean_climo`
```{code-cell} ipython3
da_ensmean
```

```{code-cell} ipython3
da_ensmean_climo
```
We can see the `da_ensmean` has one dimension less than the original dataset due to the ensemble member dimension `M` is averaged. 
The `da_ensmean_climo`, on the other hand, is groupby the month of year.


## Calculate the threshold
The monthly threshold is based on the hindcast SST anomaly with a three month  window (initial months) for each initial time (centered month) and associated lead time.
```{admonition} Example:
The January (initial month) threshold of lead time = 0.5,1.5,2.5,... month is determined by all ensemble members with December, January, and Feburary SST anomaly at lead time = 0.5,1.5,2.5,... month, respectively. 
```
### Define functions 
The function below is performing the three month window on all initial month to find the corresponding monthly threshold at each lead time.
```{code-cell} ipython3
def nmme_3mon_quantile(da_data, mhw_threshold=90.):
    """
    This function is designed for calculating the nmme 
    marine heat wave threshold.
    
    The threshold is calculated using a 3 month window
    to identified the X quantile value for each month.

    Parameters
    ----------
    da_data : xr.DataArray 
        For marine heat wave calculation, it should be the sst DataArray. 
        Since this function is designed for NMME model. The DataArray need 
        to be in the format of NMME model output.
    mhw_threshold : float
        the X quantile one wants to calculate. 

    Returns
    -------
    da_data_quantile : xr.DataArray
        The xr.DataArray for the quantile 

    Raises
    ------

    """
    
    da_data_quantile = xr.DataArray(coords={'X':da_data.X,
                                            'Y':da_data.Y,
                                            'month':np.arange(1,13),
                                            'L':da_data.L},
                                    dims = ['month','L','Y','X'])

    for i in range(1,13):
        print(f"calculate month{i} threshold")
        if i == 1:
            mon_range = [12,1,2]
        elif i == 12 :
            mon_range = [11,12,1]
        else:
            mon_range = [i-1,i,i+1]

        da_data_quantile[i-1,:,:,:] = (da_data
                                 .where((da_data['S.month'] == mon_range[0])|
                                        (da_data['S.month'] == mon_range[1])|
                                        (da_data['S.month'] == mon_range[2]),drop=True)
                                 .stack(allens=('S','M'))
                                 .quantile(mhw_threshold*0.01, dim = 'allens', method='linear',skipna = True))

    return da_data_quantile
    


            
```
With the function defined, we just need to calculate the SST anomaly in hindcast.
There are different methods to determine the quantile. 
For `xarray.quantile` method (based on `numpy.quantile`), one can choose the desired method based on [NumPy documention](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#numpy.quantile). 
```{code-cell} ipython3
print('calculating anomaly from hindcast')
da_anom = (da_model.groupby('S.month') - da_ensmean_climo)

ds_mhw_threshold = xr.Dataset()

mhw_threshold = [90]
for m in mhw_threshold:

    print(f'calculating threshold {m}')
    da_threshold = nmme_3mon_quantile(da_anom, mhw_threshold=m)
    ds_mhw_threshold[f'threshold_{m}'] = da_threshold
```
```{warning}
The function `nmme_3mon_quantile` is the most time consuming part in the whole calculation. 
It should have a more optimal way to speed up the process. 
However, this is currently (at the time of creating this notebook) our best solution.
```
The resulting threshold structure would look like below
```{code-cell} ipython3
da_threshold
```


## Calculate the anomaly of forecast
+++
### Load one initial time of forecast anomaly data for demo
```{code-cell} ipython3
for nmodel,model in enumerate(model_list):
    print('-------------')
    print(model)
    print('-------------')

    # crop data to the desired period
    with ProgressBar():
        da_model_forecast = dict_model_forecast[model].sst.sel(S="2022-01").compute()
        
print('calculating anomaly from forecast')
da_model_forecast = da_model_forecast.groupby('S.month') - da_ensmean_climo
```
Here, we only pick one initial time to download. 
For each initial time (`S`), there are 11 lead times (`L`) and 30 ensemble members (`M`) in NMME. 
The SST anomaly of forecast is determined by subtracting the ensemble mean climatoloty that we calculated based on hindcast. 


## Calculate the marine heatwave 
Using the forecast SST anomaly and the threshold determined based on hindcast, we can identify the marine heatwave in the forecast.

```{code-cell} ipython3
ds_mhw = xr.Dataset()

for m in mhw_threshold:

    print(f'calculating MHW {m}')
    da_mhw = da_model_forecast.where(da_model_forecast.groupby('S.month')>=ds_mhw_threshold[f'threshold_{m}'])
    ds_mhw[f'mhw_{m}'] = da_mhw
```
The result is stored in the `xarray.Dataset` with variable name `mhw_90`.
All grid points with SST anomaly higher than the correponsind threshold are original value preserved. 
The place that has value lower than the threshold will be masked with `NaN`.
```{code-cell} ipython3
da_mhw
```

## Show result
+++
For the following paragraph, we will start showing some map plot which is used to check if the calculation performed above make sense. 

### Define plot format function
```{tip}
When producing mutliple maps for comparison, we often time like to keep maps in the same format. 
To avoid repetition of format related code, it is good practice to define a function that handle the format. 
This way the same format will be applied to all plots that calls the function.
```
```{code-cell} ipython3
def plot_format(ax,font_size=10):
    ax.coastlines(resolution='110m',linewidths=0.8)
    ax.add_feature(cfeature.LAND,color='lightgrey')

    ax.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
    ax.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=font_size)
    ax.set_yticks([-90,-60,-30,0,30,60,90], crs=ccrs.PlateCarree())
    ax.set_yticklabels([-90,-60,-30,0,30,60,90], color='black', weight='bold',size=font_size)
    ax.yaxis.tick_left()

    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.grid(linewidth=2, color='black', alpha=0.1, linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('auto')
    
    return ax
```

### Show threshold, MHW, and SST anomaly

```{code-cell} ipython3
# comparing zos  
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig = plt.figure(2,figsize=(10,10))
fontsize = 10
level = np.linspace(-1, 1, 11)

# threshold figure
ax2 = fig.add_axes([0,0.5,0.5,0.20],projection=ccrs.PlateCarree(central_longitude=180))
im = (
    ds_mhw_threshold
        .isel(month=0,L=0).threshold_90
        .plot.pcolormesh(
            x='X',
            y='Y',
            ax=ax2, 
            levels=level, 
            extend='both', 
            cmap='RdBu_r',
            transform=ccrs.PlateCarree(central_longitude=0)
        )
)
ax2.set_title('Threshold @ S=Jan, L=0.5month', color='black', weight='bold',size=fontsize)
ax2 = plot_format(ax2,font_size=fontsize)


# mhw figure
ax3 = fig.add_axes([0,0.25,0.5,0.20],projection=ccrs.PlateCarree(central_longitude=180))
im = (
    ds_mhw
        .isel(S=0,L=0,M=0).mhw_90
        .plot.pcolormesh(
            x='X',
            y='Y',
            ax=ax3, 
            levels=level, 
            extend='both', 
            cmap='RdBu_r',
            transform=ccrs.PlateCarree(central_longitude=0)
        )
)
ax3.set_title('Forecast MHW @ S=Jan2022, L=0.5month', color='black', weight='bold',size=fontsize)
ax3 = plot_format(ax3,font_size=fontsize)

# anomaly figure
ax4 = fig.add_axes([0,0,0.5,0.20],projection=ccrs.PlateCarree(central_longitude=180))
im = (
    da_model_forecast
        .isel(S=0,L=0,M=0)
        .plot.pcolormesh(
            x='X',
            y='Y',
            ax=ax4, 
            levels=level, 
            extend='both', 
            cmap='RdBu_r',
            transform=ccrs.PlateCarree(central_longitude=0)
        )
)
ax4.set_title('Forecast Anomaly @ S=Jan2022, L=0.5month', color='black', weight='bold',size=fontsize)
ax4 = plot_format(ax4,font_size=fontsize)

```
The first map above shows the threshold for initial time = January and lead time = 0.5month. The second map shows the marine heatwave detected in the NMME SST forecast at initial time = Jan2022 and lead time = 0.5month. The third map shows the original NMME SST forecast at initial time = Jan2022 and lead time = 0.5month.

Comparison between first and second map shows that all indentified marine heatwave signal (second map) is all higher than the threshold (first map). 
Comparison between second and third map shows that all masked out region (second map) are values lower than the threshold (first map).



### calculate MHW event
After the quick senitiy check, we can  
- Assign Not a Number to masked our region (as no MHW)
- Assign 1 to regions with values (as MHW exist)
in `ds_mhw`

```{code-cell} ipython3
da_mhw_event = (ds_mhw.isel(S=0,L=11).mhw_90/ds_mhw.isel(S=0,L=11).mhw_90)
```
Here, we picked lead time = 11.5 months to demostrate the result of marine heatwave identification.

```{code-cell} ipython3
da_mhw_event.max()
```
As you can see from the result above that the maximum value is now `1` indicating the existence of marine heatwave. 


### Show all enesemble member identified MHW 
To show how different ensemble member can have very different result, we use the `.plot()` method without much formatting for quick view of the variation between members. 
```{note}
This is for specific start time (S = 2022 Jan) and lead time (11.5 months = 2022 Dec)
```
```{code-cell} ipython3
da_landmask = da_model_forecast.isel(L=0,M=0,S=0)/da_model_forecast.isel(L=0,M=0,S=0)
da_mhw_event.plot(x="X",y='Y',col='M',col_wrap=10,cmap='Greys',levels=np.arange(0,2+1))
```


### Show total identified MHW in ensemble and associated probability
By adding all the ensemble members result from above, we know at a specific start time (S = 2022 Jan), lead time (11.5 months = 2022 Dec), and grid point how many ensemble member has predicted the existence of marine heatwaves.
We subtract this total number of marine heatwave by the total number of ensemble member to produce the probability(%) of marine heatwaves in forecast.
```{important}
By doing the same calculation for different model which also have its own ensemble members, we can combined the total number of marine heatwaves from across models to get a probability forecast that consider the variations between different models.  
```

```{code-cell} ipython3

# comparing zos  
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig = plt.figure(2,figsize=(15,15))
fontsize = 10
num_level = np.arange(0, 30+1, 5)
level = np.arange(0, 1+0.1, 0.1)

# mhw event figure
ax2 = fig.add_axes([0,0.5,0.5,0.20],projection=ccrs.PlateCarree(central_longitude=180))
im = (
    (da_mhw_event.sum(dim='M')*da_landmask)
        .plot.pcolormesh(
            x='X',
            y='Y',
            ax=ax2, 
            levels=num_level, 
            extend='both', 
            cmap='viridis',
            transform=ccrs.PlateCarree(central_longitude=0)
        )
)
ax2.set_title('Number of MHW event in ensemble members @ S=Jan2022, L=11.5month', color='black', weight='bold',size=fontsize)
ax2 = plot_format(ax2,font_size=fontsize)

# mhw probability figure
ax3 = fig.add_axes([0,0.25,0.5,0.20],projection=ccrs.PlateCarree(central_longitude=180))
im = (
    (da_mhw_event.sum(dim='M')/da_mhw_event.M.max()*da_landmask)
        .plot.pcolormesh(
            x='X',
            y='Y',
            ax=ax3, 
            levels=level, 
            extend='both', 
            cmap='plasma_r',
            transform=ccrs.PlateCarree(central_longitude=0)
        )
)
ax3.set_title('Probability @ S=Jan2022, L=11.5month', color='black', weight='bold',size=fontsize)
ax3 = plot_format(ax3,font_size=fontsize)
```
The first map above shows the total number of marine heatwaves event in all 30 members of members from `GFDL-SPEAR` at initial time = January 2022 and lead time = 11.5month. The second map shows the probability of marine heatwave based on NMME SST forecast at initial time = January 2022 and lead time = 11.5month. The second map is determined by subtracting 30 on the first map.
