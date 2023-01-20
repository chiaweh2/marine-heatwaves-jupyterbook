#!/usr/bin/env python
# coding: utf-8

# # How to detect marine heatwave based on observational dataset
# 
# The marine heatwaves are anomalous warm water over the ocean. 
# To detect the warm ocean water, sea surface temperature (SST) is usually used to define if there is any marine heatwave event. 
# 
# The following example is following the paper [Jacox et al., 2022](http://doi.org/10.1038/s41586-022-04573-9)

# In[1]:


import warnings
import datetime
import intake
import xarray as xr
import numpy as np
from dask.distributed import Client


# In[2]:


warnings.simplefilter("ignore")
client = Client(processes=False)
client


# In[3]:


cat_url = 'https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.xml'
catalog = intake.open_thredds_cat(cat_url, name='noaa-oisst')
catalog


# In[4]:


list(catalog)[-10:]


# In[5]:


opendap_mon_url = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"


# In[6]:


ds_mon = xr.open_dataset(opendap_mon_url, chunks={'time':12,'lon':-1,'lat':-1})


# In[7]:


climo_start_yr = 1991             # determine the climatology/linear trend start year
climo_end_yr = 2020               # determine the climatology/linear trend end year

ds_mon_crop = ds_mon.where((ds_mon['time.year']>=climo_start_yr)&
                           (ds_mon['time.year']<=climo_end_yr),drop=True)


# In[8]:


ds_mon_crop


# In[9]:


ds_mon_climo = ds_mon_crop.groupby('time.month').mean()
ds_mon_anom = (ds_mon_crop.groupby('time.month')-ds_mon_climo).compute()


# In[10]:


ds_mon_anom.sst


# In[11]:


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


# In[12]:


get_ipython().run_line_magic('time', 'da_mon_quantile = mj_3mon_quantile(ds_mon_anom.sst, mhw_threshold=90)')


# In[13]:


da_mon_quantile.plot(col='month',vmin=0,vmax=3)


# In[14]:


ds_mon_anom.sst.isel(time=slice(0,12)).plot(col='time',vmin=0,vmax=3)


# In[15]:


da_mhw = ds_mon_anom.sst.where(ds_mon_anom.sst.groupby('time.month')>da_mon_quantile)


# In[16]:


da_mhw.isel(time=slice(0,12)).plot(col='time',vmin=0,vmax=3)


# In[ ]:




