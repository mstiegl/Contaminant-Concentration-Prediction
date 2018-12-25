
# coding: utf-8

# In[28]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2


# In[3]:


import pandas as pd
# beijing polution data
aq_17_18 = pd.read_csv('airQuality_201701-201801.csv')
aq_1802_1803 = pd.read_csv('airQuality_201802-201803.csv')
aq_1804 = pd.read_csv('aiqQuality_201804.csv')
# beijing observed weather data
ow_1701_1801 = pd.read_csv('observedWeather_201701-201801.csv')
ow_1802_1803 = pd.read_csv('observedWeather_201802-201803.csv')
ow_1804 = pd.read_csv('observedWeather_201804.csv')
ow_1804 = pd.read_csv('observedWeather_20180501-20180502.csv')
# beijing grid weather data
gw_1701_1803 = pd.read_csv('gridWeather_201701-201803.csv')
gw_1804 = pd.read_csv('gridWeather_201804.csv')
gw_180501_180502 = pd.read_csv('gridWeather_20180501-20180502.csv')


# In[17]:


ow_1701_1801.loc[0,'wind_speed']


# In[18]:


for i in range(len(ow_1701_1801)):
    ow_1701_1801.loc[i,'wind_speed'] = ow_1701_1801.loc[i,'wind_speed']*3.6


# In[19]:


ow_1701_1801.to_csv('observedWeather_201701-201801.csv')


# In[20]:


aq_17_18.describe()


# In[21]:


ow_1701_1801.head(5)


# In[22]:


longitude = list(ow_1701_1801['longitude'])
latitude = list(ow_1701_1801['latitude'])
utc_time = list(ow_1701_1801['utc_time'])
wind_direction = list(ow_1701_1801['wind_direction'])
wind_speed = list(ow_1701_1801['wind_speed'])
temperature = list(ow_1701_1801['temperature'])
pressure = list(ow_1701_1801['pressure'])
humidity = list(ow_1701_1801['humidity'])

print(ow_1701_1801.shape)
ow_1701_1801.describe()


# In[23]:


len(gw_1701_1803)


# In[24]:


# complete the observation station data by using grid weather data
from tqdm import tqdm 

gw = {}

g_longitude = list(gw_1701_1803['longitude'])
g_latitude = list(gw_1701_1803['latitude'])
g_utc_time = list(gw_1701_1803['utc_time'])
g_temperature = list(gw_1701_1803['temperature'])
g_pressure = list(gw_1701_1803['pressure'])
g_humidity = list(gw_1701_1803['humidity'])
g_wind_drirection = list(gw_1701_1803['wind_direction'])
g_wind_speed = list(gw_1701_1803['wind_speed/kph'])

for i in tqdm(range(len(gw_1701_1803))):
    t_key = str(g_longitude[i]) + str(g_latitude[i]) + str(g_utc_time[i])
    gw.setdefault(t_key,[]).append(g_temperature[i])
    gw.setdefault(t_key,[]).append(g_pressure[i])
    gw.setdefault(t_key,[]).append(g_humidity[i])
    gw.setdefault(t_key,[]).append(g_wind_drirection[i])
    gw.setdefault(t_key,[]).append(g_wind_speed[i])


# In[25]:


import numpy as np

for i in tqdm(range(len(temperature))):
    if np.isnan(wind_direction[i]) or wind_direction[i] > 360:
        t_key = str(round(longitude[i],1)) + str(round(latitude[i],1)) + str(utc_time[i])
        wind_direction[i] = gw[t_key][3]
    if np.isnan(wind_speed[i]) or wind_speed[i] > 100:
        t_key = str(round(longitude[i],1)) + str(round(latitude[i],1)) + str(utc_time[i])
        wind_speed[i] = gw[t_key][4]
    if temperature[i] > 50:
        t_key = str(round(longitude[i],1)) + str(round(latitude[i],1)) + str(utc_time[i])
        temperature[i] = gw[t_key][0]
    if pressure[i] == 999999:
        t_key = str(round(longitude[i],1)) + str(round(latitude[i],1)) + str(utc_time[i])
        pressure[i] = gw[t_key][1]
    if humidity[i] == 999999:
        t_key = str(round(longitude[i],1)) + str(round(latitude[i],1)) + str(utc_time[i])
        humidity[i] = gw[t_key][2]


# In[26]:


ow = []
ow.append(list(ow_1701_1801['station_id']))
index = [longitude,latitude,utc_time,temperature,pressure,humidity,wind_direction,wind_speed]
for each in index:
    ow.append(each)
ow.append(list(ow_1701_1801['weather']))

c_index = ['station_id','longitude','latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather']
ow_1701_1801_full = pd.DataFrame(np.transpose(ow) ,index = None, columns = c_index)

ow_1701_1801_full.to_csv('ow_1701_1801_full.csv')

ow_1701_1801_full.head()


# In[29]:


# Generate meteorological data of air condition stations
aq_position = pd.read_csv('air_station_position.csv')

aq_longitude = list(aq_position['longitude'])
aq_latitude = list(aq_position['latitude'])
aq_station_id = list(aq_position['stationId'])

# Group by coordinates
df1 = pd.read_csv('gridWeather_201701-201803.csv')
bj_meo_gen = []
bj_meo = df1.groupby(["longitude","latitude"])
for i in tqdm(range(35)):
    bj_meo_gen.append(bj_meo.get_group((round(aq_longitude[i],1),round(aq_latitude[i],1))).drop(['stationName'],axis=1))


# In[30]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2\air condition weather data


# In[31]:


# Generate meteorological data of observation points according to grid data corresponding to coordinates
# path = 'D:\HKUST\data mining and knowledge discovery\Project\work2\air condition weather data'
for i in range(35):
    bj_meo_gen[i].to_csv(str(aq_station_id[i].split('_aq')[0]) + '.csv')


# In[32]:


def get_missing_data_table(data):
    total = data.isnull().sum()
    percentage = data.isnull().sum() / data.isnull().count()
    missing_data = pd.concat([total,percentage],axis = 'columns',keys = ['TOTAL','PERCENTAGE'])
    return missing_data.sort_index(ascending = True)


# In[33]:


get_missing_data_table(aq_17_18)

