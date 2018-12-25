
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import statistics
from scipy import stats 


# In[3]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2\feature data


# In[4]:


# wind_direction = list(dongsi['wind_direction'])
# PM25 = list(dongsi['PM2.5']) 
# PM10 = list(dongsi['PM10'])
# O3 = list(dongsi['O3'])
# temperature = list(dongsi['temperature'])
# pressure = list(dongsi['pressure'])
# humidity = list(dongsi['humidity'])
# wind_speed = list(dongsi['wind_speed/kph'])


# In[5]:


def get_wind_direction(file):
    wind_direction = list(file['wind_direction'])
    for i in range(len(wind_direction)):
        if (wind_direction[i]>=337.5 and wind_direction[i]<=360) or (wind_direction[i]<22.5):
            wind_direction[i]=1
        elif wind_direction[i]>=22.5 and wind_direction[i]< 67.5:
            wind_direction[i]=2
        elif wind_direction[i]>=67.5 and wind_direction[i]< 112.5:
            wind_direction[i]=3
        elif wind_direction[i]>=112.5 and wind_direction[i]< 157.5:
            wind_direction[i]=4
        elif wind_direction[i]>=157.5 and wind_direction[i]< 202.5:
            wind_direction[i]=5
        elif wind_direction[i]>=202.5 and wind_direction[i]< 247.5:
            wind_direction[i]=6
        elif wind_direction[i]>=247.5 and wind_direction[i]< 292.5:
            wind_direction[i]=7
        elif wind_direction[i]>=292.5 and wind_direction[i]< 337.5:
            wind_direction[i]=8
        else:
            wind_direction[i]=0
    wind_direction = list(file['wind_direction'])
    for i in range(7,len(wind_direction)):
        temp_list = []
        for j in range(8):
            temp_list.append(wind_direction[i-7+j])
        wind_direction[i] = stats.mode(temp_list)[0][0]
    file['wind_direction'] = wind_direction
    return


# In[6]:


# dongsi['wind_direction'] = wind_direction


# In[7]:


def get_temperature(file):
    temperature = list(file['temperature'])
    for i in range(7,len(temperature)):
        temp_list = []
        for j in range(8):
            temp_list.append(temperature[i-7+j])
        temperature[i] = statistics.mean(temp_list)
    file['temperature'] = temperature
    return
    
def get_pressure(file):
    pressure = list(file['pressure'])
    for i in range(7,len(pressure)):
        temp_list = []
        for j in range(8):
            temp_list.append(pressure[i-7+j])
        pressure[i] = statistics.mean(temp_list)
    file['pressure'] = pressure
    return

def get_humidity(file):
    humidity = list(file['humidity'])
    for i in range(7,len(humidity)):
        temp_list = []
        for j in range(8):
            temp_list.append(humidity[i-7+j])
        humidity[i] = statistics.mean(temp_list)
    file['humidity'] = humidity
    return
    
def get_wind_speed(file):
    wind_speed = list(file['wind_speed/kph'])
    for i in range(7,len(wind_speed)):
        temp_list = []
        for j in range(8):
            temp_list.append(wind_speed[i-7+j])
        wind_speed[i] = statistics.mean(temp_list)
    file['wind_speed'] = wind_speed
    return

def wind_direction(file):
    wind_direction = list(file['wind_direction'])
    for i in range(7,len(wind_direction)):
        temp_list = []
        for j in range(8):
            temp_list.append(wind_direction[i-7+j])
        wind_direction[i] = stats.mode(temp_list)
    file['wind_direction'] = wind_direction
    return


# In[8]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2


# In[9]:


air_station_position = pd.read_csv('air_station_position.csv')
aq_index = list(air_station_position['stationId'])


# In[10]:


for i in range(len(aq_index)):
    aq_index[i] = aq_index[i].split('_')[0]


# In[11]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2\feature data


# In[12]:


def get_weather_feature():
    for aq in tqdm(aq_index):
        file = pd.read_csv('feature_' + aq + '.csv')
        wind_direction = list(file['wind_direction'])
#         PM25 = list(dongsi['PM2.5']) 
#         PM10 = list(dongsi['PM10'])
#         O3 = list(dongsi['O3'])
        temperature = list(file['temperature'])
        pressure = list(file['pressure'])
        humidity = list(file['humidity'])
        wind_speed = list(file['wind_speed/kph'])
        wind_direction = list(file['wind_direction'])
        get_temperature(file)
        get_pressure(file)
        get_humidity(file)
        get_wind_speed(file)
        get_wind_direction(file)
        file.to_csv('D:/HKUST/data mining and knowledge discovery/Project/work2/feature data2/' + 'feature_' + aq +'.csv')


# In[14]:


get_weather_feature()


# In[13]:


def get_PM25(file):
    PM25lasthour = list(file['PM25lasthour'])
    PM25 = list(file['PM25'])
    for i in range(1,len(file['PM2.5'])):
        PM25lasthour[i] = PM25[i-1]
    file['PM25lasthour'] = PMlasthour
    return


# In[14]:


def get_PM10(file):
    PM10lasthour = list(file['PM10lasthour'])
    PM10 = list(file['PM10'])
    for i in range(1,len(file['PM10'])):
        PM10lasthour[i] = PM10[i-1]
    file['PM10lasthour'] = PM10lasthour
    return


# In[15]:


def get_O3(file):
    O3lasthour = list(file['O3'])
    O3 = list(file['O3'])
    for i in range(1,len(file['O3'])):
        O3lasthour[i] = O3[i-1]
    file['O3'] = O3lasthour
    return


# In[16]:


def get_PM25_stats(file):
    PM25_temp = []
    PM25_min = list(file['PM25_min'])
    PM25_max = list(file['PM25_max'])
    PM25_max_min = list(file['PM25_max_min'])
    PM25_mean = list(file['PM25_mean'])
    PM25_median = list(file['PM25_median'])
    PM25_std = list(file['PM25_std'])
    for i in range(48,len(PM25_min)):
        for j in range(48):
            PM25_temp.append(file['PM2.5'][i-48+j])
        PM25_min[i] = min(PM25_temp)
        PM25_max[i] = max(PM25_temp)
        PM25_max_min[i] = max(PM25_temp) - min(PM25_temp)
        PM25_mean[i] = statistics.mean(PM25_temp)
        PM25_median[i] = statistics.median(PM25_temp)
        PM25_std[i] = statistics.stdev(PM25_temp)
    file['PM25_min'] = PM25_min
    file['PM25_max'] = PM25_max
    file['PM25_max_min'] = PM25_max_min
    file['PM25_mean'] = PM25_mean
    file['PM25_median'] = PM25_median
    file['PM25_std'] = PM25_std
    return


# In[17]:


def get_PM10_stats(file):
    PM10_temp = []
    PM10_min = list(file['PM10_min'])
    PM10_max = list(file['PM10_max'])
    PM10_max_min = list(file['PM10_max_min'])
    PM10_mean = list(file['PM10_mean'])
    PM10_median = list(file['PM10_median'])
    PM10_std = list(file['PM10_std'])
    for i in range(48,len(PM10_min)):
        for j in range(48):
            PM10_temp.append(file['PM10'][i-48+j])
        PM10_min[i] = min(PM10_temp)
        PM10_max[i] = max(PM10_temp)
        PM10_max_min[i] = max(PM10_temp) - min(PM10_temp)
        PM10_mean[i] = statistics.mean(PM10_temp)
        PM10_median[i] = statistics.median(PM10_temp)
        PM10_std[i] = statistics.stdev(PM10_temp)
    file['PM10_min'] = PM10_min
    file['PM10_max'] = PM10_max
    file['PM10_max_min'] = PM10_max_min
    file['PM10_mean'] = PM10_mean
    file['PM10_median'] = PM10_median
    file['PM10_std'] = PM10_std
    return


# In[18]:


def get_O3_stats(file):
    O3_temp = []
    O3_min = list(file['O3_min'])
    O3_max = list(file['O3_max'])
    O3_max_min = list(file['O3_max_min'])
    O3_mean = list(file['O3_mean'])
    O3_median = list(file['O3_median'])
    O3_std = list(file['O3_std'])
    for i in range(48,len(O3_min)):
        for j in range(48):
            O3_temp.append(file['O3'][i-48+j])
        O3_min[i] = min(O3_temp)
        O3_max[i] = max(O3_temp)
        O3_max_min[i] = max(O3_temp) - min(O3_temp)
        O3_mean[i] = statistics.mean(O3_temp)
        O3_median[i] = statistics.median(PM10_temp)
        O3_std[i] = statistics.stdev(O3_temp)
    file['O3_min'] = O3_min
    file['O3_max'] = O3_max
    file['O3_max_min'] = O3_max_min
    file['O3_mean'] = O3_mean
    file['O3_median'] = O3_median
    file['O3_std'] = O3_std
    return                       


# In[19]:


def get_air_condition_feature():
    for aq in tqdm(aq_index):
        file = pd.read_csv('feature_' + aq + '.csv')
        get_PM25_stats(file)
        get_PM10_stats(file)
        get_O3_stats(file)
        file.to_csv('D:/HKUST/data mining and knowledge discovery/Project/work2/feature data2/' + 'feature_' + aq +'.csv')


# In[20]:


get_air_condition_feature()

