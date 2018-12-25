
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np


# In[48]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2


# In[49]:


aq_17_18 = pd.read_csv('airQuality_201701-201801.csv')
aq_1802_1803 = pd.read_csv('airQuality_201802-201803.csv')
aq_1804 = pd.read_csv('aiqQuality_201804.csv')
ow_1701_1801_full = pd.read_csv('ow_1701_1801_full.csv')


# In[50]:


# Use spatial relations to fill in the neighborhood: find the nearest neighbor point data as a substitute; 
# if the same is missing, expand the range to find the nearest neighbor
# Use time to fill the missing value: take the air quality data near the time of the missing position as the substitute
def get_map():
    air_station_position = pd.read_csv('air_station_position.csv')
    aq_stationId = list(air_station_position['stationId'])
    aq_longitude = list(air_station_position['longitude'])
    aq_latitude = list(air_station_position['latitude'])
    aq_type = list(air_station_position['type'])
    n_pos = 35

    
    # Caculate the distance
    distance = []
    neighbor = []
    for i in range(13):
        temp = []
        index = []
        for j in range(n_pos):
            index.append(j)
        del index[i]
        for k in range(len(index)):
            temp.append(pow(aq_longitude[i] - aq_longitude[index[k]],2) + pow(aq_latitude[i] - aq_latitude[index[k]],2))
        distance.append(temp)
        # identify the nearest neighbor
        aq_distance = sorted(temp)
        aq_neighbor = []
        for j in range(12):
            t_index = temp.index(aq_distance[j])
            if t_index < i:
                aq_neighbor.append(aq_stationId[t_index])
            else:
                aq_neighbor.append(aq_stationId[t_index + 1])
            neighbor.append(aq_neighbor)

    # get the nearest station pair
    best_pair8full = {}
    for i in range(n_pos):
        for j in range(12):
            best_pair4full.setdefault(aq_stationId[i],[]).append(neighbor[i][j])
    return best_pair4full
best_pair12full = get_map()


# In[51]:


best_pair12full


# In[52]:


from tqdm import tqdm
import datetime

aq_map = {}
aq_id = list(aq_17_18['stationId'])
aq_time = list(aq_17_18['utc_time'])
aq_pm25 = list(aq_17_18['PM2.5'])
aq_pm10 = list(aq_17_18['PM10'])
aq_no2 = list(aq_17_18['NO2'])
aq_co = list(aq_17_18['CO'])
aq_o3 = list(aq_17_18['O3'])
aq_so2 = list(aq_17_18['SO2'])
data2full_index = [aq_pm25,aq_pm10,aq_no2,aq_co,aq_o3,aq_so2] 

for i in range(len(aq_id)):
    key = str(aq_id[i]) + str(aq_time[i]) 
    for each in data2full_index:
        aq_map.setdefault(key,[]).append(each[i])

# fill
aq_pm25,aq_pm10,aq_no2,aq_co,aq_o3,aq_so2 = list(),list(),list(),list(),list(),list()
aq_index = [aq_pm25,aq_pm10,aq_no2,aq_co,aq_o3,aq_so2]
avg_fillna = [58.785570,88.059259,45.792457,0.960677,55.692755,8.981003]

for i in tqdm(range(len(aq_id))):
    for j in range(len(aq_index)):
        # No missing value
        if not np.isnan(data2full_index[j][i]):
            aq_index[j].append(data2full_index[j][i])
        # fill missing value by using the nearest neighbor
        elif (str(best_pair8full[str(aq_id[i])][0]) + str(aq_time[i])) in aq_map:
            if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][0]) + str(aq_time[i])][j]):
                aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][0]) + str(aq_time[i])][j])
            elif (str(best_pair8full[str(aq_id[i])][1]) + str(aq_time[i])) in aq_map:
                if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][1]) + str(aq_time[i])][j]):
                    aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][1]) + str(aq_time[i])][j])
                elif (str(best_pair8full[str(aq_id[i])][2]) + str(aq_time[i])) in aq_map:
                    if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][2]) + str(aq_time[i])][j]):
                        aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][2]) + str(aq_time[i])][j])
                    elif (str(best_pair8full[str(aq_id[i])][3]) + str(aq_time[i])) in aq_map:
                        if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][3]) + str(aq_time[i])][j]):
                            aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][3]) + str(aq_time[i])][j])
                        elif (str(best_pair8full[str(aq_id[i])][4]) + str(aq_time[i])) in aq_map:
                            if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][4]) + str(aq_time[i])][j]):
                                aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][4]) + str(aq_time[i])][j])
                            elif (str(best_pair8full[str(aq_id[i])][5]) + str(aq_time[i])) in aq_map:
                                if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][5]) + str(aq_time[i])][j]):
                                    aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][5]) + str(aq_time[i])][j])
                                elif (str(best_pair8full[str(aq_id[i])][6]) + str(aq_time[i])) in aq_map:
                                    if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][6]) + str(aq_time[i])][j]):
                                        aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][6]) + str(aq_time[i])][j])
                                    elif (str(best_pair8full[str(aq_id[i])][7]) + str(aq_time[i])) in aq_map:
                                        if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][7]) + str(aq_time[i])][j]):
                                            aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][7]) + str(aq_time[i])][j])
                                        elif (str(best_pair8full[str(aq_id[i])][7]) + str(aq_time[i])) in aq_map:
                                            if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][8]) + str(aq_time[i])][j]):
                                                aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][8]) + str(aq_time[i])][j])
                                            elif (str(best_pair8full[str(aq_id[i])][9]) + str(aq_time[i])) in aq_map:
                                                if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][9]) + str(aq_time[i])][j]):
                                                    aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][9]) + str(aq_time[i])][j])
                                                elif (str(best_pair8full[str(aq_id[i])][10]) + str(aq_time[i])) in aq_map:
                                                    if not np.isnan(aq_map[str(best_pair8full[str(aq_id[i])][10]) + str(aq_time[i])][j]):
                                                        aq_index[j].append(aq_map[str(best_pair8full[str(aq_id[i])][10]) + str(aq_time[i])][j])
                                                    else:
                                                        aq_index[j].append(avg_fillna[j])  
                                                else:
                                                    aq_index[j].append(avg_fillna[j])    
                                            else:
                                                aq_index[j].append(avg_fillna[j])
                                        else:
                                            aq_index[j].append(avg_fillna[j])
                                    else:
                                        aq_index[j].append(avg_fillna[j])
                                else:
                                    aq_index[j].append(avg_fillna[j])
                            else:
                                aq_index[j].append(avg_fillna[j])
                        else:
                            aq_index[j].append(avg_fillna[j])
                    else:
                        aq_index[j].append(avg_fillna[j])
                else:
                    aq_index[j].append(avg_fillna[j])
            else:
                aq_index[j].append(avg_fillna[j])
        else:
            aq_index[j].append(avg_fillna[j])


# In[53]:


aq_index[1]


# In[54]:


for i in range(len(aq_index[1])):
    if aq_index[1][i] == 88.059259:
        aq_index[1][i] = 1.8079*aq_index[0][i]+17.7440


# In[55]:


aq_full = []
index = [aq_id,aq_time,aq_pm25,aq_pm10,aq_no2,aq_co,aq_o3,aq_so2]

for each in index:
    aq_full.append(each)

c_index = ['stationId','utc_time','PM2.5','PM10','NO2','CO','O3','SO2']

df_aq_full = pd.DataFrame(np.transpose(aq_full) ,index = None, columns = c_index)
df_aq_full.to_csv('air_quality_full.csv')
df_aq_full.tail()


# In[56]:


def get_missing_data_table(data):
    total = data.isnull().sum()
    percentage = data.isnull().sum() / data.isnull().count()
    missing_data = pd.concat([total,percentage],axis = 'columns',keys = ['TOTAL','PERCENTAGE'])
    return missing_data.sort_index(ascending = True)


# In[57]:


get_missing_data_table(df_aq_full)

