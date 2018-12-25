
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
from tqdm import tqdm


# In[23]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2


# In[24]:


air_station_position = pd.read_csv('air_station_position.csv')
aq_17_18 = pd.read_csv('air_quality_full.csv')
del aq_17_18['Unnamed: 0']
print(aq_17_18.shape)
aq_17_18


# In[25]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2\air condition data


# In[26]:


stationId = list(air_station_position['stationId'])

aq_17_18_split = []
aq = aq_17_18.groupby(["stationId"])

for i in range(35):
    aq_17_18_split.append(aq.get_group(stationId[i]))

for i in tqdm(range(35)):
    aq_17_18_split[i].to_csv('aq_17_18_split_' + str(stationId[i].split('_aq')[0]) + '.csv')


# In[27]:


aq_17_18_split


# In[58]:


from datetime import datetime 
aq_17_18.loc[0,'utc_time'][5:7]

