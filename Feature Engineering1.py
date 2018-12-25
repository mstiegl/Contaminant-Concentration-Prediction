
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tqdm import tqdm
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from __future__ import division  
from sklearn.metrics import mean_absolute_error,mean_squared_error
import requests
import matplotlib.pyplot as plt


# In[2]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2


# In[3]:


air_station_position = pd.read_csv('air_station_position.csv')
aq_index = list(air_station_position['stationId'])


# In[4]:


aq_index


# In[5]:


for i in range(len(aq_index)):
    aq_index[i] = aq_index[i].split('_')[0]


# In[6]:


def get_intersection(a,b):
    return list(set(a).intersection(set(b)))  
    
def get_aq_feature(df,time_list):
    aq_feature = [[],[],[],[],[],[]]
    for each in df.iterrows():
        if each[1]['utc_time'] in time_list:
            aq_feature[0].append(each[1]['PM2.5'])
            aq_feature[1].append(each[1]['PM10'])
            aq_feature[2].append(each[1]['NO2'])
            aq_feature[3].append(each[1]['CO'])
            aq_feature[4].append(each[1]['O3'])
            aq_feature[5].append(each[1]['SO2'])
    return aq_feature

def get_meo_feature(df,time_list):
    aq_feature = [[],[],[],[],[]]
    for each in df.iterrows():
        if each[1]['utc_time'] in time_list:
            aq_feature[0].append(each[1]['temperature'])
            aq_feature[1].append(each[1]['pressure'])
            aq_feature[2].append(each[1]['humidity'])
            aq_feature[3].append(each[1]['wind_direction'])
            aq_feature[4].append(each[1]['wind_speed/kph'])
    return aq_feature

def get_time_feature(df,time_list):
    aq_feature = [[],[],[],[]]
    for each in df.iterrows():
        if each[1]['utc_time'] in time_list:
            # is weekday?
            aq_feature[0].append(datetime.strptime(each[1]['utc_time'][:10], '%Y-%m-%d').weekday())
            # is rush hour?
            if each[1]['utc_time'][11:13] in ['00','01','02','09','10','11','12']:
                aq_feature[1].append(1)
            else:
                aq_feature[1].append(0)
            # is midnight?
            if each[1]['utc_time'][11:13] in ['15','16','17','18','19','20','21','22']:
                aq_feature[2].append(1)
            else:
                aq_feature[2].append(0)
            # get month feature
            aq_feature[3].append(each[1]['utc_time'][5:7])
    return aq_feature


# In[7]:


cd D:\HKUST\data mining and knowledge discovery\Project\work2\air condition data


# In[8]:


for each in tqdm(aq_index):
    aq_feature = []
    # load file
    df_aq = pd.read_csv('aq_17_18_split_'+ each + '.csv')
    del df_aq['Unnamed: 0']
    df_aq = df_aq.drop_duplicates(['utc_time'])
    df_meo = pd.read_csv('D:/HKUST/data mining and knowledge discovery/Project/work2/air condition weather data/'+ each + '.csv')
    del df_meo['Unnamed: 0']
    df_meo = df_meo.drop_duplicates(['utc_time'])
    # get time intersection index
    aq_time_index = get_intersection(list(df_aq['utc_time']),list(df_meo['utc_time']))
    # get value and feature
    feature_aq = get_aq_feature(df_aq,aq_time_index)
    feature_meo = get_meo_feature(df_meo,aq_time_index)
    feature_time = get_time_feature(df_meo,aq_time_index)
    aq_feature = feature_aq
    aq_feature.extend(feature_meo)
    aq_feature.extend(feature_time)
    # feature to csv
    aq_col_index = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2','temperature', 'pressure', 'humidity','wind_direction', 'wind_speed/kph','weekday','rush_time','mid_night','month']
    aq_gen_df = DataFrame(np.transpose(aq_feature) ,index = None, columns = aq_col_index)
    aq_gen_df.to_csv('D:/HKUST/data mining and knowledge discovery/Project/work2/feature data/' + 'feature_'+ each +'.csv')


# In[30]:


# Statistical characteristics of air quality in historical Windows were extracted
def get_bin_feature(agg,city,n_bin,n_in):
    if city == 0:
        n_aq = 3
    else:
        n_aq = 6

    stats_feature = []
    for i in range(n_aq * 7):
        stats_feature.append([])
    # aq in df to list
    df_aq = []
    for i in range(n_aq):
        for j in range(n_in-n_bin,n_in):
            df_aq.append(list(agg['var'+str(i + 1)+'(t-'+str(j + 1)+')']))
    # get statistics feature for each aq in historcal windows
    aq = []
    for i in range(n_aq * 7):
        aq.append([])
        # sample dimention
    for i in tqdm(range(agg.shape[0])):
        aq = []
            # aq type numbers
        for j in range(n_aq):
            aq.append([])
            for k in range(j * n_bin,(j + 1) * n_bin):
                if np.isnan(df_aq[k][i]):
                    aq[j].append(0)
                else:
                    aq[j].append(df_aq[k][i])
            stats_feature[7 * j].append(statistics.mean(aq[j]))
            stats_feature[7 * j + 1].append(statistics.median(aq[j]))
            stats_feature[7 * j + 2].append(statistics.variance(aq[j]))
            stats_feature[7 * j + 3].append(statistics.stdev(aq[j]))
            stats_feature[7 * j + 4].append(max(aq[j]))
            stats_feature[7 * j + 5].append(min(aq[j]))
            stats_feature[7 * j + 6].append(max(aq[j]) - min(aq[j]))

    insert_index = 0
    for i in range(len(stats_feature)):
        agg.insert(insert_index + i,'bin'+str(n_bin)+'-'+ str(i),stats_feature[i])
    return agg


# In[ ]:


import statistics
    
# Turn sequences into supervised learning problems
def series_to_supervised(data, meo, n_in = 1, n_out = 1, dropnan = True, city = 1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # imput sequences(t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    
    # prediction sequences (t, t+1, ... t+n)
    for i in range(n_out - 1, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t + %d)' % (j + 1, i)) for j in range(n_vars)]
    
    # concat input sequences and prediction sequences
    agg = concat(cols, axis = 1)
    agg.columns = names
    
    # Clear the fields in the value window except for the target variable
    drop_index = []
    if city == 1:
        index = [3,4,6,12,13]
        for each in index:
            drop_index.append('var' + str(each) + '(t + '+ str(n_out - 1)+')')
    else:
        index = [3,9,10]
        for each in index:
            drop_index.append('var' + str(each) + '(t + '+ str(n_out - 1)+')')
    agg.drop(drop_index, axis = 1, inplace = True)
    
    if city == 0:
        for i in range(2,n_in + 1):
            t1 = i
            for j in range(4,8):
                t2 = j
                agg.drop('var'+str(t2)+'(t-'+str(t1)+')', axis = 1, inplace = True)
    else:
        for i in range(2,n_in + 1):
            t1 = i
            for j in range(7,11):
                t2 = j
                agg.drop('var'+str(t2)+'(t-'+str(t1)+')', axis = 1, inplace = True)
    
    bin_index = [24,48]
    for each in bin_index:
        agg = get_bin_feature(agg,city,each,n_in)
        
    test_df = agg[-48:]
    if city == 1:
        test_df.drop('var1' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var2' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var5' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var7' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var8' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var9' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var10' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var11' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
    else:
        test_df.drop('var1' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var2' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var4' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var5' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var6' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var7' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
        test_df.drop('var8' + '(t + '+ str(n_out - 1)+')',axis = 1, inplace = True)
    
    if city == 1:
        meo.columns = ['var7(t + 71)','var8(t + 71)','var9(t + 71)','var10(t + 71)','var11(t + 71)']
    else:
        meo.columns = ['var4(t + 71)','var5(t + 71)','var6(t + 71)','var7(t + 71)','var8(t + 71)']
    test_df = DataFrame(np.concatenate((test_df.values, meo.values), axis = 1))
    
    if dropnan:
        agg.dropna(inplace = True)
    
    if city == 1:
        bj_var1 = agg['var5(t + 71)']
        agg.drop(labels=['var5(t + 71)'], axis=1,inplace = True)
        agg.insert(0, 'var5(t + 71)', bj_var1)
        bj_var2 = agg['var2(t + 71)']
        agg.drop(labels=['var2(t + 71)'], axis=1,inplace = True)
        agg.insert(0, 'var2(t + 71)', bj_var2)
        bj_var5 = agg['var1(t + 71)']
        agg.drop(labels=['var1(t + 71)'], axis=1,inplace = True)
        agg.insert(0, 'var1(t + 71)', bj_var5)
    else:
        ld_var1 = agg['var2(t + 71)']
        agg.drop(labels=['var2(t + 71)'], axis=1,inplace = True)
        agg.insert(0, 'var2(t + 71)', ld_var1)
        ld_var2 = agg['var1(t + 71)']
        agg.drop(labels=['var1(t + 71)'], axis=1,inplace = True)
        agg.insert(0, 'var1(t + 71)', ld_var2)
    
    if city == 1:
        test_df.columns = list(agg.columns)[3:]
    else:
        test_df.columns = list(agg.columns)[2:]
    res = [agg,test_df]
    
    return res

