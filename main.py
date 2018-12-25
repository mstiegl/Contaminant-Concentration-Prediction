
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[108]:


cd D:\HKUST\data mining and knowledge discovery\Project\work\data


# In[109]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[111]:


train["temperature"] = np.log10(train["temperature"])
train["pressure"] = np.log10(train["pressure"])
train["humidity"] = np.log10(train["humidity"])
train["wind_speed"] = np.log10(train["wind_speed"])
train['PM25_min'] = np.log10(train['PM25_min'])
train['PM25_max'] = np.log10(train['PM25_max'])
train['PM25_max_min'] = np.log10(train['PM25_max_min'])
train['PM25_mean'] = np.log10(train['PM25_mean'])
train['PM25_median'] = np.log10(train['PM25_median'])
train['PM25_std'] = np.log10(train['PM25_std'])
train['PM10_min'] = np.log10(train['PM10_min'])
train['PM10_max'] = np.log10(train['PM10_max'])
train['PM10_max_min'] = np.log10(train['PM10_max_min'])
train['PM10_mean'] = np.log10(train['PM10_mean'])
train['PM10_median'] = np.log10(train['PM10_median'])
train['PM10_std'] = np.log10(train['PM10_std'])
train['O3_min'] = np.log10(train['O3_min'])
train['O3_max'] = np.log10(train['O3_max'])
train['O3_max_min'] = np.log10(train['O3_max_min'])
train['O3_mean'] = np.log10(train['O3_mean'])
train['O3_median'] = np.log10(train['O3_median'])
train['O3_std'] = np.log10(train['O3_std'])
test["temperature"] = np.log10(test["temperature"])
test["pressure"] = np.log10(test["pressure"])
test["humidity"] = np.log10(test["humidity"])
test["wind_speed"] = np.log10(test["wind_speed"])
test['PM25_min'] = np.log10(test['PM25_min'])
test['PM25_max'] = np.log10(test['PM25_max'])
test['PM25_max_min'] = np.log10(test['PM25_max_min'])
test['PM25_mean'] = np.log10(test['PM25_mean'])
test['PM25_median'] = np.log10(test['PM25_median'])
test['PM25_std'] = np.log10(test['PM25_std'])
test['PM10_min'] = np.log10(test['PM10_min'])
test['PM10_max'] = np.log10(test['PM10_max'])
test['PM10_max_min'] = np.log10(test['PM10_max_min'])
test['PM10_mean'] = np.log10(test['PM10_mean'])
test['PM10_median'] = np.log10(test['PM10_median'])
test['PM10_std'] = np.log10(test['PM10_std'])
test['O3_min'] = np.log10(test['O3_min'])
test['O3_max'] = np.log10(test['O3_max'])
test['O3_max_min'] = np.log10(test['O3_max_min'])
test['O3_mean'] = np.log10(test['O3_mean'])
test['O3_median'] = np.log10(test['O3_median'])
test['O3_std'] = np.log10(test['O3_std'])


# In[115]:


train['wind_direction'] = train['wind_direction'].astype(str)
test['wind_direction'] = test['wind_direction'].astype(str)
train['weekday'] = train['weekday'].astype(str)
test['weekday'] = test['weekday'].astype(str)
train['rush_time'] = train['rush_time'].astype(str)
test['rush_time'] = test['rush_time'].astype(str)
train['mid_night'] = train['mid_night'].astype(str)
test['mid_night'] = test['mid_night'].astype(str)
train['month'] = train['month'].astype(str)
test['month'] = test['month'].astype(str)


# In[116]:


train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[121]:


from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.distplot(train['PM2.5'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['PM2.5'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('PM2.5')
plt.title('PM2.5 Distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['PM2.5'], plot=plt)
plt.show()


# In[122]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["PM2.5"] = np.log10(train["PM2.5"])
train["PM10"] = np.log10(train["PM10"])
train["O3"] = np.log10(train["O3"])

#Check the new distribution 
sns.distplot(train['PM2.5'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['PM2.5'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('PM2.5')
plt.title('PM2.5 distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['PM2.5'], plot=plt)
plt.show()


# In[123]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[125]:


y_train = train.loc[:,['PM2.5','PM10','O3']]
del train['PM2.5']
del train['PM10']
del train['O3']
y_train.head(5)


# In[145]:


del test['PM2.5']
del test['PM10']
del test['O3']


# In[79]:


#Validation function
n_folds = 5

def rmsle_cv(model,train,y_train):
    kf = KFold(n_folds, shuffle=True, random_state=None).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def test_PM25_predict(model):
    for i in range(len(test)):
        if i == 0:
            test.loc[i,'PM2.5'] = model.predict(test.loc[i,:])
        else:
            test.loc[i,'PM25lasthour'] = test.loc[i-1,'PM2.5']
            test.loc[i,'PM2.5'] = model.predict(test.loc[i,:])
        PM25 = list(test.loc[,'PM2.5'])
    return PM25
def test_PM10_predict(model):
    for i in range(len(test)):
        if i == 0:
            test.loc[i,'PM10'] = model.predict(test.loc[i,:])
        else:
            test.loc[i,'PM10lasthour'] = test.loc[i-1,'PM10']
            test.loc[i,'PM10'] = model.predict(test.loc[i,:])
        PM10 = list(test.loc[,'PM10'])
    return PM10
def test_O3_predict(model):
    for i in range(len(test)):
        if i == 0:
            test.loc[i,'O3'] = model.predict(test.loc[i,:])
        else:
            test.loc[i,'O3lasthour'] = test.loc[i-1,'O3']
            test.loc[i,'PM10'] = model.predict(test.loc[i,:])
        O3 = list(test.loc[,'O3'])
    return O3


# In[132]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01,
                                   max_depth=10, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber')


# In[133]:


score = rmsle_cv(GBoost,train,y_train['PM2.5'])
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[134]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.9, gamma=0,
                             learning_rate=0.1, max_depth=7, 
                             min_child_weight=8, n_estimators=2200,
                             reg_alpha=1.15, reg_lambda=2.4,
                             subsample=0.6, silent=1,
                             nthread = -1)


# In[136]:


score = rmsle_cv(model_xgb,train,y_train['PM2.5'])
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[137]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[138]:


score = rmsle_cv(model_lgb,train,y_train['PM2.5'])
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# In[141]:


from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(hidden_layer_sizes=(200,150,100,50), activation='tanh', 
                   solver='adam', alpha=0.001, batch_size='auto', 
                   learning_rate='constant', learning_rate_init=0.01, 
                   power_t=0.6, max_iter=2000000, shuffle=True, random_state=None, 
                   tol=0.00001, verbose=False, warm_start=False, momentum=0.9, 
                   nesterovs_momentum=False, early_stopping=False, validation_fraction=0.3, 
                   beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=30)


# In[142]:


score = rmsle_cv(MLP,train,y_train['PM2.5'])
print("MLP score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[143]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[151]:


# PM2.5
GBoost.fit(train, y_train['PM2.5'])
GBoost_train_pred = GBoost.predict(train)
GBoost_PM25_pred = 10**(test_PM25_predict(GBoost))
print(rmsle(y_train['PM2.5'], GBoost_train_pred))


# In[152]:


model_lgb.fit(train, y_train['PM2.5'])
lgb_train_pred = model_lgb.predict(train)
lgb_PM25_pred = 10**(test_PM25_predict(model_lgb))
print(rmsle(y_train['PM2.5'], lgb_train_pred))


# In[153]:


model_xgb.fit(train, y_train['PM2.5'])
xgb_train_pred = model_xgb.predict(train)
xgb_PM25_pred = 10**(test_PM25_predict(model_xgb))
print(rmsle(y_train['PM2.5'], xgb_train_pred))


# In[154]:


MLP.fit(train, y_train['PM2.5'])
MLP_train_pred = MLP.predict(train)
MLP_PM25_pred = 10**(test_PM25_predict(MLP))
print(rmsle(y_train['PM2.5'], MLP_train_pred))


# In[155]:


# PM10
GBoost.fit(train, y_train['PM10'])
GBoost_train_pred = GBoost.predict(train)
GBoost_PM10_pred = 10**(test_PM10_predict(GBoost))
print(rmsle(y_train['PM10'], GBoost_train_pred))


# In[156]:


model_lgb.fit(train, y_train['PM10'])
lgb_train_pred = model_lgb.predict(train)
lgb_PM10_pred = 10**(test_PM10_predict(model_lgb))
print(rmsle(y_train['PM10'], lgb_train_pred))


# In[157]:


model_xgb.fit(train, y_train['PM10'])
xgb_train_pred = model_xgb.predict(train)
xgb_PM10_pred = 10**(test_PM10_predict(model_xgb))
print(rmsle(y_train['PM10'], xgb_train_pred))


# In[158]:


MLP.fit(train, y_train['PM10'])
MLP_train_pred = MLP.predict(train)
MLP_PM10_pred = 10**(test_PM10_predict(MLP))
print(rmsle(y_train['PM10'], MLP_train_pred))


# In[159]:


# O3
GBoost.fit(train, y_train['O3'])
GBoost_train_pred = GBoost.predict(train)
GBoost_O3_pred = 10**(test_O3_predict(GBoost))
print(rmsle(y_train['O3'], GBoost_train_pred))


# In[160]:


model_lgb.fit(train, y_train['O3'])
lgb_train_pred = model_lgb.predict(train)
lgb_O3_pred = 10**(test_O3_predict(model_lgb))
print(rmsle(y_train['O3'], lgb_train_pred))


# In[161]:


model_xgb.fit(train, y_train['O3'])
xgb_train_pred = model_xgb.predict(train)
xgb_O3_pred = 10**(test_O3_predict(model_xgb))
print(rmsle(y_train['O3'], xgb_train_pred))


# In[162]:


MLP.fit(train, y_train['O3'])
MLP_train_pred = MLP.predict(train)
MLP_O3_pred = 10**(test_O3_predict(MLP))
print(rmsle(y_train['O3'], MLP_train_pred))


# In[163]:


predict_PM25 = 0.4*xgb_PM25_pred + 0.3*GBoost_PM25_pred + 0.3*lgb_PM25_pred
predict_PM10 = 0.4*xgb_PM10_pred + 0.3*GBoost_PM10_pred + 0.3*lgb_PM10_pred
predict_O3 = 0.4*xgb_O3_pred + 0.3*GBoost_O3_pred + 0.3*lgb_O3_pred


# In[165]:


sub = pd.DataFrame()
sub['PM2.5'] = predict_PM25
sub['PM10'] = predict_PM10
sub['O3'] = predict_O3
sub.to_csv('submission.csv',index=False)


# In[167]:


submission = pd.read_csv('D:\HKUST\data mining and knowledge discovery\Project\msbd5002project_18\submission.csv')


# In[168]:


submission.head()

