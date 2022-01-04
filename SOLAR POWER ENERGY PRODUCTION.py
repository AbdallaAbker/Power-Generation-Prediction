#!/usr/bin/env python
# coding: utf-8

# # Predicting Solar Energy Production
# 
# ## Capstone Project
# 
# ### Alberta Machine Learning Institute (Amii)

# #### Project Objective:
# Analyzing the performance of two solar power plants based on data gathered at the inverters level, in addition to the local weather data, in order to build a model that predicts the energy yield.

# ![Solar%20Sys-2.png](attachment:Solar%20Sys-2.png)

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Data
# 
# #### Plant Data:

# In[2]:


plant_1 = pd.read_csv('Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'])
plant_1.head()


# In[3]:


plant_2 = pd.read_csv('Plant_2_Generation_Data.csv', parse_dates=['DATE_TIME'])
plant_2.head()


# #### Fixing the AC Power Data to match DC Power Data

# In[4]:


plant_1['DC_POWER'] = plant_1['DC_POWER'] / 10


# #### Weather Data:

# In[5]:


weather_1 = pd.read_csv('Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'])
weather_2 = pd.read_csv('Plant_2_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'])


# #### Merging Data:

# In[6]:


plants = pd.concat([plant_1, plant_2], axis= 0, ignore_index = 'True')
weather = pd.concat([weather_1, weather_2], axis= 0, ignore_index = 'True')


# In[7]:


df = plants.merge(weather, on=['DATE_TIME', 'PLANT_ID'], how = 'inner')
df.head()


# ## Exploratory Data Analysis

# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.duplicated().any().sum()


# In[12]:


df.isnull().sum()


# In[13]:


df.columns


# In[14]:


import missingno as msno
msno.matrix(df)


# ### Feature Engineering

# #### Creating New Features

# In[15]:


df['PLANT_EFFICIENCY'] = df['AC_POWER'] / df['DC_POWER']
df['PLANT_EFFICIENCY'].describe()


# In[16]:


df['PLANT_EFFICIENCY'].replace(np.nan, 0, inplace = True)


# In[17]:


df.isnull().sum()


# In[18]:


df.sort_values(by=['DATE_TIME'], inplace= True, ignore_index=True)


# In[19]:


df.head()


# In[20]:


df.tail()


# In[21]:


df.shape


# In[22]:


ENERGY_PRODUCTION = []

for i in range(1, len(df.DAILY_YIELD)):
    element =  abs(df.DAILY_YIELD[i] - df.DAILY_YIELD[i-1])
    ENERGY_PRODUCTION.append(element)
    
len(ENERGY_PRODUCTION)
ENERGY_PRODUCTION.append(0)
len(ENERGY_PRODUCTION)
ENERGY_PRODUCTION[0] = 0
df['YIELD'] = ENERGY_PRODUCTION

for i in range(1, len(df.YIELD)):
    if df.YIELD[i] == df.YIELD[i-1]:
        df.YIELD[i-1] = 0
        
df.head()


# In[23]:


df[df['YIELD'] < 0]['YIELD'] = 0


# In[24]:


df.PLANT_ID.unique()


# In[25]:


df[df.PLANT_ID == 0.0]['PLANT_ID'].count()


# In[26]:


df['PLANT_ID'].replace(4135001, 'Plant1', inplace = True)
df['PLANT_ID'].replace(4136001, 'Plant2', inplace = True)


# In[27]:


df['PLANT_ID'].astype('category')


# ### Exploratory Data Analysis

# In[37]:


sns.pairplot(df, hue = 'PLANT_ID')
plt.savefig('Pairplot.png')


# In[28]:


f, ax = plt.subplots(figsize=(12, 8))
ax = sns.heatmap(df.corr(method='pearson'), robust=True, annot=True, fmt='0.3f', linewidths=.5, square=True)


# ### Investigating the Modular Temperature

# In[30]:


g = sns.scatterplot(x = 'MODULE_TEMPERATURE', y = 'AC_POWER', data = df, hue = 'PLANT_ID')
plt.grid()
plt.show()


# In[31]:


df['PLANT_ID'].unique()


# In[26]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')

xs = df['MODULE_TEMPERATURE']
ys = df['IRRADIATION']
zs = df['AC_POWER']

ax.scatter(xs, ys, zs,  c = df['PLANT_ID'] ,cmap = 'coolwarm', s=100, alpha=0.8, edgecolors='w')

ax.set_xlabel('MODULE_TEMPERATURE')
ax.set_ylabel('IRRADIATION')
ax.set_zlabel('AC_POWER')


# In[31]:


df['TIME'] = df['DATE_TIME'].dt.time
df['DATE'] = df['DATE_TIME'].dt.date
df.head()


# In[33]:


#define function to multi plot
def multi_plot(data= None, row = None, col = None, title='AC Power'):
    cols = data.columns # take all column
    gp = plt.figure(figsize=(20,20)) 
    
    gp.subplots_adjust(wspace=0.2, hspace=0.8)
    for i in range(1, len(cols)+1):
        ax = gp.add_subplot(row,col, i)
        data[cols[i-1]].plot(ax=ax, style = 'k.')
        ax.set_title('{} {}'.format(title, cols[i-1]))


# In[34]:


AC_power = df.pivot_table(values='AC_POWER', index='TIME', columns='DATE')
multi_plot(data=AC_power , row=9, col=4, title='AC POWER')


# In[35]:


DC_power = df.pivot_table(values='DC_POWER', index='TIME', columns='DATE')
multi_plot(data=DC_power , row=9, col=4, title='DC POWER')


# In[36]:


MT_power = df.pivot_table(values='MODULE_TEMPERATURE', index='TIME', columns='DATE')
multi_plot(data=MT_power , row=9, col=4, title='MODULE_TEMPERATURE')


# In[37]:


IRR_power = df.pivot_table(values='IRRADIATION', index='TIME', columns='DATE')
multi_plot(data=IRR_power , row=9, col=4, title='IRRADIATION')


# In[38]:


AT_power = df.pivot_table(values='AMBIENT_TEMPERATURE', index='TIME', columns='DATE')
multi_plot(data=AT_power , row=9, col=4, title='AMBIENT_TEMPERATURE')


# In[39]:


DAILY_YIELD = df.pivot_table(values='DAILY_YIELD', index='TIME', columns='DATE')
multi_plot(data=DAILY_YIELD , row=9, col=4, title='DAILY_YIELD')
plt.savefig('Dail Yield')
plt.show()


# In[40]:


YIELD = df.pivot_table(values='YIELD', index='TIME', columns='DATE')
multi_plot(data=YIELD , row=9, col=4, title='YIELD')
plt.savefig('Yield')
plt.show()


# In[41]:


df.plot(x= 'TIME', y='MODULE_TEMPERATURE', color = 'skyblue', style='.', figsize = (15, 8))
df.groupby('TIME')['MODULE_TEMPERATURE'].mean().plot(cmap = 'Reds_r', legend = True, label = ' AVG MODULE TEMPERATURE')
plt.ylabel('MODULE TEMPERATURE')
plt.title('MODULE TEMPERATURE DISTRIBUTION IN 24 HRS FOR 34 DAYS')
plt.grid()
plt.savefig('modeule temp')
plt.show


# In[42]:


df.plot(x= 'TIME', y='IRRADIATION', color = 'skyblue', style='.', figsize = (15, 8))
df.groupby('TIME')['IRRADIATION'].mean().plot(cmap = 'Reds_r', legend = True, label = ' AVG IRRADIATION')
plt.ylabel('IRRADIATION')
plt.title('IRRADIATION DISTRIBUTION IN 24 HRS FOR 34 DAYS')
plt.grid()
plt.savefig('IRRADIATION')
plt.show


# In[43]:


df.groupby('DATE')['MODULE_TEMPERATURE'].mean().plot(cmap = 'Reds_r', figsize = (14,8),legend= True, label = 'AVG MODULE TEMPERATURE PER DAY')
plt.ylabel('MODULE TEMPERATURE')
plt.xlabel('DATE')
plt.title('AVG MODULE TEMPERATURE IIN THE 34 DAY')
plt.grid()
plt.savefig('MODULE_TEMPERATURE')
plt.show


# In[44]:


df.plot(x= 'TIME', y='AC_POWER', color = 'skyblue', style='.', figsize = (15, 8))
df.groupby('TIME')['AC_POWER'].mean().plot(cmap = 'Reds_r', legend = True, label = 'AVG AC POWER')
plt.ylabel('AC_POWWER')
plt.title('AC POWER Distribution IN THE 24 HRS FOR 34 DAYS')
plt.grid()
plt.savefig('AC Power')
plt.show


# In[45]:


df.plot(x= 'TIME', y='PLANT_EFFICIENCY', color = 'skyblue', style='.', figsize = (15, 8))
df.groupby('TIME')['PLANT_EFFICIENCY'].mean().plot(cmap = 'Reds_r', legend = True, label = 'AVG PLANT_EFFICIENCY')
plt.ylabel('PLANT_EFFICIENCY')
plt.title('PLANT_EFFICIENCY IN 24 HRS FOR 34 DAY')
plt.grid()
plt.show


# In[46]:


df.plot(x= 'TIME', y='AMBIENT_TEMPERATURE', color = 'skyblue', style='.', figsize = (15, 8))
df.groupby('TIME')['AMBIENT_TEMPERATURE'].mean().plot(cmap = 'Reds_r', legend = True, label = ' AVG AMBIENT TEMPERATURE')
plt.ylabel('AMBIENT TEMPERATURE')
plt.title('AMBIENT_TEMPERATURE DISTRIBUTION IN 24 HRS FOR 34 DAYS')
plt.grid()
plt.savefig('Ambient Temperature')
plt.show


# In[47]:


df.columns


# ### Machine Learning

# In[32]:


X = df.drop(df[['DATE_TIME', 'DATE', 'DC_POWER',  'AC_POWER','TIME','PLANT_ID', 'SOURCE_KEY_x',
                'SOURCE_KEY_y', 'DAILY_YIELD','TOTAL_YIELD']], axis =1)
y = df.AC_POWER


# In[197]:


display(X)


# ### Analyzing The Extreme Values

# In[48]:


plt.figure(figsize =(14,6))
df.drop('TOTAL_YIELD', axis= 1).boxplot(figsize = (8,4), grid = True)
plt.tight_layout()


# #### Extreme_Values_AMBIENT_TEMPERATURE

# In[53]:


df[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']].boxplot(figsize = (8,4), grid = True)


# In[54]:


#Extreme_Values_AMBIENT_TEMPERATURE

Q1_AT = df['AMBIENT_TEMPERATURE'].quantile(0.25)
Q3_AT = df['AMBIENT_TEMPERATURE'].quantile(0.75)
IQR_AT = Q3_AT - Q1_AT    #IQR is interquartile range 

filtered_AMBIENT_TEMPERATURE = (df['AMBIENT_TEMPERATURE'] >= Q1_AT - 1.5 * IQR_AT) & (df['AMBIENT_TEMPERATURE'] <= Q3_AT + 1.5 *IQR_AT)
Extreme_Values_AMBIENT_TEMPERATURE = X.loc[~filtered_AMBIENT_TEMPERATURE]['AMBIENT_TEMPERATURE']

Extreme_Values_AMBIENT_TEMPERATURE


# #### Extreme_Values_IRRADIATION

# In[55]:


df[['IRRADIATION']].boxplot(figsize = (8,4), grid = True)


# In[56]:


#Extreme_IRRADIATION
Q1_IR = df['IRRADIATION'].quantile(0.25)
Q3_IR = df['IRRADIATION'].quantile(0.75)
IQR_IR = Q3_IR - Q1_IR    #IQR is interquartile range 

Filtered_IRRADIATION = (df['IRRADIATION'] >= Q1_IR - 1.5 * IQR_IR) & (df['IRRADIATION'] <= Q3_IR + 1.5 *IQR_IR)
Extreme_IRRADIATION = X.loc[~Filtered_IRRADIATION]['IRRADIATION']

Extreme_IRRADIATION


# In[57]:


df[['DAILY_YIELD', 'AC_POWER', 'DC_POWER']].boxplot(figsize = (8,4), grid = True)


# In[59]:


#Extreme_Values_AC_POWER

Q1_AC = df['AC_POWER'].quantile(0.25)
Q3_AC = df['AC_POWER'].quantile(0.75)
IQR_AC = Q3_AC - Q1_AC    #IQR is interquartile range 

Filtered_AC_POWER = (df['AC_POWER'] >= Q1_AC - 1.5 * IQR_AC) & (df['AC_POWER'] <= Q3_AC + 1.5 *IQR_AC)
Extreme_Values_AC_POWER = df.loc[~Filtered_AC_POWER]['AC_POWER']

print(Extreme_Values_AC_POWER)


# In[64]:


#Extreme_Values_DAILY_YIELD
Q1_AC = df['YIELD'].quantile(0.25)
Q3_AC = df['YIELD'].quantile(0.75)
IQR_AC = Q3_AC - Q1_AC    #IQR is interquartile range 

Filtered_YIELD = (df['YIELD'] >= Q1_AC - 1.5 * IQR_AC) & (df['YIELD'] <= Q3_AC + 1.5 *IQR_AC)
Extreme_Values_YIELD = df.loc[~Filtered_YIELD]['YIELD']

print(Extreme_Values_YIELD)


# In[61]:


df[['PLANT_EFFICIENCY']].boxplot(figsize = (8,4), grid = True)


# In[61]:


#Extreme_Values_PLANT_EFFIENCY
Q1_AC = df['PLANT_EFFICIENCY'].quantile(0.25)
Q3_AC = df['PLANT_EFFICIENCY'].quantile(0.75)
IQR_AC = Q3_AC - Q1_AC    #IQR is interquartile range 

Filtered_PLANT_EFFICIENCY = (df['PLANT_EFFICIENCY'] >= Q1_AC - 1.5 * IQR_AC) & (df['PLANT_EFFICIENCY'] <= Q3_AC + 1.5 *IQR_AC)
Extreme_Values_PLANT_EFFICIENCY = df.loc[~Filtered_PLANT_EFFICIENCY]['PLANT_EFFICIENCY']

print(Extreme_Values_AC_POWER)


# In[29]:


df['PLANT_EFFICIENCY'].clip(upper=1.0)


# #### Defining my Target & Features

# In[63]:


X.shape, df.shape


# #### Standaradization Vs. Normalization

# In[33]:


#NORMALIZATION
from sklearn.preprocessing import MinMaxScaler


# In[34]:


df_Norm = X.copy()


# In[35]:


Min_Max_scaler = MinMaxScaler()


# In[36]:


Min_Max_scaled_X_train = Min_Max_scaler.fit_transform(df_Norm)


# In[37]:


sns.boxplot(data = Min_Max_scaled_X_train, width = 0.2, color = 'orange')


# In[38]:


sns.histplot(data = Min_Max_scaled_X_train)


# In[39]:


np.std(Min_Max_scaled_X_train)


# In[40]:


np.mean(Min_Max_scaled_X_train)


# #### Standardization

# In[41]:


X.columns


# In[42]:


from sklearn.preprocessing import StandardScaler


# In[43]:


scaler = StandardScaler()


# In[44]:


df_SS = X.copy()


# In[45]:


scaled = scaler.fit_transform(df_SS)


# In[46]:


sns.boxplot(data = scaled, width = 0.2, color = 'green')


# In[47]:


sns.histplot(data = scaled)


# In[48]:


np.mean(scaled)


# In[49]:


np.std(scaled)


# In[50]:


X.columns


# In[51]:


df_SS_clean = X.copy()
scaled_clean = scaler.fit_transform(df_SS_clean)
sns.boxplot(data = scaled_clean, width = 0.2, color = 'green')


# In[52]:


sns.histplot(data = scaled_clean)


# In[53]:


# I have more condidence to choose Standarization with filtered dataset mean =0 & std = 1


# ### Train, Test, and Split, and applying Standarization 

# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# #### Aplying STD on train set and test seperately 

# In[56]:


from sklearn.preprocessing import StandardScaler


# In[57]:


scaler = StandardScaler()


# In[58]:


y_train = np.array(y_train).reshape(-1,1)


# In[59]:


scaled_X_train = scaler.fit_transform(X_train)


# In[60]:


scaled_y_train = scaler.fit_transform(y_train)


# ## KNN 

# ### n = 5

# In[61]:


from sklearn.neighbors import KNeighborsRegressor


# In[62]:


knn_5 = KNeighborsRegressor(n_neighbors= 5)


# In[63]:


knn_5.fit(scaled_X_train, scaled_y_train)


# In[64]:


pred_5 = knn_5.predict(scaled_X_train)


# In[146]:


print('%.5f' % knn_5.score(scaled_X_train, scaled_y_train))


# In[66]:


scaled_X_test = scaler.fit_transform(X_test)


# In[67]:


scaled_y_test = scaler.fit_transform(np.array(y_test).reshape(-1,1))


# In[68]:


pred_5 = knn_5.predict(scaled_X_test)


# In[75]:


print('%.5f' % knn_5.score(scaled_X_test, scaled_y_test))


# In[70]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[71]:


print('MAE:  %.5f' % mean_absolute_error(scaled_y_test, pred_5))
print('MSE:  %.5f' % mean_squared_error(scaled_y_test, pred_5))
print('RMSE: %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_5)))


# ### n = 11

# In[76]:


knn_11 = KNeighborsRegressor(n_neighbors= 11)
knn_11.fit(scaled_X_train, scaled_y_train)
pred_11_trained = knn_11.predict(scaled_X_train)
print('%.5f' % knn_11.score(scaled_X_train, scaled_y_train))


# In[77]:


pred_11 = knn_11.predict(scaled_X_test)
print('%.5f' % knn_11.score(scaled_X_test, scaled_y_test))


# In[78]:


print('MAE:  %.5f' % mean_absolute_error(scaled_y_test, pred_11))
print('MSE:  %.5f' % mean_squared_error(scaled_y_test, pred_11))
print('RMSE:  %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_11)))


# ### n =15

# In[79]:


knn_15 = KNeighborsRegressor(n_neighbors= 15)
knn_15.fit(scaled_X_train, scaled_y_train)
pred_15_trained = knn_15.predict(scaled_X_train)
print('%.5f' % knn_15.score(scaled_X_train, scaled_y_train))


# In[80]:


pred_15 = knn_15.predict(scaled_X_test)
print('%.5f' % knn_15.score(scaled_X_test, scaled_y_test))


# In[81]:


print('MAE:  %.5f' % mean_absolute_error(scaled_y_test, pred_15))
print('MAE:  %.5f' % mean_squared_error(scaled_y_test, pred_15))
print('MAE:  %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_15)))


# ### weight = 'distance' & n= 5

# In[82]:


knn_5 = KNeighborsRegressor(n_neighbors= 5, weights= 'distance')
knn_5.fit(scaled_X_train, scaled_y_train)
pred_5_trained = knn_5.predict(scaled_X_train)
print('%.5f' % knn_5.score(scaled_X_train, scaled_y_train))


# In[83]:


pred_5 = knn_5.predict(scaled_X_test)
print('%.5f' % knn_5.score(scaled_X_test, scaled_y_test))


# In[85]:


print('MAE:  %.5f'% mean_absolute_error(scaled_y_test, pred_5))
print('MSE:  %.5f'% mean_squared_error(scaled_y_test, pred_5))
print('RMSE:  %.5f'% np.sqrt(mean_squared_error(scaled_y_test, pred_5)))


# #### Applying Grid Search Technique to evaluate multiple hyperparameters  

# In[86]:


from sklearn.model_selection import GridSearchCV


# In[87]:


from sklearn.metrics import accuracy_score, r2_score


# In[88]:


accuracy_score


# In[89]:


knn_pid_search = GridSearchCV(estimator=KNeighborsRegressor(), cv=10, 
                              param_grid=dict(n_neighbors=[5,11,15], 
                              p=[1, 2, 3, 4],  weights= ['uniform', 'distance']), scoring='accuracy')


# In[90]:


knn_pid_search.fit(scaled_X_train, scaled_y_train)


# In[91]:


knn_pid_search.best_estimator_


# In[92]:


knn_pid_search.param_grid


# ### KNN best_params_ Winners

# In[95]:


knn_pid_search.best_params_


# In[93]:


knn_best = KNeighborsRegressor(n_neighbors= 5, weights= 'uniform', p=1)
knn_best.fit(scaled_X_train, scaled_y_train)
print('accuracy for training dataset: %.5f'% knn_best.score(scaled_X_train, scaled_y_train))
pred_knn_best = knn_best.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% knn_best.score(scaled_X_test, scaled_y_test))


# In[94]:


print('MAE_knn_best:   %.5f'% mean_absolute_error(scaled_y_test, pred_knn_best))
print('MSE_knn_best:   %.5f'% mean_squared_error(scaled_y_test, pred_knn_best))
print('RMSE_knn_best:  %.5f'% np.sqrt(mean_squared_error(scaled_y_test, pred_knn_best)))
print('r2_score_knn_best:   %.5f'% r2_score(scaled_y_test, pred_knn_best))


# ### KNN Learning Curve

# In[95]:


from sklearn.model_selection import learning_curve


# In[96]:


data_sizes, training_scores, validation_scores = learning_curve(KNeighborsRegressor(n_neighbors= 5, p= 1, weights='uniform'), 
X = scaled_X_train, y =scaled_y_train, cv=10,error_score=0, scoring='neg_mean_squared_error', train_sizes= np.array([0.1  , 0.325, 0.55 , 0.775, 1.]))


# In[97]:


display(data_sizes)


# In[98]:


training_scores


# In[99]:


display(training_scores)
display(training_scores.shape)


# In[100]:


display(validation_scores)
display(validation_scores.shape)


# In[101]:


training_mean = - training_scores.mean(axis=1) 
training_standard_deviation = -training_scores.std(axis=1) 


# In[102]:


validation_mean = - validation_scores.mean(axis=1) 
validation_standard_deviation = -validation_scores.std(axis=1)


# In[103]:


import plotly.graph_objects as go


# In[104]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean,
                        mode='lines',
                        name='Training',
                        line=dict(color='red')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean - training_standard_deviation,
                        mode='lines',
                        name='Training lower bound',
                        line=dict(width=0, color='red'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean + training_standard_deviation,
                        mode='lines',
                        name='Training upper bound',
                        line=dict(width=0, color='red'),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        showlegend=False))

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean,
                        mode='lines',
                        name='Validation',
                        line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean - validation_standard_deviation,
                        mode='lines',
                        name='Validation lower bound',
                        line=dict(width=0, color='blue'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean + validation_standard_deviation,
                        mode='lines',
                        name='Validation upper bound',
                        line=dict(width=0, color='blue'),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.3)',
                        showlegend=False))

fig.update_layout(title='KNN Learning curve',
                 xaxis_title='Dataset size',
                 yaxis_title='Accuracy')
fig.show()


# ## Decision Tree

# In[105]:


from sklearn.tree import DecisionTreeRegressor


# In[106]:


dt = DecisionTreeRegressor()


# ### Exploring Decision Tree Parameters Manually:
# 
# #### Based on Criterion: 

# In[107]:


dt_mse = DecisionTreeRegressor(criterion='mse')


# In[108]:


dt_mse.fit(scaled_X_train, scaled_y_train)


# In[109]:


dt_mse.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_mse.score(scaled_X_train, scaled_y_train))


# In[110]:


pred_dt_mse = dt_mse.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_mse.score(scaled_X_test, scaled_y_test))


# In[112]:


dt_friedman_mse = DecisionTreeRegressor(criterion='friedman_mse')


# In[113]:


dt_friedman_mse.fit(scaled_X_train, scaled_y_train)
dt_friedman_mse.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_friedman_mse.score(scaled_X_train, scaled_y_train))


# In[114]:


pred_dt_friedman_mse = dt_friedman_mse.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_friedman_mse.score(scaled_X_test, scaled_y_test))


# In[115]:


dt_mae = DecisionTreeRegressor(criterion='mae')
dt_mae.fit(scaled_X_train, scaled_y_train)
dt_mae.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_mae.score(scaled_X_train, scaled_y_train))


# In[116]:


pred_dt_mae = dt_mae.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_mae.score(scaled_X_test, scaled_y_test))


# ##### BEST CRITERION IS 'friedman_mse'

# #### Based on Splitter:

# In[117]:


dt_splitter_best = DecisionTreeRegressor(splitter='best')
dt_splitter_best.fit(scaled_X_train, scaled_y_train)
dt_splitter_best.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_splitter_best.score(scaled_X_train, scaled_y_train))

pred_dt_splitter_best = dt_splitter_best.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_splitter_best.score(scaled_X_test, scaled_y_test))


# In[118]:


dt_splitter_random = DecisionTreeRegressor(splitter='random')
dt_splitter_random.fit(scaled_X_train, scaled_y_train)
dt_splitter_random.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_splitter_random.score(scaled_X_train, scaled_y_train))

pred_dt_splitter_random = dt_splitter_random.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_splitter_random.score(scaled_X_test, scaled_y_test))


# #### BEST SPILITTER IS "best"

# #### Based on Mini_sample_spilit

# In[119]:


dt_min_samples_split_1 = DecisionTreeRegressor(min_samples_split = 1.0)
dt_min_samples_split_1.fit(scaled_X_train, scaled_y_train)
dt_min_samples_split_1.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_min_samples_split_1.score(scaled_X_train, scaled_y_train))

pred_dt_min_samples_split_1 = dt_min_samples_split_1.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_min_samples_split_1.score(scaled_X_test, scaled_y_test))


# In[120]:


dt_min_samples_split_2 = DecisionTreeRegressor(min_samples_split = 2)
dt_min_samples_split_2.fit(scaled_X_train, scaled_y_train)
dt_min_samples_split_2.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_min_samples_split_2.score(scaled_X_train, scaled_y_train))

pred_dt_min_samples_split_2 = dt_min_samples_split_2.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_min_samples_split_2.score(scaled_X_test, scaled_y_test))


# #### BEST min_samples_split = 2

# #### Based on mini_sample_leaf:

# In[121]:


dt_min_samples_leaf_1 = DecisionTreeRegressor(min_samples_leaf=1)
dt_min_samples_leaf_1.fit(scaled_X_train, scaled_y_train)
dt_min_samples_leaf_1.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_min_samples_leaf_1.score(scaled_X_train, scaled_y_train))

pred_dt_min_samples_leaf_1 = dt_min_samples_leaf_1.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_min_samples_leaf_1.score(scaled_X_test, scaled_y_test))


# In[122]:


dt_min_samples_leaf_2 = DecisionTreeRegressor(min_samples_leaf=2)
dt_min_samples_leaf_2.fit(scaled_X_train, scaled_y_train)
dt_min_samples_leaf_2.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_min_samples_leaf_2.score(scaled_X_train, scaled_y_train))

pred_dt_min_samples_leaf_2 = dt_min_samples_leaf_2.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_min_samples_leaf_2.score(scaled_X_test, scaled_y_test))


# #### BEST min_samples_leaf=2

# #### based on Max_depth

# In[123]:


dt_max_depth_4 = DecisionTreeRegressor(max_depth=4)
dt_max_depth_4.fit(scaled_X_train, scaled_y_train)
dt_max_depth_4.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_max_depth_4.score(scaled_X_train, scaled_y_train))

pred_dt_max_depth_4 = dt_max_depth_4.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_max_depth_4.score(scaled_X_test, scaled_y_test))


# In[124]:


dt_max_depth_8 = DecisionTreeRegressor(max_depth=8)
dt_max_depth_8.fit(scaled_X_train, scaled_y_train)
dt_max_depth_8.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_max_depth_8.score(scaled_X_train, scaled_y_train))

pred_dt_max_depth_8 = dt_max_depth_8.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_max_depth_8.score(scaled_X_test, scaled_y_test))


# In[125]:


dt_max_depth_12 = DecisionTreeRegressor(max_depth=12)
dt_max_depth_12.fit(scaled_X_train, scaled_y_train)
dt_max_depth_12.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_max_depth_12.score(scaled_X_train, scaled_y_train))

pred_dt_max_depth_12 = dt_max_depth_12.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_max_depth_12.score(scaled_X_test, scaled_y_test))


# In[126]:


dt_max_depth_16 = DecisionTreeRegressor(max_depth=16)
dt_max_depth_16.fit(scaled_X_train, scaled_y_train)
dt_max_depth_16.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_max_depth_16.score(scaled_X_train, scaled_y_train))

pred_dt_max_depth_16 = dt_max_depth_16.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_max_depth_16.score(scaled_X_test, scaled_y_test))


# #### BEST MAX_DEPTH = 12

# In[128]:


dt_best = DecisionTreeRegressor(criterion='friedman_mse',
    splitter='best',
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=2)


# In[129]:


dt_best.fit(scaled_X_train, scaled_y_train)


# In[130]:


dt_best.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% dt_best.score(scaled_X_train, scaled_y_train))

pred_dt_best = dt_best.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% dt_best.score(scaled_X_test, scaled_y_test))


# In[131]:


print('MAE_dt_best:  %.5f'% mean_absolute_error(scaled_y_test, pred_dt_best))
print('MSE_dt_best:  %.5f'% mean_squared_error(scaled_y_test, pred_dt_best))
print('RMSE_dt_best: %.5f'% np.sqrt(mean_squared_error(scaled_y_test, pred_dt_best)))
print('r2_score_dt_best:  %.5f'% r2_score(scaled_y_test, pred_dt_best))


# In[173]:


from sklearn.tree import plot_tree


# In[176]:


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,8), dpi=300)
plot_tree(dt_best,
    max_depth=5,
    feature_names=None,
    class_names=None,
    label='all',
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rotate='deprecated',
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None)
plt.title('Decision Tree')
plt.savefig('Dtree')

plt.show()


# ### Insights from The Decision Tree Model:

# In[132]:


print('The feature was used for the first split is: IRRADIATION')
print('The number of leaves in the optimal QuAM is: ', dt_best.get_n_leaves())
print('The maximum depth in the optimal Quamis: ', dt_best.max_depth)


# In[133]:


data_sizes, training_scores, validation_scores = learning_curve(dt_best,X = scaled_X_train, y =scaled_y_train,
                                                    cv=10,error_score=0, scoring='neg_mean_squared_error', train_sizes= np.array([0.1  , 0.325, 0.55 , 0.775, 1.]))


training_mean = - training_scores.mean(axis=1) 
training_standard_deviation = -training_scores.std(axis=1) 

validation_mean = - validation_scores.mean(axis=1) 
validation_standard_deviation =- validation_scores.std(axis=1)



import plotly.graph_objects as go


fig = go.Figure()

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean,
                        mode='lines',
                        name='Training',
                        line=dict(color='red')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean - training_standard_deviation,
                        mode='lines',
                        name='Training lower bound',
                        line=dict(width=0, color='red'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean + training_standard_deviation,
                        mode='lines',
                        name='Training upper bound',
                        line=dict(width=0, color='red'),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        showlegend=False))

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean,
                        mode='lines',
                        name='Validation',
                        line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean - validation_standard_deviation,
                        mode='lines',
                        name='Validation lower bound',
                        line=dict(width=0, color='blue'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean + validation_standard_deviation,
                        mode='lines',
                        name='Validation upper bound',
                        line=dict(width=0, color='blue'),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.3)',
                        showlegend=False))

fig.update_layout(title='Learning curve  for The Decision Tree Model',
                 xaxis_title='Dataset size',
                 yaxis_title='Accuracy')
fig.show()


# In[134]:


feat_importance = pd.Series(dt_best.feature_importances_, index = X.columns)
feat_importance.nlargest().plot(kind = 'barh')
plt.title('First Spiliter = Most Important Feature')

plt.grid()
plt.tight_layout()
plt.show()


# In[145]:


#Scores from dicts_dt:
dicts_dt = {'MAE': 0.39026,'MSE': 0.60622,'RMSE':  0.77860, 'r2_score':  0.39378}
dicts_dt


# In[147]:


#Scores from Knn_best
dicts_knn = {'MAE': 0.05272,'MSE': 0.02256,'RMSE':  0.15020,'r2_score': 0.97744}


# In[148]:


fig, [ax1,ax2] = plt.subplots(1,2, sharey=True, figsize=(12,6))
ax1.bar(*zip(*dicts_dt.items()), color = 'red')
ax1.set_title('Decision Tree')
ax1.grid()

ax2.bar(*zip(*dicts_knn.items()), color = 'blue')
ax2.set_title('KNN')
ax2.grid()

fig.tight_layout()
plt.show()


# ### Pre-Pruning Automated hyper-tuning for Decision Tree:

# In[135]:


dt_criterion_search = GridSearchCV(estimator=DecisionTreeRegressor(), cv=10, 
                              param_grid=dict(criterion=['mse', 'friedman_mse', 'mae'],
                                   splitter=['best', 'random'],
                                   min_samples_split = [2,3],
                                   min_samples_leaf = [2,3],
                                   max_depth=[12,16]))


# In[198]:


#dt_criterion_search.fit(scaled_X_train, scaled_y_train)


# In[199]:


#dt_criterion_search.best_param_


# ### Post Pruning:

# In[137]:


hyperparameter_grid = {'ccp_alpha': np.linspace(0.0, 0.2, 10)}


# In[138]:


search = GridSearchCV(DecisionTreeRegressor(), 
                      hyperparameter_grid,
                      cv=10)


# In[140]:


#search.fit(scaled_X_train, scaled_y_train)


# In[200]:


#search.predict(scaled_X_train)
#print('accuracy for training dataset: %.5f'% search.score(scaled_X_train, scaled_y_train))
#pred_search = search.predict(scaled_X_test)
#print('accuracy for testing dataset: %.5f'% search.score(scaled_X_test, scaled_y_test))


# In[201]:


#print('MAE_PP:  %.5f'% mean_absolute_error(scaled_y_test, pred_search))
#print('MSE_PP: %.5f'% mean_squared_error(scaled_y_test, pred_search))
#print('RMSE_PP: %.5f'% np.sqrt(mean_squared_error(scaled_y_test, pred_search)))
#print('r2_score_PP:  %.5f'% r2_score(scaled_y_test, pred_search))


# ### Random Forest

# In[141]:


from sklearn.ensemble import RandomForestRegressor


# In[142]:


#Applying same 'best_params' from Decision Tree. Although a seperate hyper-tuning required for Random Forest
RF = RandomForestRegressor(criterion='friedman_mse',
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=2)


# In[143]:


RF.fit(scaled_X_train, scaled_y_train)
RF.predict(scaled_X_train)
print('accuracy for training dataset: %.5f'% RF.score(scaled_X_train, scaled_y_train))
pred_RF = RF.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f'% RF.score(scaled_X_test, scaled_y_test))


# In[144]:


print('MAE_RF:  %.5f'% mean_absolute_error(scaled_y_test, pred_RF))
print('MSE_RF:  %.5f'% mean_squared_error(scaled_y_test, pred_RF))
print('RMSE_RF: %.5f'% np.sqrt(mean_squared_error(scaled_y_test, pred_RF)))
print('r2_score_RF:  %.5f'% r2_score(scaled_y_test, pred_RF))


# In[149]:


dict_RF = {'MAE': 0.40185,'MSE': 0.59869,'RMSE': 0.77375,'r2_score': 0.77375}


# In[159]:


data_sizes, training_scores, validation_scores = learning_curve(RF,X = scaled_X_train, y =scaled_y_train,
                                                    cv=10,error_score=0, scoring='neg_mean_squared_error', train_sizes= np.array([0.1  , 0.325, 0.55 , 0.775, 1.]))


training_mean = - training_scores.mean(axis=1) 
training_standard_deviation = -training_scores.std(axis=1) 

validation_mean = - validation_scores.mean(axis=1) 
validation_standard_deviation =- validation_scores.std(axis=1)



import plotly.graph_objects as go


fig = go.Figure()

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean,
                        mode='lines',
                        name='Training',
                        line=dict(color='red')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean - training_standard_deviation,
                        mode='lines',
                        name='Training lower bound',
                        line=dict(width=0, color='red'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean + training_standard_deviation,
                        mode='lines',
                        name='Training upper bound',
                        line=dict(width=0, color='red'),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        showlegend=False))

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean,
                        mode='lines',
                        name='Validation',
                        line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean - validation_standard_deviation,
                        mode='lines',
                        name='Validation lower bound',
                        line=dict(width=0, color='blue'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean + validation_standard_deviation,
                        mode='lines',
                        name='Validation upper bound',
                        line=dict(width=0, color='blue'),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.3)',
                        showlegend=False))

fig.update_layout(title='Random Forest Learning curve',
                 xaxis_title='Dataset size',
                 yaxis_title='Accuracy')
fig.show()


# In[150]:


fig, [ax1,ax2, ax3] = plt.subplots(1,3, sharey=True, figsize=(12,6))
ax1.bar(*zip(*dicts_dt.items()), color = 'red')
ax1.set_title('Decision Tree')
ax1.grid()

ax2.bar(*zip(*dicts_knn.items()), color = 'blue')
ax2.set_title('KNN')
ax2.grid()

ax3.bar(*zip(*dict_RF.items()), color = 'green')
ax3.set_title('Random Forest')
ax3.grid()

fig.tight_layout()
plt.show()


# ### Linear Regression

# In[160]:


from sklearn.linear_model import LinearRegression


# In[161]:


LR = LinearRegression()


# In[162]:


LR.fit(scaled_X_train, scaled_y_train)
LR.predict(scaled_X_train)
print('accuracy for training dataset: %.5f' % LR.score(scaled_X_train, scaled_y_train))
pred_LR = LR.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f' % LR.score(scaled_X_test, scaled_y_test))


# In[163]:


print('MAE_LR:  %.5f' % mean_absolute_error(scaled_y_test, pred_LR))
print('MSE_LR:  %.5f' % mean_squared_error(scaled_y_test, pred_LR))
print('RMSE_LR: %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_LR)))
print('r2_score_LR:  %.5f' % r2_score(scaled_y_test, pred_LR))


# In[164]:


dicts_LR = {'MAE': 0.24754,'MSE' : 0.16078,'RMSE' : 0.40097,'r2_score' :  0.83922}


# In[157]:


data_sizes, training_scores, validation_scores = learning_curve(LR,X = scaled_X_train, y =scaled_y_train,
                                                    cv=10,error_score=0, scoring='neg_mean_squared_error', train_sizes= np.array([0.1  , 0.325, 0.55 , 0.775, 1.]))


training_mean = - training_scores.mean(axis=1) 
training_standard_deviation = -training_scores.std(axis=1) 

validation_mean = - validation_scores.mean(axis=1) 
validation_standard_deviation =- validation_scores.std(axis=1)



import plotly.graph_objects as go


fig = go.Figure()

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean,
                        mode='lines',
                        name='Training',
                        line=dict(color='red')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean - training_standard_deviation,
                        mode='lines',
                        name='Training lower bound',
                        line=dict(width=0, color='red'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=training_mean + training_standard_deviation,
                        mode='lines',
                        name='Training upper bound',
                        line=dict(width=0, color='red'),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        showlegend=False))

fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean,
                        mode='lines',
                        name='Validation',
                        line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean - validation_standard_deviation,
                        mode='lines',
                        name='Validation lower bound',
                        line=dict(width=0, color='blue'),
                        showlegend=False))
fig.add_trace(go.Scatter(x=data_sizes, 
                        y=validation_mean + validation_standard_deviation,
                        mode='lines',
                        name='Validation upper bound',
                        line=dict(width=0, color='blue'),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.3)',
                        showlegend=False))

fig.update_layout(title='Linear Regression Learning curve',
                 xaxis_title='Dataset size',
                 yaxis_title='Accuracy')
fig.show()


# ### Ridge Regression

# In[165]:


from sklearn.linear_model import Ridge


# In[166]:


Ridge1 = Ridge()


# In[167]:


Ridge1.fit(scaled_X_train, scaled_y_train)
Ridge1.predict(scaled_X_train)
print('accuracy for training dataset: %.5f' % Ridge1.score(scaled_X_train, scaled_y_train))
pred_Ridge = Ridge1.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f' % Ridge1.score(scaled_X_test, scaled_y_test))


# In[169]:


print('MAE_LR:  %.5f' % mean_absolute_error(scaled_y_test, pred_Ridge))
print('MSE_LR:  %.5f' % mean_squared_error(scaled_y_test, pred_Ridge))
print('RMSE_LR: %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_Ridge)))
print('r2_score_LR:  %.5f' % r2_score(scaled_y_test, pred_Ridge))


# #### Ridge Grid Search for Alpha

# In[170]:


Ridge_criterion_search = GridSearchCV(estimator=Ridge(), cv=10, 
                              param_grid=dict(alpha = [0.001,0.01,1,10,100]))


# In[171]:


Ridge_criterion_search.fit(scaled_X_train, scaled_y_train)


# In[172]:


Ridge_criterion_search.predict(scaled_X_train)
print('accuracy for training dataset: %.5f' % Ridge_criterion_search.score(scaled_X_train, scaled_y_train))
pred_Ridge_criterion_search = Ridge_criterion_search.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f' % Ridge_criterion_search.score(scaled_X_test, scaled_y_test))


# In[173]:


Ridge_criterion_search.best_params_


# In[175]:


print('MAE_Ridge:  %.5f' % mean_absolute_error(scaled_y_test, pred_Ridge_criterion_search))
print('MSE_Ridge:  %.5f' % mean_squared_error(scaled_y_test, pred_Ridge_criterion_search))
print('RMSE_Ridge: %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_Ridge_criterion_search)))
print('r2_score_Ridge:  %.5f' % r2_score(scaled_y_test, pred_Ridge_criterion_search))


# In[176]:


dicts_Ridge = {'MAE': 0.24754,'MSE' : 0.16078,'RMSE' : 0.40097, 'r2_score' :  0.83922}


# ### Lasso Regression

# In[177]:


from sklearn.linear_model import Lasso


# In[178]:


Lasso1 = Lasso()


# In[179]:


Lasso1.fit(scaled_X_train, scaled_y_train)
Lasso1.predict(scaled_X_train)
print('accuracy for training dataset: ', Lasso1.score(scaled_X_train, scaled_y_train))
pred_Lasso1 = Lasso1.predict(scaled_X_test)
print('accuracy for testing dataset: ', Lasso1.score(scaled_X_test, scaled_y_test))


# In[183]:


print('MAE_Lasso:  %.5f' % mean_absolute_error(scaled_y_test, pred_Lasso1))
print('MSE_Lasso:  %.5f' % mean_squared_error(scaled_y_test, pred_Lasso1))
print('RMSE_Lasso: %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_Lasso1)))
print('r2_score_Lasso: %.5f' % r2_score(scaled_y_test, pred_Lasso1))


# #### Lasso Grid Search for Alpha

# In[181]:


Lasso_criterion_search = GridSearchCV(estimator=Lasso(), cv=10, 
                              param_grid=dict(alpha = [0.000001, 0.00001, 0.0001, 0.001,0.01,1]))


# In[184]:


Lasso_criterion_search.fit(scaled_X_train, scaled_y_train)
Lasso_criterion_search.predict(scaled_X_train)
print('accuracy for training dataset: %.5f' % Lasso_criterion_search.score(scaled_X_train, scaled_y_train))
pred_Lasso_criterion_search = Lasso_criterion_search.predict(scaled_X_test)
print('accuracy for testing dataset: %.5f' % Lasso_criterion_search.score(scaled_X_test, scaled_y_test))


# In[186]:


Lasso_criterion_search.best_params_


# In[187]:


print('MAE_L:  %.5f' % mean_absolute_error(scaled_y_test, pred_Lasso_criterion_search))
print('MSE_L:  %.5f' % mean_squared_error(scaled_y_test, pred_Lasso_criterion_search))
print('RMSE_L: %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_Lasso_criterion_search)))
print('r2_score_L:  %.5f' % r2_score(scaled_y_test, pred_Lasso_criterion_search))


# In[188]:


LassoB = Lasso(alpha = 1e-05)
LassoB.fit(scaled_X_train, scaled_y_train)
LassoB.predict(scaled_X_train)
print('accuracy for training dataset: ', LassoB.score(scaled_X_train, scaled_y_train))
pred_LassoB = LassoB.predict(scaled_X_test)
print('accuracy for testing dataset: ', LassoB.score(scaled_X_test, scaled_y_test))
print('MAE_LB:  %.5f' % mean_absolute_error(scaled_y_test, pred_LassoB))
print('MSE_LB:  %.5f' % mean_squared_error(scaled_y_test, pred_LassoB))
print('RMSE_LB: %.5f' % np.sqrt(mean_squared_error(scaled_y_test, pred_LassoB)))
print('r2_score_LB: %.5f' % r2_score(scaled_y_test, pred_LassoB))


# In[189]:


dicts_Lasso = {'MAE': 0.24754,'MSE' : 0.16078,'RMSE' : 0.40097,'r2_score' :  0.83922}


# ### Comparison among the various Learning Algorithms

# In[193]:


fig, [(ax1,ax2), (ax3,ax4)] = plt.subplots(2,2, sharey=True, figsize=(12,12))

ax1.bar(*zip(*dicts_dt.items()), color = 'red')
ax1.set_title('Decision Tree')
ax1.grid()

ax2.bar(*zip(*dicts_knn.items()), color = 'blue')
ax2.set_title('KNN')
ax2.grid()

ax3.bar(*zip(*dict_RF.items()), color = 'green')
ax3.set_title('Random Forest')
ax3.grid()

ax4.bar(*zip(*dicts_LR.items()), color = 'orange')
ax4.set_title('Linear Rigression/ Ridge/ Lasso')
ax4.grid()


fig.tight_layout()
plt.show()


# ### Thank You!
