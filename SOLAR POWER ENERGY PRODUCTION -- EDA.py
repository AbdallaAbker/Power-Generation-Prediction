#!/usr/bin/env python
# coding: utf-8

# # Predicting Solar Energy Production
# 
# ## Capstone Project  -- EDA
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


# ### Thank you!
