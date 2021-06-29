# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:11:31 2021

@author: Emanuele D'Argenzio
"""
# =============================================================================
# Forecasting model for the energy prices in Italy
# =============================================================================

# ====================================================================================================================
# YOU CAN RAN THE TOTAL CODE WITH NO PROBLEM, BUT THE DASH SECTION MUST BE RUNNED ALONE (BECAUSE OF AN UNSOLVED ERROR)
# ====================================================================================================================

#%% Clear variables
import sys
sys.modules[__name__].__dict__.clear()

#%% Import library
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import holidays
from sklearn.cluster import KMeans

#%%
# =============================================================================
# #Data uploading and cleaning
# =============================================================================

#Prices Data uploading
price_2013=pd.read_excel('Anno 2013.xlsx',sheet_name='Prezzi-Prices',usecols=['Date_corr','Hour_corr','Date_corr Hour_corr','NORD'])
price_2014=pd.read_excel('Anno 2014.xlsx',sheet_name='Prezzi-Prices',usecols=['Date_corr','Hour_corr','Date_corr Hour_corr','NORD'])
price_2015=pd.read_excel('Anno 2015.xlsx',sheet_name='Prezzi-Prices',usecols=['Date_corr','Hour_corr','Date_corr Hour_corr','NORD'])
price_2016=pd.read_excel('Anno 2016.xlsx',sheet_name='Prezzi-Prices',usecols=['Date_corr','Hour_corr','Date_corr Hour_corr','NORD']) #2016 was a leap year, it has 1 day more
price_2017=pd.read_excel('Anno 2017.xlsx',sheet_name='Prezzi-Prices',usecols=['Date_corr','Hour_corr','Date_corr Hour_corr','NORD'])
price_2018=pd.read_excel('Anno 2018.xlsx',sheet_name='Prezzi-Prices',usecols=['Date_corr','Hour_corr','Date_corr Hour_corr','NORD'])
price_2019=pd.read_excel('Anno 2019.xlsx',sheet_name='Prezzi-Prices',usecols=['Date_corr','Hour_corr','Date_corr Hour_corr','NORD'])

#%%Info printing

print(price_2013.info())
print(price_2014.info())
print(price_2015.info())
print(price_2016.info())
print(price_2017.info())
print(price_2018.info())
print(price_2019.info())

#%% Remove NaN from the 2013 price vector (due to an error on the excel)
price_2013=price_2013.dropna() 

#%% Price data merging

price_nord_all=pd.merge(price_2013,price_2014,how='outer')
price_nord_all=pd.merge(price_nord_all,price_2015,how='outer')
price_nord_all=pd.merge(price_nord_all,price_2016,how='outer')
price_nord_all=pd.merge(price_nord_all,price_2017,how='outer')
price_nord_all=pd.merge(price_nord_all,price_2018,how='outer')
price_nord_all=pd.merge(price_nord_all,price_2019,how='outer')

#%%
print(price_nord_all.info())

#%%
#delete the initial data frames to clean the output
del price_2013
del price_2014
del price_2015
del price_2016
del price_2017
del price_2018
del price_2019

#rename columns for the electricity prices
price_nord_all.rename(columns={'Hour_corr':'Hour','Date_corr':'Date','Date_corr Hour_corr':'Date Complete'},inplace=True)
print(price_nord_all.info())

#%%Holiday Dataframe Creation
it_holiday=[]
for date in holidays.Italy(years=[2013,2014,2015,2016,2017,2018,2019]).items():
    it_holiday.append(str(date[0]))

print(it_holiday)
    
#%% Date dataframe creation
Date_new=pd.DataFrame()      
Date_new['Dates']=pd.date_range(start="2013-01-01",end="2020-01-01")   #creation of the dates

#%% If holiday is in the date, put 1. Otherwise put 0
Date_new['Holiday']=[
   1 if str(val).split()[0] in it_holiday else 0 for val in Date_new['Dates']
   ]

print(Date_new)

#%% Create date dataframe with 1h timestep
Date=pd.date_range(start="2013-01-01",end="2020-01-01",freq='H').to_frame(index=False)   #creation of the dates
mapping = {Date.columns[0]:'Date'}                                                       #value assignment
Date=Date.rename(columns=mapping)                                                        #column rename
del mapping                                        #not needed anymore
Date['Hours']=Date['Date'].dt.hour                 #extract hours from date
Date['Day']=Date['Date'].dt.day                    #extract day from date
Date['Week Day']=Date['Date'].dt.weekday           #extract weekday from date
Date['Month']=Date['Date'].dt.month                #extract month from date
Date['Year']=Date['Date'].dt.year                  #extract year from date

print(Date.info())
#%% Resample the dataframe of the holiday in 1h time step
Date_new=Date_new.set_index('Dates')
Date_new=Date_new.resample('1H').ffill(23)
Date_new=Date_new.fillna(0)

Date_new=Date_new.reset_index()
Date_new.rename(columns={'Dates':'Date'},inplace=True)

#%% Merge data dataframe with holiday dataframe
Date=pd.merge(Date,Date_new,on='Date')
print(Date.info())

#%% Uploading temperature weather datas

#I was looking for downloading those datas with an by using an API, but unfortunately with the account
# I was able to download 500 datas per time, respect to the 60'000 data needed. So I found an hystorical
#  weather database for the city of Milan

############################################################################
##HERE the API i was looking to use

# import requests

# url = "https://visual-crossing-weather.p.rapidapi.com/history"

# querystring = {"startDateTime":"2013-01-01T00:00:00","aggregateHours":"1","location":"Milan,MI,IT","endDateTime":"2019-12-31T00:00:00","unitGroup":"metric","contentType":"csv","shortColumnNames":"false"}

# headers = {
#     'x-rapidapi-key': "c413aec41dmsh9906401275d6fedp186f95jsnf62ef2f8052a",
#     'x-rapidapi-host': "visual-crossing-weather.p.rapidapi.com"
#     }

############################################################################
#these data have been downloaded from https://www.arpalombardia.it/Pages/Meteorologia/Richiesta-dati-misurati/Guida-richiesta-dati.aspx
#it is a regional agency for the environment protection (public administration body)
  
raw_temperature=pd.read_csv('Temperature_Milan.csv')
raw_irradiance=pd.read_csv('Solar_Irradiance_Milan.csv')
weather_df=pd.merge(raw_temperature,raw_irradiance,on='Data-Ora')
weather_df.rename(columns={'Data-Ora':'Date',' Medio_x':'Temperature [°C]',' Medio_y':'Irradiance [W/m2]'},inplace=True)
weather_df=weather_df.drop(columns=['Id Sensore_x','Id Sensore_y'])

print(weather_df.info())

#delete the df not needed anymore
del raw_temperature
del raw_irradiance

weather_df['Date']=pd.to_datetime(weather_df['Date'], format='%Y/%m/%d %H:%M') #Transform the date into dataframe

#%% Remove the first raw from date and weather, since it is not present in price nord and reset the correct index
Date.drop(Date.index[0], inplace=True)
Date.index=Date.index -1  #set the index in the right position
print(Date.info())

weather_df.drop(weather_df.index[0], inplace=True)
weather_df.index=weather_df.index-1
print(weather_df.info())

#since the price north has a date that is impossible to convert in the format YYYYDDMM H:M:S, i'm adding to the price the date vector
price_nord_all['Date_correct']=Date['Date']

del price_nord_all['Hour']
del price_nord_all['Date']
del price_nord_all['Date Complete']
price_nord_all.rename(columns={'Date_correct':'Date', 'NORD':'Price [€/MWh]'},inplace=True)
print(price_nord_all.info())
#in this way, also the dates are the same. For sure, before continuing the data analysis, i checked that everything was fine.

#%% Merging all the datas in an unique dataframe
raw_data_all=pd.merge(price_nord_all,Date,on='Date')
raw_data_all=pd.merge(raw_data_all,weather_df,on='Date')

#%%Data cleaning with the data provider indication

print(raw_data_all.info())
print(raw_data_all.columns) #there is no value that is null.

#To be noticed: the data provider for the weather says that each time there is a value of temperature or  equal to -999, 
#it means that this value is an NaN. So it is needed to remove it.

raw_data_all['Temperature [°C]'] = raw_data_all['Temperature [°C]'].replace(-999, np.nan)
raw_data_all['Irradiance [W/m2]'] = raw_data_all['Irradiance [W/m2]'].replace(-999, np.nan)

raw_data_all=raw_data_all.dropna()
print(raw_data_all.info())  #less then 1000 datas has been removed

#%%
# =============================================================================
# Exploratory Data Analysis
# =============================================================================

# Price vs temperature
x  = (raw_data_all['Date'])              
y1 = (raw_data_all['Price [€/MWh]'])       
y2 = (raw_data_all['Temperature [°C]'])  

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price [€/MWh]', color='g')
ax2.set_ylabel('Temperature [°C]', color='b')
plt.title('Prices and Temperature profiles')
plt.show()


#%% Price vs temerature zoomed
x  = (raw_data_all.iloc[1100:1123]['Date'])              
y1 = (raw_data_all.iloc[1100:1123]['Price [€/MWh]'])       
y2 = (raw_data_all.iloc[1100:1123]['Temperature [°C]'])  

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price [€/MWh]', color='g')
ax2.set_ylabel('Temperature [°C]', color='b')
plt.title('Prices and Temperature profiles - Zoomed')
plt.show()

#%% Price vs irradiance
x  = (raw_data_all['Date'])              
y2 = (raw_data_all['Price [€/MWh]'])        
y1 = (raw_data_all['Irradiance [W/m2]']) 

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, y1, 'c-')
ax2.plot(x, y2, 'm-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Irradiance [W/m2]', color='m')
ax2.set_ylabel('Price [€/MWh]', color='c')
plt.title('Irradiance and Price profiles')
plt.show()

#%% Price vs irradiance

x  = (raw_data_all.iloc[1100:1123]['Date'])              
y2 = (raw_data_all.iloc[1100:1123]['Price [€/MWh]'])        
y1 = (raw_data_all.iloc[1100:1123]['Irradiance [W/m2]']) 

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, y1, 'c-')
ax2.plot(x, y2, 'm-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Irradiance [W/m2]', color='m')
ax2.set_ylabel('Price [€/MWh]', color='c')
plt.title('Irradiance and Price profiles')
plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DATA CLEANING 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Plot to check for points that could be deleted
import matplotlib.ticker as ticker                             # import a special package

# Price plot
fig, ax = plt.subplots()                                       # create objects of the plot (figure and plot inside)
fig.set_size_inches(20,10)                                     # define figure size

ax.xaxis.set_major_locator (ticker.MultipleLocator(60)) 
ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)

plt.plot (raw_data_all['Price [€/MWh]'], '-o', color = 'blue', 
         markersize = 10, linewidth = 1, 
         markerfacecolor = 'red', 
         markeredgecolor = 'black', 
         markeredgewidth = 3)
plt.xlabel('Date')
plt.ylabel('Prices[€/MWh]')
plt.title('Electricity prices')

#There are some outliers, i will delete them in the next part

#%%Temperature plot

fig, ax = plt.subplots()                   # create objects of the plot (figure and plot inside)
fig.set_size_inches(20,10)                 # define figure size

ax.xaxis.set_major_locator (ticker.MultipleLocator(60))
   
ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
plt.plot (raw_data_all['Temperature [°C]'], '-o', color = 'cyan',
         markersize = 10, linewidth = 1, 
         markerfacecolor = 'yellow', 
         markeredgecolor = 'black', 
         markeredgewidth = 3)
plt.xlabel('Date')
plt.ylabel('Temperature [°C]')
plt.title('Temperature profile over the years')

#it seems there are not outliers

#%%Irradiance plot

fig, ax = plt.subplots()                   # create objects of the plot (figure and plot inside)
fig.set_size_inches(20,10)                 # define figure size

ax.xaxis.set_major_locator (ticker.MultipleLocator(60))
   
ax.xaxis.set_tick_params (which = 'major', pad = 5, labelrotation = 50)
plt.plot (raw_data_all['Irradiance [W/m2]'], '-o', color = 'cyan',
         markersize = 10, linewidth = 1, 
         markerfacecolor = 'yellow', 
         markeredgecolor = 'black', 
         markeredgewidth = 3)
plt.xlabel('Date')
plt.ylabel('Irradiance [W/m2]')
plt.title('Irradiance profile over the years')

#it seems there are not strange temperatures coming from error in the measurement

#%% Boxplot and hystogram for the Electricity prices
import seaborn as sns
sns.set(style="ticks")
x=raw_data_all['Price [€/MWh]']

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.10, .90)})
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True) 
plt.show()

#%%Boxplot and hystogram for the temperature
sns.set(style="ticks")
x=raw_data_all['Temperature [°C]']

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.10, .90)})
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True) 
plt.show()

#%%Boxplot and hystogram for the irradiance
sns.set(style="ticks")
x=raw_data_all['Irradiance [W/m2]']

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.10, .90)})
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True) 
plt.show()

#%% Clean the prices profile with z score

from scipy import stats
z = np.abs(stats.zscore(raw_data_all['Price [€/MWh]']))
print(z)

threshold = 3 # 3 sigma...Includes 99.7% of the data
print(z>3)
print(np.where(z > 3))
raw_data_all=raw_data_all[(z < 3)]  #cleaning the outliers

#%% Boxplot and hystogram for the Electricity prices cleaned
import seaborn as sns
sns.set(style="ticks")
x=raw_data_all['Price [€/MWh]']

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.10, .90)})
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True) 
plt.show()

#%%

# =============================================================================
# Clustering
# =============================================================================

# create kmeans object
raw_data_all=raw_data_all.set_index('Date')                     #set date as index to perform the clusering
Solar_Rad = raw_data_all.pop('Irradiance [W/m2]')               #For the clustering, the solar irradiance gives problems
model = KMeans(n_clusters=2).fit(raw_data_all)                  #creation of a model based on KMeans with 2 clusters using raw_all_data variable
pred = model.labels_
print(pred)                                                     #each value is associated to a different cluster
#this method does not tell so much

Nc = range(1, 20)                                             #test of different number of clusters 
kmeans = [KMeans(n_clusters=i) for i in Nc]                   #try from 1 to 20 clusters
score = [kmeans[i].fit(raw_data_all).score(raw_data_all) for i in range(len(kmeans))]  #score creation
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')                                      #the graph is used to understand the ideal number of clusters 
                                                              #for the datas under consideration

raw_data_all['Cluster']=pred                                  #create a new column that states the cluster for each data
#print(raw_data_all)

#%%Plotting the results of the clustering

#instead of tab20, i can put the colormap I can find in this site https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

ax1=raw_data_all.plot.scatter(x='Price [€/MWh]',y='Temperature [°C]',color=[plt.cm.tab20(float(i) /10) for i in raw_data_all['Cluster']])   #clustering price vs temperature
plt.xlabel('Price [€/MWh]')
plt.ylabel('Temperature[°C]')
plt.title('Price clustered with respect to the temperature')
ax2=raw_data_all.plot.scatter(x='Price [€/MWh]',y='Hours',color=[plt.cm.tab20(float(i) /10) for i in raw_data_all['Cluster']])              #clustering price vs hours
plt.xlabel('Price [€/MWh]')
plt.ylabel('Hours')
plt.title('Hourly Price profile clustered')
ax3=raw_data_all.plot.scatter(x='Price [€/MWh]',y='Week Day',color=[plt.cm.tab20(float(i) /10) for i in raw_data_all['Cluster']])           #clustering price vs week day
plt.xlabel('Price [€/MWh]')
plt.ylabel('Week Day')
plt.title('Price clustered with respect to the week day')
ax4=raw_data_all.plot.scatter(x='Hours',y='Price [€/MWh]',color=[plt.cm.tab20(float(i) /10) for i in raw_data_all['Cluster']])              #clustering hours vs price
plt.xlabel('Hours')
plt.ylabel('Price [€/MWh]')
plt.title('Hourly Price profile clustered (1)')

#%% Daily pattern identification
df=raw_data_all
df=df.drop(columns=['Temperature [°C]','Day','Week Day','Month','Year','Cluster','Holiday']) #remove columns that are not needed: since we are looking for the price pattern, remove everything that is not price
print(df)

#To create a pivot table it is needed to remove the hours from the date
df=df.reset_index()
df['Date']=pd.to_datetime(df['Date']).dt.date
df=df.set_index('Date')
df_pivot = df.pivot(columns='Hours')
df_pivot = df_pivot.dropna()
print(df_pivot)
df_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02)        #T is used to transpose
plt.xlabel('Hours')
plt.ylabel('Price [€/MWh]')
plt.title('Hourly prices')
plt.show()
#in this way, the matrix has been reshaped obtaining the price divided by hours per each day -> price curve can be represented thanks to that

#%%Identification of the clusters for price pattern
from sklearn.preprocessing import MinMaxScaler  #library used to scale datas
from sklearn.metrics import silhouette_score

sillhoute_scores = [] 
n_cluster_list = np.arange(2,10).astype(int)    #there will be from 2 to 10 clusters, integer values

X = df_pivot.values.copy()                      #copy pivot table to then scale it
    
sc = MinMaxScaler()     
X = sc.fit_transform(X)                         #scaling of the pivot table 

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
    
plt.plot(n_cluster_list,sillhoute_scores)
plt.title('Sillhoute score')
plt.xlabel('Cluster')
plt.ylabel('Score')
plt.show()
#From the sillhoute score, Better to use 2 clusters 

#%% Taking the result from the previous section to obtain the clusters and plotting the clusters on the previous graph

kmeans = KMeans(n_clusters=2)                   #in n_cluster i putted the value obtained fror the sillhoute score -> 2
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster') 
df_pivot = df_pivot.set_index(cluster_found_sr, append=True)

print(df_pivot)

fig, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(
        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}'
        )
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'
    )
ax.set_ylabel('Prices [€/MWh]')
plt.title('Clustering result')
fig.savefig('assets/Clustering/Daily Pattern.png', dpi=fig.dpi)
plt.show()
#plt.savefig('assets/Clustering/Daily Pattern.png')

#%% Extract the clustered curves to be used for the daily pattern identification in the dashboard
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
df_price_pattern=[]
for cluster, color in zip(cluster_values, color_list):
    df_price_pattern.append(df_pivot.xs(cluster, level=1).median())

df_price_pattern_0=df_price_pattern[0]
df_price_pattern_0=df_price_pattern_0.reset_index()
del df_price_pattern_0['level_0']

df_price_pattern_1=df_price_pattern[1]
df_price_pattern_1=df_price_pattern_1.reset_index()
del df_price_pattern_1['level_0']

#%%
# =============================================================================
# Feature Selection
# =============================================================================
raw_data_all=raw_data_all.join(Solar_Rad)
raw_data_feature=raw_data_all

raw_data_feature=np.around(raw_data_feature, decimals=2)
raw_data_feature['Price-1 [€/MWh]']=raw_data_feature['Price [€/MWh]'].shift(1)
raw_data_feature['day2']=np.square(raw_data_feature['Week Day'])                        #squaring the day of the week
raw_data_feature=np.around(raw_data_feature, decimals=2)                                #round to the second decimal
raw_data_feature=raw_data_feature.dropna()                                              #drop NaN

#%%
print(raw_data_feature.info())

#  #   Column             Non-Null Count  Dtype  
# ---  ------             --------------  -----  
#  0   Price [€/MWh]      60792 non-null  float64
#  1   Hours              60792 non-null  int64  
#  2   Day                60792 non-null  int64  
#  3   Week Day           60792 non-null  int64  
#  4   Month              60792 non-null  int64  
#  5   Year               60792 non-null  int64  
#  6   Holiday            60792 non-null  int64  
#  7   Temperature [°C]   60792 non-null  float64
#  8   Cluster            60792 non-null  int32  
#  9   Irradiance [W/m2]  60792 non-null  float64
#  10  Price-1 [€/MWh]    60792 non-null  float64
#  11  day2               60792 non-null  int64  

#%% Input and output definition for the model creation
X=raw_data_feature.values                     #remove the index and create an array 
 
Y=X[:,0]                                     #The prices are the output of the model -> it is positioned in column 0
X=X[:,[1,2,3,4,5,6,7,8,9,10,11]]             #All the other columns are input, it has no sense to put price as input of the model
print(Y)
print(X)

#%%
from sklearn.feature_selection import SelectKBest                              #selection method - kbest
from sklearn.feature_selection import f_regression, mutual_info_regression     #score metric

#%% f_regression - KBEST
features=SelectKBest(k=2,score_func=f_regression)              #create a variable using selectkbest with K features and fregression
fit=features.fit(X,Y)                                          #calculate the correlation between features and output
print(fit.scores_)                                             #score printing
features_results=fit.transform(X)                           
print(features_results)           
plt.title('Feature to select - f regression')
plt.xlabel('Feature')
plt.bar([i for i in range(len(fit.scores_))],fit.scores_)      #The inputs that most affect the the model are: 
plt.show()                                                     #Cluster, price-1 and Hours
                                                               #results can be checked in the bar plot
                                                               
#put the results in a dataframe to then visualize it in the dash
F_reg_score=pd.DataFrame(fit.scores_)
F_reg_score=F_reg_score.transpose()
F_reg_score=F_reg_score.rename(columns={0:'Hours',1:'Day',2:'Week Day',3:'Month',4:'Year',5:'Holiday',6:'Temperature [°C]',7:'Cluster',8:'Irradiance [W/m2]',9:'Price-1 [€/MWh]',10:'day2'})

#%% mutual_info_regression - KBEST
features=SelectKBest(k=4,score_func=mutual_info_regression)       #create a variable using selectkbest with K features and mutual_info_regression
fit=features.fit(X,Y)                                             #calculate the correlation between features and output
print(fit.scores_)                                                #score printing
features_results=fit.transform(X)  
plt.title('Feature to select - Mutual info regression')
plt.xlabel('Feature')                         
print(features_results)                                      
plt.bar([i for i in range(len(fit.scores_))],fit.scores_)         #The inputs that most affect th model are:
plt.show()                                                        #Price-1, Cluster, Year, Month and Hours 

#  #   Column             Non-Null Count  Dtype  
# ---  ------             --------------  -----  
#  0   Price [€/MWh]      60792 non-null  float64
#  1   Hours              60792 non-null  int64  
#  2   Day                60792 non-null  int64  
#  3   Week Day           60792 non-null  int64  
#  4   Month              60792 non-null  int64  
#  5   Year               60792 non-null  int64  
#  6   Holiday            60792 non-null  int64  
#  7   Temperature [°C]   60792 non-null  float64
#  8   Cluster            60792 non-null  int32  
#  9   Irradiance [W/m2]  60792 non-null  float64
#  10  Price-1 [€/MWh]    60792 non-null  float64
#  11  day2               60792 non-null  int64  

# [0.15196826 0.08889795 0.06899452 0.20300251 0.24100584 0.00930919
#  0.03909627 0.62863775 0.02999161 1.22161143 0.06876455]

#put the results in a dataframe to then visualize it in the dash
F_mut_info_reg_score=pd.DataFrame(fit.scores_)
F_mut_info_reg_score=F_mut_info_reg_score.transpose()
F_mut_info_reg_score=F_mut_info_reg_score.rename(columns={0:'Hours',1:'Day',2:'Week Day',3:'Month',4:'Year',5:'Holiday',6:'Temperature [°C]',7:'Cluster',8:'Irradiance [W/m2]',9:'Price-1 [€/MWh]',10:'day2'})

#%% Wrapper methods - linear regression

#Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression       #linear model

model=LinearRegression()                                #LinearRegression Model as Estimator
rfe=RFE(model,n_features_to_select=2)                   #using 2 features
rfe2=RFE(model,n_features_to_select=3)                  #using 3 features
rfe3=RFE(model,n_features_to_select=1)                  #using 1 features

fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)
fit3=rfe3.fit(X,Y)
#depending on the number of features used for the model, different results

print( "Feature Ranking (Linear Model, 1 features): %s" % (fit3.ranking_))     #[11 8 4 7 5 3 9 1 10 2 6] -> Cluster is the most important feature
print( "Feature Ranking (Linear Model, 2 features): %s" % (fit.ranking_))      #[10 7 3 6 4 2 8 1 9 1 5] -> Price-1 and cluster are the most important features
print( "Feature Ranking (Linear Model, 3 features): %s" % (fit2.ranking_))     #[9 6 2 5 3 1 7 1 8 1 4] -> Price-1, cluster and holiday are the most important features

#depending on the number of features used for the model, different results
#this method directly gives the rank of the features by using a model

#since it uses linear regression as a model, it is weak 
#(it was used only to see different results for different models)

#%%Emsemble methods - RANDOM FOREST 
from sklearn.ensemble import RandomForestRegressor   #random forest regressor
model = RandomForestRegressor()
model.fit(X,Y)
print(model.feature_importances_) 
plt.title('Features to select - RANDOM FOREST')
plt.bar([i for i in range(len(model.feature_importances_))],model.feature_importances_)  
plt.show()                                                          # [0.03925388 0.00930371 0.00383803 0.00835744 0.01068362 0.00069125
                                                                    #  0.01593355 0.59430607 0.01002096 0.30380541 0.00380607]
                                         # -> Price-1, cluster, hour, Temperature, Year
#  #   Column             Non-Null Count  Dtype  
# ---  ------             --------------  -----  
#  0   Price [€/MWh]      60792 non-null  float64
#  1   Hours              60792 non-null  int64  
#  2   Day                60792 non-null  int64  
#  3   Week Day           60792 non-null  int64  
#  4   Month              60792 non-null  int64  
#  5   Year               60792 non-null  int64  
#  6   Holiday            60792 non-null  int64  
#  7   Temperature [°C]   60792 non-null  float64
#  8   Cluster            60792 non-null  int32  
#  9   Irradiance [W/m2]  60792 non-null  float64
#  10  Price-1 [€/MWh]    60792 non-null  float64
#  11  day2               60792 non-null  int64 
                                            
#comments: random forest has been used after computing the first two models (linear regression and kbest)
#it is one of the best method to be used

#CHOOSEN FEATURES: As a result, i choose : Price-1, Cluster, Hour, WeekDay and holiday.

random_forest_score=pd.DataFrame(model.feature_importances_)
random_forest_score=random_forest_score.transpose()
random_forest_score=random_forest_score.rename(columns={0:'Hours',1:'Day',2:'Week Day',3:'Month',4:'Year',5:'Holiday',6:'Temperature [°C]',7:'Cluster',8:'Irradiance [W/m2]',9:'Price-1 [€/MWh]',10:'day2'})

#%% Drop the column not needed 
raw_data_feature_selected=raw_data_feature.drop(columns=['Day','Month','Year','Temperature [°C]','Irradiance [W/m2]','day2'])
  
#The selected features are: Hours,Week Day,Holiday,Cluster,Price-1 [€/MWh]

#%%
# =============================================================================
# REGRESSION MODEL FOR ALL THE DATAS
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn import  metrics

print(raw_data_feature_selected.info())                                      #used to look at different columns

#  #   Column           Non-Null Count  Dtype  
# ---  ------           --------------  -----  
#  0   Price [€/MWh]    60792 non-null  float64
#  1   Hours            60792 non-null  int64  
#  2   Week Day         60792 non-null  int64  
#  3   Holiday          60792 non-null  int64  
#  4   Cluster          60792 non-null  int32  
#  5   Price-1 [€/MWh]  60792 non-null  float64

#%% Input and output definition for the model creation
X=raw_data_feature_selected.values
Y=X[:,0]                 #[Price]
X=X[:,[1,2,3,4,5]]

#%%
X_train, X_test, y_train, y_test = train_test_split(X,Y)        #this function is a random generation function: each run different results
print(X_train) 
print(y_train)

#%%RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

#create a function called parameters with all the options
parameters = {'bootstrap': True,                   #i changed some of the parameters
              'min_samples_leaf': 3,               #to try to obtain the best result
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'log2',              #I used log2 instead of sqrt, better results
              #'max_depth': 20,                    #I removed the max depth limit 
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)

RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

plt.title('Random forest')
plt.plot(y_test[1000:1200])
plt.plot(y_pred_RF[1000:1200])
plt.ylabel('Price [€/MWh]')
plt.xlabel('Hour index')
plt.legend(['y_test', 'y_pred_RF']);
plt.show()

plt.title('Random forest')
plt.xlabel('y test')
plt.ylabel('y pred RF')  
plt.scatter(y_test,y_pred_RF)
plt.show()

#Error evaluation
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

# 2.860787705940392 20.876909495342105 4.56912568171878 0.08566554211899113

#df creation for the dash visualization 
df_0_RF=pd.DataFrame(y_pred_RF,y_test)
df_0_RF=df_0_RF.reset_index()
df_0_RF=df_0_RF.rename(columns={'index':'Test Prices [€/MWh]',0:'Predicted Prices [€/MWh]'})

#%%Extreme gradient boosting
from xgboost import XGBRegressor

XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)

plt.title('Extreme Gradient boosting')
plt.plot(y_test[1000:1200])
plt.plot(y_pred_XGB[1000:1200])
plt.ylabel('Price [€/MWh]')
plt.xlabel('Hour index')
plt.legend(['y_test', 'y_pred_XGB']);
plt.show()

plt.title('Extreme Gradient boosting')
plt.xlabel('y test')
plt.ylabel('y pred XGB') 
plt.scatter(y_test,y_pred_XGB)
plt.show()

MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB)

# 2.8685758983453553 21.004356927819938 4.583051050099697 0.08592662581741747

#df creation for the dash visualization 
df_0_XGB=pd.DataFrame(y_pred_XGB,y_test)
df_0_XGB=df_0_XGB.reset_index()
df_0_XGB=df_0_XGB.rename(columns={'index':'Test Prices [€/MWh]',0:'Predicted Prices [€/MWh]'})

#%%
# =============================================================================
# REGRESSION MODEL FOR THE FIRST CLUSTER - LOW PRICES
# =============================================================================
raw_data_feature_selected_cluster_0=raw_data_feature_selected
raw_data_feature_selected_cluster_0['Cluster'] = raw_data_feature_selected_cluster_0['Cluster'].replace(0, np.nan)
raw_data_feature_selected_cluster_0=raw_data_feature_selected_cluster_0.dropna()

X=raw_data_feature_selected_cluster_0.values
Y=X[:,0]                 #[Price]
X=X[:,[1,2,3,5]]

#%%
X_train, X_test, y_train, y_test_1 = train_test_split(X,Y)        #this function is a random generation function: each run different results
print(X_train) 
print(y_train)

#%%RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

#create a function called parameters with all the options
parameters = {'bootstrap': True,                   #i changed some of the parameters
              'min_samples_leaf': 3,               #to try to obtain the best result
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'log2',              #I used log2 instead of sqrt, better results
              #'max_depth': 20,                    #I removed the max depth limit 
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)

RF_model.fit(X_train, y_train)
y_pred_RF_1 = RF_model.predict(X_test)

plt.title('Random forest')
plt.plot(y_test_1[1000:1200])
plt.plot(y_pred_RF_1[1000:1200])
plt.ylabel('Price [€/MWh]')
plt.xlabel('Hour index')
plt.legend(['y_test', 'y_pred_RF']);
plt.show()

plt.title('Random forest')
plt.xlabel('y test')
plt.ylabel('y pred RF')  
plt.scatter(y_test_1,y_pred_RF_1)
plt.show()

#Error evaluation
MAE_RF_1=metrics.mean_absolute_error(y_test_1,y_pred_RF_1) 
MSE_RF_1=metrics.mean_squared_error(y_test_1,y_pred_RF_1)  
RMSE_RF_1= np.sqrt(metrics.mean_squared_error(y_test_1,y_pred_RF_1))
cvRMSE_RF_1=RMSE_RF_1/np.mean(y_test_1)
print(MAE_RF_1,MSE_RF_1,RMSE_RF_1,cvRMSE_RF_1)

# 2.115524903248258 9.095721277349147 3.0159113510428566 0.07036378457622536

#df creation for the dash visualization 
df_1_RF=pd.DataFrame(y_pred_RF_1,y_test_1)
df_1_RF=df_1_RF.reset_index()
df_1_RF=df_1_RF.rename(columns={'index':'Test Prices [€/MWh]',0:'Predicted Prices [€/MWh]'})

#%%Extreme gradient boosting
from xgboost import XGBRegressor

XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB_1 =XGB_model.predict(X_test)

plt.title('Extreme Gradient boosting')
plt.plot(y_test_1[1000:1200])
plt.plot(y_pred_XGB_1[1000:1200])
plt.ylabel('Price [€/MWh]')
plt.xlabel('Hour index')
plt.legend(['y_test', 'y_pred_XGB']);
plt.show()

plt.title('Extreme Gradient boosting')
plt.xlabel('y test')
plt.ylabel('y pred XGB') 
plt.scatter(y_test_1,y_pred_XGB_1)
plt.show()

MAE_XGB_1=metrics.mean_absolute_error(y_test_1,y_pred_XGB_1) 
MSE_XGB_1=metrics.mean_squared_error(y_test_1,y_pred_XGB_1)  
RMSE_XGB_1= np.sqrt(metrics.mean_squared_error(y_test_1,y_pred_XGB_1))
cvRMSE_XGB_1=RMSE_XGB_1/np.mean(y_test_1)
print(MAE_XGB_1,MSE_XGB_1,RMSE_XGB_1,cvRMSE_XGB_1)

# 2.1118287154108706 9.011493697853915 3.0019150051015626 0.0700372378856779

#df creation for the dash visualization 
df_1_XGB=pd.DataFrame(y_pred_XGB_1,y_test_1)
df_1_XGB=df_1_XGB.reset_index()
df_1_XGB=df_1_XGB.rename(columns={'index':'Test Prices [€/MWh]',0:'Predicted Prices [€/MWh]'})

#%%
# =============================================================================
# REGRESSION MODEL FOR THE SECOND CLUSTER - HIGH PRICES
# =============================================================================
del raw_data_feature_selected
raw_data_feature_selected=raw_data_feature.drop(columns=['Day','Month','Year','Temperature [°C]','Irradiance [W/m2]','day2'])

raw_data_feature_selected_cluster_1=raw_data_feature_selected
raw_data_feature_selected_cluster_1['Cluster'] = raw_data_feature_selected_cluster_1['Cluster'].replace(1, np.nan)
raw_data_feature_selected_cluster_1=raw_data_feature_selected_cluster_1.dropna()

X=raw_data_feature_selected_cluster_1.values
Y=X[:,0]                 #[Price]
X=X[:,[1,2,3,5]]

#%%
X_train, X_test, y_train, y_test_2 = train_test_split(X,Y)        #this function is a random generation function: each run different results
print(X_train) 
print(y_train)

#%%RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

#create a function called parameters with all the options
parameters = {'bootstrap': True,                   #i changed some of the parameters
              'min_samples_leaf': 3,               #to try to obtain the best result
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'log2',              #I used log2 instead of sqrt, better results
              #'max_depth': 20,                    #I removed the max depth limit 
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)

RF_model.fit(X_train, y_train)
y_pred_RF_2 = RF_model.predict(X_test)

plt.title('Random forest')
plt.plot(y_test_2[1000:1200])
plt.plot(y_pred_RF_2[1000:1200])
plt.ylabel('Price [€/MWh]')
plt.xlabel('Hour index')
plt.legend(['y_test', 'y_pred_RF']);
plt.show()

plt.title('Random forest')
plt.xlabel('y test')
plt.ylabel('y pred RF')  
plt.scatter(y_test_2,y_pred_RF_2)
plt.show()

#Error evaluation
MAE_RF_2=metrics.mean_absolute_error(y_test_2,y_pred_RF_2) 
MSE_RF_2=metrics.mean_squared_error(y_test_2,y_pred_RF_2)  
RMSE_RF_2= np.sqrt(metrics.mean_squared_error(y_test_2,y_pred_RF_2))
cvRMSE_RF_2=RMSE_RF_2/np.mean(y_test_2)
print(MAE_RF_2,MSE_RF_2,RMSE_RF_2,cvRMSE_RF_2)

# 2.0824156068878774 8.514496346938012 2.917960991332477 0.06796182443497317

#df creation for the dash visualization 
df_2_RF=pd.DataFrame(y_pred_RF_2,y_test_2)
df_2_RF=df_2_RF.reset_index()
df_2_RF=df_2_RF.rename(columns={'index':'Test Prices [€/MWh]',0:'Predicted Prices [€/MWh]'})

#%%Extreme gradient boosting
from xgboost import XGBRegressor

XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB_2 =XGB_model.predict(X_test)

plt.title('Extreme Gradient boosting')
plt.plot(y_test_2[1000:1200])
plt.plot(y_pred_XGB_2[1000:1200])
plt.ylabel('Price [€/MWh]')
plt.xlabel('Hour index')
plt.legend(['y_test', 'y_pred_XGB']);
plt.show()

plt.title('Extreme Gradient boosting')
plt.xlabel('y test')
plt.ylabel('y pred XGB') 
plt.scatter(y_test_2,y_pred_XGB_2)
plt.show()

MAE_XGB_2=metrics.mean_absolute_error(y_test_2,y_pred_XGB_2) 
MSE_XGB_2=metrics.mean_squared_error(y_test_2,y_pred_XGB_2)  
RMSE_XGB_2= np.sqrt(metrics.mean_squared_error(y_test_2,y_pred_XGB_2))
cvRMSE_XGB_2=RMSE_XGB_2/np.mean(y_test_2)
print(MAE_XGB_2,MSE_XGB_2,RMSE_XGB_2,cvRMSE_XGB_2)

# 2.085017713587977 8.500655143235084 2.915588301395635 0.0679065624429845

#df creation for the dash visualization 
df_2_XGB=pd.DataFrame(y_pred_XGB_2,y_test_2)
df_2_XGB=df_2_XGB.reset_index()
df_2_XGB=df_2_XGB.rename(columns={'index':'Test Prices [€/MWh]',0:'Predicted Prices [€/MWh]'})


#%% 
# =============================================================================================
#  DASHBOARD
# =============================================================================================

#creation of the dataframe needed for the daily pattern identification
raw_data_all_new=raw_data_all.reset_index()
raw_data_all_new['DateDay']=raw_data_all_new['Date'].dt.date
raw_data_all_new['DateDay']=pd.to_datetime(raw_data_all_new['DateDay'], format='%Y-%m-%d')
raw_data_all_new=raw_data_all_new.set_index('Date')

#%importing libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from datetime import datetime as dt
import base64

__name__ = '__main__' #needed because of problems with an error

##insert the static image
image_dpattern= 'assets/Clustering/Daily Pattern.png'
encoded_image_dailyp = base64.b64encode(open(image_dpattern, 'rb').read())

#% Needed for dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)   #App creation and suppression of the callback exceptions

#% Style for the tabs
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#d6d6d6",
                }

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
                }

# =============================================================================
# #Application start - main 
# =============================================================================

sidebar = html.Div([                                             #main division
                                      
    html.Center(html.H3('Italian Electricity Prices Forecast', className="display-7")),            #heading
         html.Hr(),
        html.Img(src='assets/IST-1.png'                 #IST logo
              ,
            style={
                'height': '10%',
                'width': '100%',
            }
            ),
        html.Hr(),
   html.Center(html.P('Emanuele D\'Argenzio ist1100846', className="lead")),        #my name
   
# =============================================================================
# #Tabs identification and definition
# =============================================================================
dbc.Nav(                    #Here the value states the starting tab when we open the dashboard
        [
        dbc.NavLink('About', href="/about", active="exact" ,external_link=True),
        dbc.NavLink('Exploratory Data Analysis', href="/EDA", active="exact",external_link=True),
        dbc.NavLink('Clustering', href="/cluster", active="exact",external_link=True),
        dbc.NavLink('Feature Selection', href="/feature", active="exact",external_link=True), 
        dbc.NavLink('General Regression model', href="/regression-alldata", active="exact",external_link=True), 
        dbc.NavLink('Regression model - 1st cluster', href="/regression-cluster-1", active="exact",external_link=True),
        dbc.NavLink('Regression model - 2nd cluster', href="/regression-cluster-2", active="exact",external_link=True),
        ],
            vertical=True,
            pills=True,
        ),
   ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

tab1_layout = html.Div([html.H3('The aim of this project is to forecast the electricity prices for the italian day-ahead market.'),
                        html.H6('Italy can be divided in 7 zones, represented in the figure below:'),
                        html.Center(html.Img(src='assets/Day_ahead_market_zones.png')),
                        html.H3('For the project, it has been decided to forecast the eletricity prices for the zone North.'),
                        html.H6('The italian market regulator is called GME (Gestore Mercati Elettrici) and from its website it is possible to download the hystorical data for the electricity prices (https://www.mercatoelettrico.org/it/Download/DatiStorici.aspx).'),
                        html.H6('For the analysis, 7 years (from 2013 to 2019) have been downloaded.'),
                        html.H6('The datas are hourly based, so all the project has been computed considering hourly based datas.'),
                        html.H3('As additional datas, the temperature and the global irradiance have been considered.'),
                        html.H6('In particular, no API file has been downloaded since the account was limited to 500 datas (and the project needed almost 60k datas).'),
                        html.H6('Because of that, the hystorial weather datas have been dowloaded from the ARPA ,Regional agency for the environmental protection (https://www.arpalombardia.it/Pages/Meteorologia/Richiesta-dati-misurati/Guida-richiesta-dati.aspx) from 2013 to 2019.'),
                        html.H6('In particular, since the north zone is quite big, it has been decided to take the weather datas for Milan (biggest city in the north).'),
                        html.H4('The project will be divided in different sections:'),
                        dcc.Markdown('''
                                     -Exploratory data analysis, when you can plot the data on daily-basis and look at the probability distribution for temperature and power
                                     
                                     -Clustering identification with the daily pattern
                                     
                                     -Feature selection for the forecasting model
                                     
                                     -Three forecasting: for all the datas, the first and the second cluster
    
                                     '''),
                                     html.H6('The cluster are of two types: low prices day forecasting and high prices day forecasting'),
                        ])

#Layout for the 2nd tab - Exploratory data analysis
tab2_layout= html.Div([
                      html.H5('In this section, different graphs show the data cleaned. Please select the one you want to display:'),
    dcc.Dropdown( 
        id='dropdown_exp',
        options=[
            {'label': 'Price Vs Temperature vs Solar Radiation', 'value': 11},
            {'label': 'Price probability distribution', 'value': 22},
            {'label': 'Temperature probability distribution', 'value': 33},
            {'label': 'Irradiance probability distribution', 'value': 44},
        ], 
        value=11
        ),
        html.Div(id='Exploratory_data_analysis'),
    ])

# Layout for the 3rd tab - clustering
tab3_layout= html.Div([
                html.H5('Please, select the clustering model you want to display:'),
                dcc.Dropdown( 
        id='dropdown_clustering',
        options=[
            {'label': 'Price Vs Temperature', 'value': 1},
            {'label': 'Price vs Hours', 'value': 2},
            {'label': 'Price vs Week Day', 'value': 3},
            {'label': 'Hours vs Price', 'value': 4},
            {'label': 'Daily pattern', 'value': 5}
        ], 
        value=1
        ),
        html.Div(id='Clustering_id'),
    ])

#Layout for the 4th tab - feature selection
tab4_layout= html.Div([
                        html.H5('In this section, the feature selection process is reported. In particular, three different models can be selected.'),
  html.H6('Please, select the model you want to display:'),
        dcc.RadioItems(
        id='image-dropdown-feature',
        options=[
            {'label': 'F-regression', 'value': 300},
            {'label': 'Mutual Info Regression', 'value': 400},
            {'label': 'Random Forest', 'value': 500},
            ],
        value=300
    ),
    html.Div(id='image-feature')
    ])

#Layout for the 5th tab forecasting models   
tab5_layout= html.Div([
                            html.H5('In this section, the regression model used in the project are presented. The selected features are:'),
                            html.H6('Hours, Week Day, Holiday, Cluster, Price-1'),
                            html.H5('Please, select the model you want to show'),
        dcc.Dropdown( 
        id='image-dropdown-regression',
        options=[
            {'label': 'Random Forest', 'value': 111},
            {'label': 'Extreme Gradient Boosting', 'value': 222},
            {'label': 'Forecasting results', 'value': 333},
        ], 
        value=111
        ),
    html.H6(id='image-regression')
    ])

tab6_layout= html.Div([
                            html.H5('Forecasting for the 1st cluster cluster'),
        dcc.Dropdown( 
        id='regression_cluster_1',
        options=[
            {'label': 'Random Forest', 'value': 1111},
            {'label': 'Extreme Gradient Boosting', 'value': 2221},
            {'label': 'Forecasting results', 'value': 3331},
        ], 
        value=1111
        ),
    html.H6(id='reg_cluster_1')
    ])
tab7_layout= html.Div([
                            html.H5('Forecasting for the 2nd cluster cluster'),
        dcc.Dropdown( 
        id='regression_cluster_2',
        options=[
            {'label': 'Random Forest', 'value': 11111},
            {'label': 'Extreme Gradient Boosting', 'value': 22211},
            {'label': 'Forecasting results', 'value': 33311},
        ], 
        value=11111
        ),
    html.H6(id='reg_cluster_2')
    ])

# =============================================================================
#  Callback section                                 
# =============================================================================

# Exploratory data analysis callbacks
@app.callback(Output('Exploratory_data_analysis', 'children'), 
              Input('dropdown_exp', 'value'))
def render_figure_exp(exp_graphs):
    
    if exp_graphs == 11:
        return html.Div([
                        html.H6('For this graph, you can select the date or the data range you prefer and, on the right side, you can delete the data you don\'t want to show by clicking on it'),
                        html.Div([
                        dcc.DatePickerRange(
                            id='my-date-picker-range',             # ID to be used for callback
                            calendar_orientation='horizontal',     # vertical or horizontal
                            day_size=39,                           # size of calendar image. Default is 39
                            end_date_placeholder_text="Return",    # text that appears when no end date chosen
                            with_portal=False,                     # if True calendar will open in a full screen overlay portal
                            first_day_of_week=0,                   # Display of calendar when open (0 = Sunday)
                            reopen_calendar_on_clear=True,
                            is_RTL=False,                          # True or False for direction of calendar
                            clearable=True,                        # whether or not the user can clear the dropdown
                            number_of_months_shown=1,              # number of months shown when calendar is open
                            min_date_allowed=dt(2013, 1, 1),       # minimum date allowed on the DatePickerRange component
                            max_date_allowed=dt(2020, 1, 1),       # maximum date allowed on the DatePickerRange component
                            initial_visible_month=dt(2013, 1, 1),  # the month initially presented when the user opens the calendar
                            start_date=dt(2013, 1, 1, 1, 00, 00),
                            end_date=dt(2013, 1, 2, 1, 00, 00),
                            display_format='MMM Do, YY',           # how selected dates are displayed in the DatePickerRange component.
                            month_format='MMMM, YYYY',             # how calendar headers are displayed when the calendar is opened.
                            minimum_nights=0,                      # minimum number of days between start and end date
                            persistence=True,
                            #persisted_props=['start_date'],
                            persistence_type='session',            # session, local, or memory. Default is 'local'
                            updatemode='singledate'                # singledate or bothdates. Determines when callback is triggered
    )
]),
    html.Div(id='datatable-interactivity-container'),
    dcc.Graph(id='mymap')
])

    elif exp_graphs == 22:
        return html.Div([html.P("Distributions:"),
    dcc.RadioItems(                                       #radio items to select between box,violin,pivot
        id='dist-marginal',
        options=[{'label': x, 'value': x} 
                 for x in ['box', 'violin', 'rug']],
        value='box'
    ),
    dcc.Graph(id="prob"),
    ])            #probability distribution for the price

    elif exp_graphs == 33:
        return html.Div([html.P("Distributions:"),
    dcc.RadioItems(                                        #radio items to select between box,violin,pivot
        id='dist-marginal2',
        options=[{'label': x, 'value': x} 
                  for x in ['box', 'violin', 'rug']],
        value='box'
    ),
    dcc.Graph(id="prob2"),
    ])           #probability distribution for the temeprature
    elif exp_graphs == 44:
        return html.Div([html.P("Distributions:"),
    dcc.RadioItems(                                        #radio items to select between box,violin,pivot
        id='dist-marginal3',
        options=[{'label': x, 'value': x} 
                  for x in ['box', 'violin', 'rug']],
        value='box'
    ),
    dcc.Graph(id="prob3"),
    ])           #probability distribution for the temeprature

#Callback for the data visualization with data pick range    
@app.callback(
    Output('mymap', 'figure'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date')]
)
def update_output(start_date, end_date):
    dff = raw_data_all.loc[start_date:end_date]
    dff=dff.reset_index()
    figure={
                      'data': [
                          {'x': dff['Date'], 'y': dff['Price [€/MWh]'],'type': 'line', 'name': 'Price [€/MWh]'},
                          {'x': dff['Date'], 'y': dff['Temperature [°C]'], 'type': 'line', 'name': 'Temperature [°C]'},
                          {'x': dff['Date'], 'y': dff['Irradiance [W/m2]'], 'type': 'line', 'name': 'Irradiance [W/m<sup>2</sup>]'},
                          ],
                      'layout': {
                          'title': 'Electricity prices vs Temperature vs Irradiance'
            }
        }
    return figure
 
# Callback for the price hystogram: possible to select among violin, boxplot and rug
@app.callback(
    Output("prob", "figure"), 
    [Input("dist-marginal", "value")])
def display_graph(marginal):
    fig = px.histogram(
        raw_data_all['Price [€/MWh]'], x="Price [€/MWh]",
        marginal=marginal,title= 'Probability distribution for the Prices')
    return fig

#callback for the temperature hystogram : possible to select among violin, boxplot and rug
@app.callback(
    Output("prob2", "figure"), 
[Input("dist-marginal2", "value")])
def display_graph_1(marginal):
    fig2 = px.histogram(
        raw_data_all['Temperature [°C]'], x="Temperature [°C]",color_discrete_sequence=['orange'],
        title= 'Probability distribution for the temperature',
        marginal=marginal)
    return fig2

@app.callback(
    Output("prob3", "figure"), 
[Input("dist-marginal3", "value")])
def display_graph_2(marginal):
    fig3 = px.histogram(
        raw_data_all['Irradiance [W/m2]'], x="Irradiance [W/m2]",color_discrete_sequence=['green'],
        title= 'Probability distribution for the irradiance',
        marginal=marginal)
    return fig3

#%clustering callback
@app.callback(Output('Clustering_id', 'children'), 
              Input('dropdown_clustering', 'value'))
def render_figure_png(cluster_diff):

    if cluster_diff == 1:
        return html.Div([dcc.Graph(
        figure=px.scatter(raw_data_all,x='Price [€/MWh]', y='Temperature [°C]',                                     #clustering price vs temperature
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Price vs Temperature')
        ),])
    elif cluster_diff == 2:
        return html.Div([dcc.Graph(id='prova2',
        figure=px.scatter(raw_data_all,x='Price [€/MWh]', y='Hours',                                                #clustering price vs hours
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Price vs Hours')
        ),])
    elif cluster_diff == 3:
        return html.Div([dcc.Graph(id='prova3',
        figure=px.scatter(raw_data_all,x='Price [€/MWh]', y='Week Day',                                              #clustering price vs Week Day
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Price vs Week day')
        ),])
    elif cluster_diff == 4:
        return html.Div([dcc.Graph(id='prova4',
        figure=px.scatter(raw_data_all,x='Hours',y='Price [€/MWh]',                                                  #clustering Hours vs price
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Hours vs Price')
         ),])
    elif cluster_diff == 5:
        return html.Div([html.H6('In below, the daily pattern, with the corresponding two clusters, are reported:'),
         dcc.Graph(id='dnaknjksa',
         figure={
                      'data': [
                          {'x': raw_data_all_new['Hours'], 'y': raw_data_all_new['Price [€/MWh]'],'type': 'line', 'name': 'Price [€/MWh]','stackgroup':'DateDay'},
                          {'x': df_price_pattern_0['Hours'], 'y': df_price_pattern_0[0], 'type': 'line', 'name': 'Price [€/MWh] 1<sup>st</sup> cluster'},
                          {'x': df_price_pattern_1['Hours'], 'y': df_price_pattern_1[0], 'type': 'line', 'name': 'Prices [€/MWh] 2<sup>nd</sup> cluster'},
                          ],
                      'layout': {
                          'title': 'Cluster representation','height':'700'
            }
        }),
          html.Img(src='data:image/png;base64,{}'.format(encoded_image_dailyp.decode())),
                   
        ])

#feature selection callbacks
@app.callback(
    dash.dependencies.Output('image-feature', 'children'),
    [dash.dependencies.Input('image-dropdown-feature', 'value')])
def update_image_src_fe(image_fe):    
    if   image_fe==300:
        return html.Div([
            dcc.Graph(       
             figure={
             'data': [
             {'x': ['Hours'], 'y': F_reg_score['Hours'], 'type': 'bar', 'name': 'Hours'},
             {'x': ['Day'], 'y': F_reg_score['Day'], 'type': 'bar', 'name': 'Day'},
             {'x': ['Week Day'], 'y': F_reg_score['Week Day'], 'type': 'bar', 'name': 'Week Day'},
             {'x': ['Month'], 'y': F_reg_score['Month'], 'type': 'bar', 'name': 'Month'},
             {'x': ['Year'], 'y': F_reg_score['Year'], 'type': 'bar', 'name': 'Year'},
             {'x': ['Holiday'], 'y': F_reg_score['Holiday'], 'type': 'bar', 'name': 'Holiday'},
             {'x': ['Temperature [°C]'], 'y': F_reg_score['Temperature [°C]'], 'type': 'bar', 'name': 'Temperature [°C]'},
             {'x': ['Cluster'], 'y': F_reg_score['Cluster'], 'type': 'bar', 'name': 'Cluster'},
             {'x': ['Irradiance [W/m2]'], 'y': F_reg_score['Irradiance [W/m2]'], 'type': 'bar', 'name': 'Irradiance [W/m<sup>2</sup>]'},
             {'x': ['Price-1 [€/MWh]'], 'y': F_reg_score['Price-1 [€/MWh]'], 'type': 'bar', 'name': 'Price-1 [€/MWh]'},   
             {'x': ['day2'], 'y': F_reg_score['day2'], 'type': 'bar', 'name': 'day2'},

            ],
             'layout': {
                 'title': 'F-regression'
             }
             }                                                               

        ),])
    elif image_fe==400:
        return html.Div([
            dcc.Graph(        
            figure={
            'data': [
            {'x': ['Hours'], 'y': F_mut_info_reg_score['Hours'], 'type': 'bar', 'name': 'Hours'},
            {'x': ['Day'], 'y': F_mut_info_reg_score['Day'], 'type': 'bar', 'name': 'Day'},
            {'x': ['Week Day'], 'y': F_mut_info_reg_score['Week Day'], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Month'], 'y': F_mut_info_reg_score['Month'], 'type': 'bar', 'name': 'Month'},
            {'x': ['Year'], 'y': F_mut_info_reg_score['Year'], 'type': 'bar', 'name': 'Year'},
            {'x': ['Holiday'], 'y': F_mut_info_reg_score['Holiday'], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Temperature [°C]'], 'y': F_mut_info_reg_score['Temperature [°C]'], 'type': 'bar', 'name': 'Temperature [°C]'},
            {'x': ['Cluster'], 'y': F_mut_info_reg_score['Cluster'], 'type': 'bar', 'name': 'Cluster'},
            {'x': ['Irradiance [W/m2]'], 'y': F_mut_info_reg_score['Irradiance [W/m2]'], 'type': 'bar', 'name': 'Irradiance [W/m<sup>2</sup>]'},
            {'x': ['Price-1 [€/MWh]'], 'y': F_mut_info_reg_score['Price-1 [€/MWh]'], 'type': 'bar', 'name': 'Price-1 [€/MWh]'},   
            {'x': ['day2'], 'y': F_mut_info_reg_score['day2'], 'type': 'bar', 'name': 'day2'},

           ],
            'layout': {
                'title': 'Mutual info regression'
            }
        }
        ),])    
    elif image_fe==500:
        return html.Div([
            dcc.Graph(
            figure={
            'data': [
            {'x': ['Hours'], 'y': random_forest_score['Hours'], 'type': 'bar', 'name': 'Hours'},
            {'x': ['Day'], 'y': random_forest_score['Day'], 'type': 'bar', 'name': 'Day'},
            {'x': ['Week Day'], 'y': random_forest_score['Week Day'], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Month'], 'y': random_forest_score['Month'], 'type': 'bar', 'name': 'Month'},
            {'x': ['Year'], 'y': random_forest_score['Year'], 'type': 'bar', 'name': 'Year'},
            {'x': ['Holiday'], 'y': random_forest_score['Holiday'], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Temperature [°C]'], 'y': random_forest_score['Temperature [°C]'], 'type': 'bar', 'name': 'Temperature [°C]'},
            {'x': ['Cluster'], 'y': random_forest_score['Cluster'], 'type': 'bar', 'name': 'Cluster'},
            {'x': ['Irradiance [W/m2]'], 'y': random_forest_score['Irradiance [W/m2]'], 'type': 'bar', 'name': 'Irradiance [W/m<sup>2</sup>]'},
            {'x': ['Price-1 [€/MWh]'], 'y': random_forest_score['Price-1 [€/MWh]'], 'type': 'bar', 'name': 'Price-1 [€/MWh]'},   
            {'x': ['day2'], 'y': random_forest_score['day2'], 'type': 'bar', 'name': 'day2'},
           ],
            'layout': {
                'title': 'Random forest'
            }
            }
        ),])    

#%rgeneral regression model callback
@app.callback(
    dash.dependencies.Output('image-regression', 'children'),
    [dash.dependencies.Input('image-dropdown-regression', 'value')])
def update_image_src_re(value):
    if value == 111:
                return html.Div([dcc.Graph(
              id='lineRF',                                      #real vs forecasted prices - line plot
                  figure={
                      'data': [
                          {'y': y_test, 'type': 'line', 'name': ' Real Prices [€/MWh]'},
                          {'y': y_pred_RF,'type': 'line', 'name': 'Forecasted Prices [€/MWh]'}
                          ],
                      'layout': {
                          'title': 'Predicted vs real prices - Random forest'
            }
        }),
            dcc.Graph(id='scatter0RF',                          #real vs forecasted prices - scatter plot
            figure=px.scatter(df_0_RF, x='Predicted Prices [€/MWh]', y='Test Prices [€/MWh]', title='Random forest')
        ),],)
                    
    elif value == 222:
        return html.Div([dcc.Graph(
              id='lineXGB',                                     #real vs forecasted prices - line plot
                  figure={
                      'data': [
                          {'y': y_test, 'type': 'line', 'name': 'Real Price [€/MWh]'},
                          {'y': y_pred_XGB,'type': 'line', 'name': 'Forecasted Price [€/MWh]'}
                          ],
                      'layout': {
                          'title': 'Predicted vs real prices - Extreme Gradient Boosting'
            }
        }),
            dcc.Graph(id='scatter0xgb',                          #real vs forecasted prices - scatter plot
            figure=px.scatter(df_0_XGB, x='Predicted Prices [€/MWh]', y='Test Prices [€/MWh]', title='Extreme Gradient Boosting'),      
        ),],)
    elif value == 333:
        return html.Div([html.H4('Here you can look at the numerical results of the regression models'),
            dcc.Graph(
            id='0results',
            figure = go.Figure(data=[
        go.Bar(name='Random Forest', x=['MAE [€/MWh]','MSR [€/MWh]^2','RMSE [€/MWh]','cv RMSE [%]'], y=[MAE_RF, MSE_RF, RMSE_RF, cvRMSE_RF*100]),
    go.Bar(name='XGB', x=['MAE [€/MWh]','MSR [€/MWh]^2','RMSE [€/MWh]','cv RMSE [%]'], y=[MAE_XGB, MSE_XGB, RMSE_XGB, cvRMSE_XGB*100])],
    layout=go.Layout(barmode='group'), layout_title_text='Forecasting results - general model')
    )])            

#%1st cluster regression model callback
@app.callback(
    dash.dependencies.Output('reg_cluster_1', 'children'),
    [dash.dependencies.Input('regression_cluster_1', 'value')])
def update_image_src_re_1(value):
    if value == 1111:
                return html.Div([dcc.Graph(
              id='lineRF1',    
                  figure={
                      'data': [
                          {'y': y_test_1, 'type': 'line', 'name': ' Real Prices [€/MWh]'},
                          {'y': y_pred_RF_1,'type': 'line', 'name': 'Forecasted Prices [€/MWh]'}
                          ],
                      'layout': {
                          'title': 'Predicted vs real prices - Random forest'
            }
        }),
            dcc.Graph(id='scatter1RF',
            figure=px.scatter(df_1_RF, x='Predicted Prices [€/MWh]', y='Test Prices [€/MWh]', title='Random forest')      
        ),],)
                    
    elif value == 2221:
        return html.Div([dcc.Graph(
              id='lineXGB1',   
                  figure={
                      'data': [
                          {'y': y_test_1, 'type': 'line', 'name': 'Real Price [€/MWh]'},
                          {'y': y_pred_XGB_1,'type': 'line', 'name': 'Forecasted Price [€/MWh]'}
                          ],
                      'layout': {
                          'title': 'Predicted vs real prices - Extreme Gradient Boosting'
            }
        }),
            dcc.Graph(id='scatter1xgb',
            figure=px.scatter(df_1_XGB, x='Predicted Prices [€/MWh]', y='Test Prices [€/MWh]', title='Extreme Gradient Boosting')      
        ),],)
    elif value == 3331:
        return html.Div([html.H4('Here you can look at the numerical results of the regression models'),
            dcc.Graph(
            id='1results',
            figure = go.Figure(data=[
    go.Bar(name='Random Forest', x=['MAE [€/MWh]','MSR [€/MWh]^2','RMSE [€/MWh]','cv RMSE [%]'], y=[MAE_RF_1,MSE_RF_1,RMSE_RF_1,cvRMSE_RF_1*100]),
    go.Bar(name='XGB', x=['MAE [€/MWh]','MSR [€/MWh]^2','RMSE [€/MWh]','cv RMSE [%]'], y=[MAE_XGB_1,MSE_XGB_1,RMSE_XGB_1,cvRMSE_XGB_1*100])],
    layout=go.Layout(barmode='group'),layout_title_text='Forecasting results - 1<sup>st</sup> cluster')
    )])    


#%2nd cluster regression model callback
@app.callback(
    dash.dependencies.Output('reg_cluster_2', 'children'),
    [dash.dependencies.Input('regression_cluster_2', 'value')])
def update_image_src_re_2(value):
    if value == 11111:
                return html.Div([dcc.Graph(
              id='lineRF2',    
                  figure={
                      'data': [
                          {'y': y_test_2, 'type': 'line', 'name': ' Real Prices [€/MWh]',},
                          {'y': y_pred_RF_2,'type': 'line', 'name': 'Forecasted Prices [€/MWh]'}
                          ],
                      'layout': {
                          'title': 'Predicted vs real prices - Random forest'
            }
        }),
            dcc.Graph(id='scatter1RF',
            figure=px.scatter(df_2_RF, x='Predicted Prices [€/MWh]', y='Test Prices [€/MWh]', title='Random forest')      
        ),],)
                    
    elif value == 22211:
        return html.Div([dcc.Graph(
              id='lineXGB2',    
                  figure={
                      'data': [
                          {'y': y_test_2, 'type': 'line', 'name': 'Real Price [€/MWh]'},
                          {'y': y_pred_XGB_2,'type': 'line', 'name': 'Forecasted Price [€/MWh]'}
                          ],
                      'layout': {
                          'title': 'Predicted vs real prices - Extreme Gradient Boosting'
            }
        }),
            dcc.Graph(id='scatter2xgb',
            figure=px.scatter(df_2_XGB, x='Predicted Prices [€/MWh]', y='Test Prices [€/MWh]', title='Extreme Gradient Boosting')      
        ),],)
    elif value == 33311:
        return html.Div([html.H4('Here you can look at the numerical results of the regression models'),
            dcc.Graph(
            id='2results',
            figure = go.Figure(data=[
    go.Bar(name='Random Forest', x=['MAE [€/MWh]','MSR [€/MWh]^2','RMSE [€/MWh]','cv RMSE [%]'], y=[MAE_RF_2,MSE_RF_2,RMSE_RF_2,cvRMSE_RF_2*100]),
    go.Bar(name='XGB', x=['MAE [€/MWh]','MSR [€/MWh]^2','RMSE [€/MWh]','cv RMSE [%]'], y=[MAE_XGB_2,MSE_XGB_2,RMSE_XGB_2,cvRMSE_XGB_2*100])],
    layout=go.Layout(barmode='group'),layout_title_text='Forecasting results - 2<sup>nd</sup> cluster')
    )])    
    
#tab callback
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/about":
        return tab1_layout
    elif pathname == "/EDA":
        return tab2_layout
    elif pathname == "/cluster":
        return tab3_layout
    elif pathname == "/feature":
        return tab4_layout
    elif pathname == "/regression-alldata":
        return tab5_layout
    elif pathname == "/regression-cluster-1":
        return tab6_layout
    elif pathname == "/regression-cluster-2":
        return tab7_layout    

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)