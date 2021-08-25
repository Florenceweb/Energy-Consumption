# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:10:48 2021

@author: Florence
"""

#Energy Consumption in France - Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
import seaborn as sns
#from sklearn.decomposition import PCA
#from sklearn import model_selection
#from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
#from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
#from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
#from sklearn.svm import SVR, LinearSVR
#from sklearn import datasets, linear_model,svm
from sklearn.metrics import mean_absolute_error 
from sklearn.linear_model import SGDRegressor
import scipy.stats as stats
#import pmdarima as pm
#import statsmodels.regression.linear_model as linmod
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.stattools import adfuller  
#from pandas.plotting import autocorrelation_plot
#from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 

from bokeh.plotting import figure, output_notebook, show
#from bokeh.models import ColumnDataSource
from bokeh.models import Range1d, LinearAxis
output_notebook()

#Reading dataset 'energy'
energy = pd.read_csv("energie_projet.csv", ";")

#Show the first 5 lines
#energy.head()

#Type of value
#energy.info()

#Description of dataset
#energy.describe()

#Delete
Col = ["Flux physiques d'Auvergne-Rhône-Alpes vers Grand-Est",
       'Flux physiques de Bourgogne-Franche-Comté vers Grand-Est',
       'Flux physiques de Bretagne vers Grand-Est',
       'Flux physiques de Centre-Val de Loire vers Grand-Est',
       'Flux physiques de Grand-Est vers Grand-Est',
       'Flux physiques de Hauts-de-France vers Grand-Est',
       "Flux physiques d'Ile-de-France vers Grand-Est",
       'Flux physiques de Normandie vers Grand-Est',
       'Flux physiques de Nouvelle-Aquitaine vers Grand-Est',
       "Flux physiques d'Occitanie vers Grand-Est",
       'Flux physiques de Pays-de-la-Loire vers Grand-Est',
       'Flux physiques de PACA vers Grand-Est',
       'Flux physiques de Grand-Est vers Auvergne-Rhône-Alpes',
       'Flux physiques de Grand-Est vers Bourgogne-Franche-Comté',
       'Flux physiques de Grand-Est vers Bretagne',
       'Flux physiques de Grand-Est vers Centre-Val de Loire',
       'Flux physiques de Grand-Est vers Grand-Est.1',
       'Flux physiques de Grand-Est vers Hauts-de-France',
       'Flux physiques de Grand-Est vers Ile-de-France',
       'Flux physiques de Grand-Est vers Normandie',
       'Flux physiques de Grand-Est vers Nouvelle-Aquitaine',
       'Flux physiques de Grand-Est vers Occitanie',
       'Flux physiques de Grand-Est vers Pays-de-la-Loire',
       'Flux physiques de Grand-Est vers PACA',
       'Flux physiques Allemagne vers Grand-Est',
       'Flux physiques Belgique vers Grand-Est',
       'Flux physiques Espagne vers Grand-Est',
       'Flux physiques Italie vers Grand-Est',
       'Flux physiques Luxembourg vers Grand-Est',
       'Flux physiques Royaume-Uni vers Grand-Est',
       'Flux physiques Suisse vers Grand-Est',
       'Flux physiques de Grand-Est vers Allemagne',
       'Flux physiques de Grand-Est vers Belgique',
       'Flux physiques de Grand-Est vers Espagne',
       'Flux physiques de Grand-Est vers Italie',
       'Flux physiques de Grand-Est vers Luxembourg',
       'Flux physiques de Grand-Est vers Royaume-Uni',
       'Flux physiques de Grand-Est vers Suisse', 'Ech. physiques (MW)', 'Nature', 'Date - Heure',
       'TCH Thermique (%)', 'TCH Nucléaire (%)',
       'TCH Eolien (%)', 'TCH Solaire (%)', 'TCH Hydraulique (%)',
       'TCH Bioénergies (%)','TCO Thermique (%)', 'TCO Nucléaire (%)','TCO Eolien (%)', 'TCO Solaire (%)', 
       'TCO Hydraulique (%)','TCO Bioénergies (%)']

for i in Col:
    energy=energy.drop([i], axis = 1)

#Number of NaN
#energy.isnull().sum()
    
#Deleted NaN
energy = energy.dropna(subset=['Consommation (MW)','Thermique (MW)','Eolien (MW)','Solaire (MW)','Hydraulique (MW)',
                           'Bioénergies (MW)'])

#Replaced NaN by 0
energy['Nucléaire (MW)']=energy['Nucléaire (MW)'].fillna(0)
energy['Pompage (MW)']=energy['Pompage (MW)'].fillna(0)

#energy.isnull().sum()

#Group by "Région" and "Date"
energy_group = energy.groupby(['Date', 'Région'], as_index=False).agg({'Consommation (MW)':'sum'})
energy_group['Date']= pd.to_datetime(energy_group['Date'])

#energy_group.head()

energyDate = energy.groupby(['Date'], as_index=False).agg({'Consommation (MW)': 'sum'})
#energyDate.head()

#Distribution of consumption over time
fig = plt.figure(figsize=(16,9))
plt.plot(energyDate['Date'], energyDate['Consommation (MW)'])
plt.axvline('2013-12-31', color='red')
plt.axvline('2014-12-31', color='red')
plt.axvline('2015-12-31', color='red')
plt.axvline('2016-12-31', color='red')
plt.axvline('2017-12-31', color='red')
plt.axvline('2018-12-31', color='red')
plt.axvline('2019-12-31', color='red')
plt.axvline('2020-12-31', color='red')
plt.ylabel("Consumption (MW)")
plt.title("Distribution of energy consumption over time (2013 to March 2021)")
plt.savefig("Distribution energy.png")
plt.show()

#Reading dataset 'meteo'
meteo = pd.read_csv("synop.csv", ";")

#Show the first 5 lines
#meteo.head()

#Type of value
#meteo.info()

#In the column 'Date', keep only "day, month and year"
meteo[['Date','heure']] = meteo.Date.str.split("T",expand=True,)
meteo['Date']= pd.to_datetime(meteo['Date'])

#Rename the column 'region (name)'
meteo.rename(columns={'region (name)':'Région'},inplace=True)

#Group by "Région" and "Date"
meteo_group = meteo.groupby(['Date', 'Région'], as_index=False).agg({'Température (°C)':'mean','Humidité':'mean',
                                                                            'Visibilité horizontale':'mean', 'Précipitations dans les 24 dernières heures':'mean'})

#meteo_group.info()

#Merging datasets
energyMeteo = pd.merge(energy_group, meteo_group,  how='inner', left_on=['Date','Région'], right_on = ['Date','Région'])

#Show the first 5 lines
energyMeteo.head()

#Information
#energyMeteo.info()

#Deleted NaN
energyMeteo = energyMeteo.dropna(subset=['Humidité','Précipitations dans les 24 dernières heures'])

#energyMeteo.isnull().sum()

#Consumption depending on the weather
energyMeteoDate = energyMeteo.groupby(['Date'], as_index=False).agg({'Consommation (MW)': 'sum', 'Température (°C)':'mean',
                                                                    'Humidité':'mean', 'Visibilité horizontale':'mean',
                                                                     'Précipitations dans les 24 dernières heures':'mean'})
#energyMeteoDate.head()

#Energy consumption and temperatures
fig, ax1 = plt.subplots(figsize = (20,5), sharey=True)
ax1.plot(energyMeteoDate['Date'], energyMeteoDate['Température (°C)'], color = 'orange')
ax1.set_ylabel('Temperature (°C)', color = 'orange')
ax2 = ax1.twinx()
ax2.plot(energyMeteoDate['Date'],energyMeteoDate['Consommation (MW)'],color = 'black')
ax2.set_ylabel('Consumption (MW)',color = 'black')
plt.title("Evolution of energy consumption and temperatures")
fig.tight_layout()
plt.savefig("Consumption Temperature.png")
plt.show()

#Energy consumption and precipitations
fig, ax1 = plt.subplots(figsize = (20,5))
ax1.plot(energyMeteoDate['Date'], energyMeteoDate['Précipitations dans les 24 dernières heures'], color = 'blue')
ax1.set_ylabel('Precipitation',color = 'blue')
ax2 = ax1.twinx()
ax2.plot(energyMeteoDate['Date'],energyMeteoDate['Consommation (MW)'],color = 'black')
ax2.set_ylabel('Consumption (MW)',color = 'black')
plt.title("Evolution of energy consumption and precipitations")
fig.tight_layout()
plt.savefig("Consumption precipitations.png")
plt.show()

#Energy consumption and horizontal visibility
fig, ax1 = plt.subplots(figsize = (20,5))
ax1.plot(energyMeteoDate['Date'], energyMeteoDate['Visibilité horizontale'], color = 'green')
ax1.set_ylabel('Horizontal visibility',color = 'green')
ax2 = ax1.twinx()
ax2.plot(energyMeteoDate['Date'],energyMeteoDate['Consommation (MW)'],color = 'black')
ax2.set_ylabel('Consumption (MW)',color = 'black')
plt.title("Evolution of energy consumption and horizontal visibility")
fig.tight_layout()
plt.savefig("Consumption Visibility.png")
plt.show()

#Energy consumption and humidity
fig, ax1 = plt.subplots(figsize = (20,5))
ax1.plot(energyMeteoDate['Date'], energyMeteoDate['Humidité'], color = 'pink')
ax1.set_ylabel('Humidity',color = 'pink')
ax2 = ax1.twinx()
ax2.plot(energyMeteoDate['Date'],energyMeteoDate['Consommation (MW)'],color = 'black')
ax2.set_ylabel('Consumption (MW)',color = 'black')
plt.title("Evolution of energy consumption and humidity")
fig.tight_layout()
plt.savefig("Consumption Humidity.png")
plt.show()

#Consumption depending on the population
#Reading dataset of French population density of 2018 
population = pd.read_csv("Regions.csv", ";")

#population.info()

#Show the dataset
#population.head(18)

#Delate rows that we don't need (Corse, Guadeloupe, Guyane, La Réunion, Martinique)
population.drop([4,6,7,10,11], 0, inplace = True)

#Delate the columns and keep only CODREG, REG, PTOT
population.drop(['NBARR', 'NBCAN', 'NBCOM', 'PMUN'], axis = 1, inplace = True)

#Split the 'Date'
energyMeteo['année'], energyMeteo['mois'],energyMeteo['jour'] = energyMeteo['Date'].dt.year, energyMeteo['Date'].dt.month, energyMeteo['Date'].dt.day
#energyMeteo.head()

#Keep the consumption of 2018
energyDixHuit = energyMeteo[(energyMeteo['année'] == 2018)]

#energyDixHuit.head()

energyPop = energyDixHuit.groupby(['Région']).agg({'Consommation (MW)':'sum'})

#Rename the column 'region (name)'
population.rename(columns={'REG':'Région'},inplace=True)

#Merging datasets
energyPopulation = pd.merge(energyPop, population,  how='inner', left_on=['Région'], right_on = ['Région'])

#energyPopulation.head(17)

modalites = ['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté', 'Bretagne','Centre-Val de Loire','Grand Est',
             'Hauts-de-France', 'Normandie','Nouvelle-Aquitaine', 'Occitanie','Pays de la Loire',
             "Provence-Alpes-Côte d'Azur",'Île-de-France']

p = figure(plot_width = 950, plot_height= 550, tools="hover,box_zoom, reset, pan",
           tooltips=[('PTOT',"@PTOT"),('Consommation (MW)', "@height")],
           title = "Energy consumption in fact of legal population in 2018", x_range = modalites, toolbar_location="above")

#Tooltips is not ok

y=energyPopulation['Consommation (MW)']

adjy = y/2

p.rect(modalites, adjy, width = 0.9, height = y, legend_label= "Energy consumption",color = "green")
p.y_range = Range1d(0,138000000)
p.extra_y_ranges = {"PTOT" : Range1d(start = 0, end = 12600000)}
p.add_layout(LinearAxis(y_range_name = "PTOT"), 'right')
p.xaxis.major_label_orientation = 3.14/4
p.line(modalites, energyPopulation['PTOT'], y_range_name = "PTOT", legend_label= "Population", color = "orange", line_width = 2)
show(p)

#Consumption check, by year.
sns.boxplot( x='année', y='Consommation (MW)', hue=None, data=energyMeteo);

energyMeteo = energyMeteo[(energyMeteo['Date'] < '2021-01-01')]

#Heatmap of values (Correlation matrix)
plt.figure(figsize=(13,13))
sns.heatmap(energyMeteo.corr(), annot=True, center=0, cmap='RdBu_r')
plt.savefig("Heatmap.png");

#Correlation values
energyMeteo.corr()['Consommation (MW)'].abs().sort_values(ascending = False)

#Standardization of data because the variables are not of the same order
column = ['Température (°C)','Humidité','Visibilité horizontale', 
          'Précipitations dans les 24 dernières heures']

energyMeteo[column] = pd.DataFrame(preprocessing.StandardScaler().fit_transform(energyMeteo[column]))

#energyMeteo.head()

#Dichotomization of temporal variables
energyMeteo = energyMeteo.join(pd.get_dummies(energyMeteo.année, prefix = 'année'))
energyMeteo = energyMeteo.join(pd.get_dummies(energyMeteo.mois, prefix = 'mois'))
energyMeteo =energyMeteo.join(pd.get_dummies(energyMeteo.jour, prefix = 'jour'))
energyMeteo = energyMeteo.join(pd.get_dummies(energyMeteo.Région, prefix = 'Région'))

energyMeteo.rename(columns={'année_2013':'2013', 'année_2014':'2014','année_2015':'2015', 'année_2016':'2016','année_2017':'2017',
                     'année_2018':'2018', 'année_2019':'2019','année_2020':'2020', 'mois_1':'janvier',
                     'mois_2':'février', 'mois_3':'mars', 'mois_4':'avril','mois_5':'mai', 'mois_6':'juin', 'mois_7':'juillet', 
                     'mois_8':'août', 'mois_9':'septembre', 'mois_10':'octobre', 'mois_11':'novembre','mois_12':'décembre'},
                       inplace=True)

#'Date' colum positioned in index
energyMeteo.set_index('Date', drop=True, inplace=True)

#Remove the columns 'Région', 'année', 'mois' and 'jour'
energyMeteo = energyMeteo.drop(['Région','année','mois','jour'] , axis =1)
#energyMeteo.head()

#energyMeteo.isnull().sum()



energyMeteo = energyMeteo.dropna(subset=['Température (°C)','Humidité','Visibilité horizontale', 'Précipitations dans les 24 dernières heures'])

#Split of dataset, target variable : target = 'Consommation' and the rest in data
target = energyMeteo['Consommation (MW)']
data = energyMeteo.drop('Consommation (MW)', axis =1)

#Split of training data (2013 to 2017) and the data test (2018 to 2020)
X_train = data[energyMeteo.index < '2018-01-01']
X_test =  data[energyMeteo.index >='2018-01-01'] 

y_train = target[energyMeteo.index <'2018-01-01']
y_test = target[energyMeteo.index >='2018-01-01']

#Model SGDRegressor
#Fit model to training data
sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)

#Precision score on the training sample
scoreTrain = sgdr.score(X_train, y_train)
print("Score Train :", scoreTrain)

#Precision score on the test set
scoreTest = sgdr.score(X_test, y_test)
print("Score Test :", scoreTest)

#Definition of parameters
sgdr.get_params(deep=True)

#Study of residus
pred_train = sgdr.predict(X_train)
residus = pred_train - y_train

plt.scatter(y_train, residus, color = '#980a10', s=15, label = 'Residus')
plt.plot((y_train.min(),y_train.max()), (0,0), lw=3, color = '#0a5798', label = 'Mean of residus')
plt.title("Residus function of training data (SGDR model)")
plt.savefig("Residus function SGDR.png")
plt.legend();

#Calculate quantiles for a probability plot
residus_norm = (residus-residus.mean())/residus.std()
stats.probplot(residus_norm, plot=plt)
plt.savefig("Quantiles SGDR.png")
plt.show()

print("Mean of residus is (SGDR model) :",residus.mean())

y_pred = sgdr.predict(X_test)

#Display of actual consumption data vs predicted consumption data
plt.subplot(121)
plt.scatter(target.index,energyMeteo['Consommation (MW)'], label = "Real")
plt.scatter(X_test.index,y_pred, color = 'green',label = "Predict")
plt.xticks(rotation=90)
plt.grid()
plt.ylabel('Consumption (MW)')
plt.subplot(122)
plt.scatter(X_test.index,y_test, label = "Real")
plt.scatter(X_test.index,y_pred, color = 'green',label = "Predict")
plt.grid()
plt.xticks(rotation=90)
plt.legend()
plt.title('Original data vs Predict data (SGDR model)')
plt.savefig("Priedict SGDR.png")
plt.show()

#Calculation of the mean squared error of prediction
print('Mean Squared Error Train :',mean_squared_error(pred_train, y_train))
print('Mean Squared Error Test :',mean_squared_error(y_pred,y_test))

#Calculation of thr mean absolute error 
print('Mean Absolute Error test :',mean_absolute_error(y_test, sgdr.predict(X_test)))
print('Mean Absolute Error train :',mean_absolute_error(y_train, sgdr.predict(X_train)))

#Model Lasso
#Fit model to training data
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

#Definition of parameters
lasso_reg.get_params(deep=True)

#Displays the estimated value of the coefficient for each data variable
lasso_coef = lasso_reg.coef_
plt.figure(figsize=(10,10))
plt.plot(range(len(data.columns)), lasso_coef)
plt.xticks(range(len(data.columns)), data.columns.values, rotation=90)
plt.title("Coefficient for each data variable")
plt.show()

#Precision score on the training sample and test set
print("Score train:",lasso_reg.score(X_train,y_train))
print("Score test :", lasso_reg.score(X_test,y_test))

lasso_pred_train = lasso_reg.predict(X_train)
lasso_pred_test = lasso_reg.predict(X_test)

#Calculation of the mean squared error of prediction
print('Mean Squared Error test :',mean_squared_error(y_test, lasso_reg.predict(X_test)))
print('Mean Squared Error train :',mean_squared_error(y_train, lasso_reg.predict(X_train)))

#Calculation of thr mean absolute error 
print('Mean Absolute Error test :',mean_absolute_error(y_test, lasso_reg.predict(X_test)))
print('Mean Absolute Error train :',mean_absolute_error(y_train, lasso_reg.predict(X_train)))

#Study of residus
residus_lasso = lasso_pred_train - y_train

plt.scatter(y_train, residus_lasso, color = '#980a10', s=15, label = 'Residus')
plt.plot((y_train.min(),y_train.max()), (0,0), lw=3, color = '#0a5798', label = 'Mean of residus')
plt.title("Residus function of training data (Lasso model)")
plt.savefig("Residus Lasso.png")
plt.legend();

#Calculate quantiles for a probability plot
residus_norm_lasso = (residus_lasso-residus_lasso.mean())/residus_lasso.std()
stats.probplot(residus_norm_lasso, plot=plt)
plt.savefig("Quantile Lasso.png")
plt.show()

print("Mean of residus is (Lasso model) :",residus_lasso.mean())

#Display of actual consumption data vs predicted consumption data
plt.subplot(121)
plt.scatter(target.index,energyMeteo['Consommation (MW)'], label = "Real")
plt.scatter(X_test.index,lasso_pred_test, color = 'green',label = "Predict")
plt.xticks(rotation=90)
plt.grid()
plt.ylabel('Consumption (MW)')
plt.subplot(122)
plt.scatter(X_test.index,y_test, label = "Real")
plt.scatter(X_test.index,lasso_pred_test, color = 'green',label = "Predict")
plt.grid()
plt.xticks(rotation=90)
plt.legend()
plt.title('Original data vs Predict data (Lasso model)')
plt.savefig("Predict Lasso.png")
plt.show()

#Time series
#Split of training data (2013 à 2017) and the data test (2018 à 2021)
energytemp = energy[energy.Date < '2018-01-01']
energytest = energy[energy.Date >= '2018-01-01']

#Remove the variables that we do not need to perform this analysis
energytemp = energytemp.drop(['Code INSEE région', 'Région','Heure'], axis=1)
energytest = energytest.drop(['Code INSEE région', 'Région','Heure'], axis=1)

#Group by average monthly consumption:

#Training data
energytemp.index = energytemp.Date
energytemp.index = pd.to_datetime(energytemp.index)
energytemp = energytemp.drop(['Date'], axis=1)            
energylog = energytemp['Consommation (MW)'].resample('M').mean()

#Test data
energytest.index = energytest.Date
energytest.index = pd.to_datetime(energytest.index)
energytest = energytest.drop(['Date'], axis=1)            
y_test = energytest['Consommation (MW)'].resample('M').mean()

#Graphic representation of training data
plt.figure(figsize = (16,9))
energylog.to_frame()
plt.plot(energylog)
plt.title("Evolution of monthly energy consumption in France between 2013 and 2018")
plt.xlabel("Year")
plt.ylabel("Consumption (MW)")
plt.savefig("Evolution consumption.png")
plt.show()

#Decomposition of the series
res = seasonal_decompose(energylog)
res.plot()
plt.savefig("Decomposition.png")
plt.show()

#Autocorrelation function on the data series
pd.plotting.autocorrelation_plot(energylog)
plt.savefig("Autocorrelation.png");

#Apply a seasonality differentiation and autocorrelation function on the differentiated series
df_1 = energylog.diff(periods = 12).dropna()
pd.plotting.autocorrelation_plot(df_1)
plt.savefig("Autocorrelation2.png");

#Statistical test of Dickey-Fuller

result = sm.tsa.stattools.adfuller(df_1)
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
    
#Search parameters P, Q, p, q

#Plot the simple autocorrelation function and the partial autocorrelation function
plt.figure(figsize= (14,7))
plt.subplot(121)
plot_acf(df_1, lags = 12, ax=plt.gca())
plt.subplot(122)
plot_pacf(df_1, lags = 12, ax=plt.gca())
plt.savefig("Partial autocorrelation.png")
plt.show()

#Linear regression by trying to estimate the best model.

model=sm.tsa.SARIMAX(energylog,order=(1,1,1),seasonal_order=(1,1,1,12))
results=model.fit()
print(results.summary())

model=sm.tsa.SARIMAX(energylog,order=(1,1,0),seasonal_order=(1,1,1,12))
results=model.fit()
print(results.summary())

model=sm.tsa.SARIMAX(energylog,order=(1,1,0),seasonal_order=(0,1,1,12))
results=model.fit()
print(results.summary())

#Prediction between Jan 2018 and March 2021 to compare with the test dataset 
pred = results.predict(60, 99)

plt.figure(figsize = (10,5))

ax = y_test.plot(color='blue', grid=True, label = "Real")
pred.plot(ax=ax,color='red',grid=True, label = "Prediction")
plt.ylabel("Consumption (MW)")
plt.title("Real vs Prediction of consumption (MW)")
plt.legend()
plt.savefig("Real vs Predict.png")
plt.show()

#Calcul of Mean Absolute Prediction Error
y_true, y_pred = np.array(y_test), np.array(pred)
MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("Mean Absolute Prediction Error : %0.2f%%"% MAPE)

#Prediction between Jan 2018 and Dec 2021 to compare with the test dataset 
pred = results.predict(60, 108)

plt.figure(figsize = (10,5))

ax = y_test.plot(color='blue', grid=True, label = "Real")
pred.plot(ax=ax,color='red',grid=True, label = "Prediction")
plt.ylabel("Consumption (MW)")
plt.title("Real vs Prediction of consumption (MW)")
plt.legend()
plt.savefig("Predict.png")
plt.show()