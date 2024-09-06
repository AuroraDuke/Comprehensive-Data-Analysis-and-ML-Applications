# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:10:47 2024

@author: ilker
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
from sklearn.exceptions import ConvergenceWarning,DataConversionWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import ARDRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time

df_main = pd.read_csv('arcgis_ngas_usak_v02.csv')


df_main.drop(df_main[df_main['Total'] < 300 ].index, inplace=True)
df_main=df_main.head(1000)
df_main = df_main.fillna(0)

df_year = pd.get_dummies(df_main['Year']).astype(int)
#feature extraction--------------------------------------------------------------------
def get_long(long):
    if 29.0 <= long < 29.5:
        return 'long1'
    elif 29.5 <= long < 30.0:
        return 'long2'
    else:
        return 'long3'

df_longitude = df_main['longitude'].apply(get_long)
df_longitude_encode = pd.get_dummies(df_longitude).astype(int)

#feature extraction--------------------------------------------------------------------
def get_lat(lat):
    if 38.0 <= lat < 38.5:
        return 'lati1'
    elif 38.5 <= lat < 39.0:
        return 'lati2'
    else:
        return 'lati3'

df_latitude = df_main['latitude'].apply(get_lat)
df_latitude_encode = pd.get_dummies(df_latitude).astype(int)


#feature extraction--------------------------------------------------------------------
month= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                  'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df_month = pd.DataFrame({'Month': month})
#get month values datas
for line in range(len(df_main)):
    if line == 0:
        df_temp = df_main.iloc[line,15:27]
    else:
        df_temp0=pd.DataFrame()
        df_temp0 = df_main.iloc[line,15:27]  
        df_temp = pd.concat([df_temp,df_temp0],axis=0)

df_temp.reset_index(drop=True, inplace=True)

df_month = pd.concat([df_month] *len(df_main), ignore_index=True)
df_temp = pd.concat([df_month,df_temp],axis=1)

df_temp.columns.values[1] = 'm3'


#sm for everyline
df_concat = pd.concat([df_year, df_main.iloc[:,1:13],df_longitude_encode,df_latitude_encode,df_main['Total']], axis=1)
df_concat.reset_index(drop=True, inplace=True)
# get into seasonal_decompose
import statsmodels.api as sm
temp=0
df_dict = {}
i=12
while(i < len(df_temp)+1):
    res_spec = sm.tsa.seasonal_decompose(df_temp.iloc[temp:i,1], model='additive', period=3)
   
    if i < 100 :
        res_spec.plot()
   
    res_spec_df = pd.concat([res_spec.trend, res_spec.seasonal,res_spec.resid],axis=1)
    
    res_spec_df.reset_index(drop=True, inplace=True)
   
    user_id=int((i-1)/11) 
    df_dict[f'user_{user_id}'] = res_spec_df
    temp=i
    i+=12
    

#Trend--------------------------------------------------------------------------------------------------------
time_res_df = pd.concat([df for df in df_dict.values()], axis=0)
time_res_df = time_res_df.fillna(0)
time_res_df.reset_index(drop=True, inplace=True)

#look at dic 
#trend  seasonilty resides
#enlarge the dataset 12 times
for line in range(len(df_concat)):
    if line == 0:
        repeat_line = df_concat.iloc[[line]]     
        repeat_df = pd.concat([repeat_line] * 12, ignore_index=True)
        
    else:
        repeat_line = df_concat.iloc[[line]]     
        repeat_df0 = pd.concat([repeat_line] * 12, ignore_index=True)
       
        repeat_df = pd.concat([repeat_df,repeat_df0] ,axis=0)

repeat_df.reset_index(drop=True, inplace=True)
time_res_df.reset_index(drop=True, inplace=True)


df_trend = pd.concat([repeat_df,df_temp,time_res_df],axis=1)
df_trend.columns = df_trend.columns.astype(str)

#predict preparation
output = df_trend['Total']
inputs = df_trend.drop(columns=['Total','Month'])

output = output.astype(float)
inputs = inputs.astype(float)

#Graph-------------------------------------------------------------------------------------------------------
corr = pd.concat([inputs,output],axis=1)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, fmt=".2f", linewidths=0.9, center=0)
plt.title("Corr Matrix)")
plt.show()

total_columns = len(inputs.columns)

num_cols_per_plot = 10

num_plots = total_columns // num_cols_per_plot
if total_columns % num_cols_per_plot != 0:
    num_plots += 1

# Plotları oluşturma
for plot_index in range(num_plots):
    start_index = plot_index * num_cols_per_plot
    end_index = min((plot_index + 1) * num_cols_per_plot, total_columns)
    
    
    plt.figure(figsize=(20, 15))
    #graph
    for i, col in enumerate(inputs.columns[start_index:end_index]):
        plt.subplot(2, 5, i+1)  
        x = inputs[col]
        y = output
        plt.plot(x, y, 'o')
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Total')

    plt.tight_layout()
    plt.show()
    #hist
    for i, col in enumerate(inputs.columns[start_index:end_index]):
        plt.subplot(2, 5, i+1)  
        inputs[col].hist(bins=10, grid=False)
        plt.title(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

#box
for i in range(0, len(inputs.columns), 10):
    subset_columns = inputs.columns[i:i+10]  # İlgili sütun aralığını seç
    fig, axes = plt.subplots(1, 10, figsize=(15, 5), sharex=False, sharey=False)
    for idx, col in enumerate(subset_columns):
        inputs[col].plot(kind='box', ax=axes[idx])
    plt.tight_layout()
    plt.show()
    

#Regression-------------------------------------------------------------------------------------------------------
# Separate data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2,train_size=0.8, random_state=47,shuffle=True,stratify=output)

# ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



#Loading the models to will be tried------------------------------------------------------------------------------
models = []
#models.append(('LR', LinearRegression()))
#models.append(('gaus', GaussianProcessRegressor( random_state=47)))

#models.append(('ridge', RidgeCV()))
#models.append(('LASSO', Lasso()))
#models.append(('EN', ElasticNet(alpha=2.5, l1_ratio=0.1, max_iter=1000,selection= 'random', tol=0.0001, random_state=47,warm_start= True)))
#models.append(('KNN', KNeighborsRegressor()))


#models.append(('DT', DecisionTreeRegressor(criterion = 'friedman_mse',max_depth = 15,max_features = 1.0,min_impurity_decrease = 0.0,min_samples_leaf = 2,splitter = 'random',min_samples_split=2,min_weight_fraction_leaf= 0.0,random_state=47)))
#models.append(('ExtraT', ExtraTreesRegressor(max_features=1.0,n_estimators=120, max_depth=9, min_samples_split=2,min_samples_leaf=1, random_state=47)))

models.append(('Adaboost',AdaBoostRegressor(base_estimator= DecisionTreeRegressor(max_depth=5), learning_rate=0.1, n_estimators= 100, random_state=47)))
models.append(('Graboost',GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, min_samples_split= 10,min_samples_leaf= 1, max_features= None,random_state=47)))

#models.append(('ard',ARDRegression()))
#models.append(('sgd',SGDRegressor(max_iter=1000, tol=1e-3)))

#models.append(('lars', make_pipeline(StandardScaler(with_mean=False), Lars()) ))
#models.append(( 'histgb',HistGradientBoostingRegressor()))

#Voting
#models.append(('vote',VotingRegressor(estimators=[('a',HistGradientBoostingRegressor()), 
#('b',ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=47)),
#('c',GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.1,  min_samples_split= 10,min_samples_leaf= 1,  max_features= None,random_state=42)),('d',DecisionTreeRegressor(criterion='friedman_mse', max_depth =10, min_samples_leaf= 2,max_features= None,min_weight_fraction_leaf=0, min_samples_split= 2,splitter= 'random',random_state=42))])))

#SVR
models.append(('SVR-Linear', SVR(kernel="linear")))
#models.append(('SVR-RBF', SVR(kernel='rbf')))
#models.append(('SVR-Sigmoid', SVR(kernel="sigmoid")))
#models.append(('SVR-Poly2', SVR(kernel="poly",degree=2)))
#models.append(('SVR-Poly3', SVR(kernel="poly",degree=3)))

#ANN
#models.append(('ANN-lbfgs',MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))
#models.append(('ANN-sgd',MLPRegressor(solver='sgd', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))
#models.append(('ANN-adam',MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))

#Boosting
#models.append(('xgb',xgb.XGBRegressor(verbosity=0,learning_rate = 0.01,max_depth= 5,n_estimators= 300,subsample= 0.8)))
#models.append(('lgb',lgb.LGBMRegressor(verbose=-1)))
#models.append(('catb',CatBoostRegressor(verbose=False)))

#Testing models------------------------------------------------------------------------------
num_folds = 10
results = {}
#NOTE catboost is very successful but takes too much time
start_time0 = time.time()
for name, model in models:
    start_time = time.time()
    
    kfold = KFold(n_splits=num_folds,shuffle=True,random_state=47)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2")
    
    mean_r2score = cv_results.mean()
    
    model.fit(X_train, y_train)
    test_r2score = model.score(X_test, y_test)
    results[name] = test_r2score
    
    
    print(f"{name} CV - r2score :  {mean_r2score:.16f}")
    print(f"{name} test-r2score :  {test_r2score:.16f}")
    
   
    end_time = time.time();elapsed_time = end_time - start_time;print(f"{name.ljust(8)}: {elapsed_time} saniye\n")
    
end_time0 = time.time();elapsed_time = end_time0 - start_time0;print(f"{elapsed_time} saniye")
    

print("\n")

best_model = max(results, key=results.get)
print("best dependOnTest: ", best_model)


models_dict = dict(models)
# Train the best model
best_model_instance = models_dict.get(best_model)
best_model_instance.fit(X_train, y_train)

# Evaluate the performance of the best model on the test set
test_r2score = best_model_instance.score(X_test, y_test)
print("Test set r2 score (best model):", test_r2score)

# Calculate evaluation metrics for the best model
y_pred = best_model_instance.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation metrics for the best model
print("\nEvaluation metric scores for best model which is {}:".format(best_model))
print("Test set r2:", test_r2score)
print("Test set MSE (Mean Squared Error):", mse)
print("Test set MAE (Mean Absolute Error):", mae)

import pickle


best_model = max(results, key=results.get)
print("Best model:", best_model)


best_model_instance = models_dict.get(best_model)


with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model_instance, f)
 