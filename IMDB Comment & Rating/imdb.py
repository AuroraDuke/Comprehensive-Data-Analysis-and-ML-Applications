# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:58:11 2024

@author: ilker
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

datas=pd.read_csv('imdb_reviews_ratings.csv')
datas=datas.head(20000)


movie_id = datas.iloc[:,0]
movie_id = movie_id.apply(lambda x: re.sub('[^0-9]', '', x))
movie_id = pd.get_dummies(movie_id).astype(int)

#encoding
def remove_non_alphabetic(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text)) # Replace non-alphabetic characters with space

# Apply the function to the 'Review,Liked' column
genres = datas['genres'].apply(remove_non_alphabetic)


dir_genres = pd.DataFrame(columns=['Action','Drama','Adventure','SciFi','Thriller','Crime','Romance','Horror'])

def find_category(row, category):
    return 1 if category in row else 0

for col in dir_genres.columns:
    dir_genres[col] = genres.apply(find_category, category=col)
    
#part2
    
    
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def remove_non_alphabetic(text):
    return re.sub('[^a-zA-Z]', ' ', str(text)).lower()


review = datas['review'].apply(remove_non_alphabetic)

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


edited_strings = []

for text in review:
    words = text.split()
    edited_words = [ps.stem(word) for word in words if word not in stop_words]
    edited_text = ' '.join(edited_words)
    edited_strings.append(edited_text)

print('vector is starting...')
from scipy.sparse import csr_matrix#it was so fast but it didn't mean it for me
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=100)
text_vector = cv.fit_transform(edited_strings) #undependent variables
df_text = pd.DataFrame(text_vector.toarray(), columns=cv.get_feature_names_out())

helpful_rating=datas.iloc[:,5:]
#I dropped movie_title because it had ids
df_result = pd.concat([movie_id,dir_genres,df_text,helpful_rating],axis=1)
df_result = df_result.dropna()
df_result=df_result.astype(int)
output = df_result['rating']
inputs = df_result.drop('rating',axis=1)

print('regresyon is starting...')


x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2,train_size=0.8, random_state=47,shuffle=True)

sc=StandardScaler(with_mean=False)
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#Loading the models to be tried------------------------------------------------------------------------------
models = []
#models.append(('LR', LinearRegression()))
#models.append(('gaus', GaussianProcessRegressor( random_state=47)))

#models.append(('ridge', RidgeCV()))
#models.append(('LASSO', Lasso()))
#models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))

#models.append(('DT', DecisionTreeRegressor(random_state=47)))
#models.append(('RaFoReg',RandomForestRegressor(random_state=47)))
#models.append(('ExtraT', ExtraTreesRegressor( random_state=47)))
#models.append(('Ada',AdaBoostRegressor(base_estimator= DecisionTreeRegressor(max_depth=6), learning_rate=0.01, n_estimators= 100, random_state=47)))

#models.append(('Graboost',GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_split= 10,min_samples_leaf= 1, max_features= None,random_state=47)))

models.append(('ard',ARDRegression()))
#models.append(('sgd',SGDRegressor()))

#models.append(('lars',Lars()))
#models.append(( 'histgb',HistGradientBoostingRegressor()))

#Voting
#models.append(('vote',VotingRegressor(estimators=[('a',HistGradientBoostingRegressor()), ('b',ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)),('c',GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.1,  min_samples_split= 10,min_samples_leaf= 1,  max_features= None,random_state=42)),('d',DecisionTreeRegressor(criterion='friedman_mse', max_depth =10, min_samples_leaf= 2,max_features= None,min_weight_fraction_leaf=0, min_samples_split= 2,splitter= 'random',random_state=42))])))

#SVR
#models.append(('SVR-Linear', SVR(kernel="linear")))
#models.append(('SVR-RBF', SVR(kernel='rbf')))
#models.append(('SVR-Sigmoid', SVR(kernel="sigmoid")))
#models.append(('SVR-Poly2', SVR(kernel="poly",degree=2)))
#models.append(('SVR-Poly3', SVR(kernel="poly",degree=3)))

#ANN
#models.append(('ANN-lbfgs',MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))
#models.append(('ANN-sgd',MLPRegressor(solver='sgd', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))
#models.append(('ANN-adam',MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))

#Boosting
#models.append(('xgb',xgb.XGBRegressor(verbosity=0)))
#models.append(('lgb',lgb.LGBMRegressor(verbose=-1)))
#models.append(('catb',CatBoostRegressor(verbose=False)))

#Testing models------------------------------------------------------------------------------
num_folds = 10
results = {}

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


























from sklearn.model_selection import GridSearchCV
"""
param_grid_df = {
   'criterion': ['friedman_mse', 'squared_error' ,'absolute_error'],#, 'poisson', 'squared_error' ,'absolute_error'
    'max_depth': [5,10,15,20],
   'min_samples_split': [0.1,1.0,2,3],
    "min_samples_leaf": [0.1,1,2,3],
    "min_weight_fraction_leaf": [0.0,0.3,0.5],
    'max_features': [1.0, None, 'sqrt', 'log2'],#, None, 'sqrt', 'log2'
    #"max_leaf_nodes": [None],
   "splitter": ['random','best'],#,'best'
   "ccp_alpha":[0.0,0.5,1.0],
   "min_impurity_decrease":[0.0,0.5,1.0],
}

# GridSearchCV is create
dt_cv = GridSearchCV(DecisionTreeRegressor(random_state=47),param_grid_df,cv=kfold,scoring="r2", error_score='raise')

# GridSearchCV fit
dt_cv.fit(X_train, y_train)
#DT Best model parameters: {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'max_depth': 5, 'max_features': 1.0, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 0.1, 'min_weight_fraction_leaf': 0.0, 'splitter': 'best'}
#DT Best score: 0.4647619047619047
# Best fit and scoring
print("DT Best model parameters:", dt_cv.best_params_)
print("DT Best score:", dt_cv.best_score_)
"""
"""
param_grid_ada = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5],
    'base_estimator': [DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=10)],
}

# GridSearchCV oluştur
ada_cv = GridSearchCV(AdaBoostRegressor(random_state=47),param_grid_ada,cv=kfold,scoring="r2")

# GridSearchCV ile modeli eğit
ada_cv.fit(X_train, y_train)

#Best Parameter and Best score
print("Best Parameter:", ada_cv.best_params_)
print("Best score:", ada_cv.best_score_)

df_result.to_csv('Imdb_rating_edited.csv', index=False)

"""