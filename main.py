import pandas as pd
import numpy as np
from src.scripts.concatFiles import concatFiles
from src.scripts.filter_csv import filterOutliers
from src.scripts.naValues import naValues
from src.scripts.featureSelection import featureSelection
from src.scripts.featureEncoding import featureEncoding
from src.scripts.normalization import normalizeData
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
import xgboost as xgb
mfdf=pd.read_csv('src/data/player_statistics.csv',sep=';') 
filterOutliers(mfdf)
df=concatFiles()
df=naValues(df)
encodedDf=featureEncoding(df)
normalizedDf =normalizeData(encodedDf)
finalData = featureSelection(normalizedDf)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score,r2_score

from sklearn.linear_model import LinearRegression

# Split the data into features (X) and target variable (y)
X = finalData.drop('market_value_in_eur', axis=1)  
y = finalData['market_value_in_eur'] 


def performLR(X,y):  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_regression = LinearRegression()

    linear_regression.fit(X_train, y_train)

    y_pred = linear_regression.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error(Logistic Regression):", mse)
    r2 = r2_score(y_test, y_pred)
    print("R-squared (Logistic Regression):", r2)


from sklearn.ensemble import GradientBoostingRegressor

def performGB(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Gradient Boosting
    gradient_boosting = GradientBoostingRegressor()
    gradient_boosting.fit(X_train, y_train)
    gb_predictions = gradient_boosting.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_predictions)
    print("Mean Squared Error(Gradient Boosting):", gb_mse)
    r2 = r2_score(y_test, gb_predictions)
    print("R-squared (Gradient Boosting):", r2)

from sklearn.neighbors import KNeighborsRegressor

def performkNN(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)

    
    predictions = knn.predict(X_test)

    
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error (KNN):", mse)
    r2 = r2_score(y_test, predictions)
    print("R-squared (KNN):", r2)
   
   
from sklearn.tree import DecisionTreeRegressor

def performDT(X,y):
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the Decision Tree model
    tree = DecisionTreeRegressor()
    tree.fit(X_train, y_train)

    
    predictions = tree.predict(X_test)

    
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error (Decision Tree):", mse)

    # Compute R-squared (coefficient of determination)
    r2 = r2_score(y_test, predictions)
    print("R-squared (Decision Tree):", r2)



performLR(X,y)
performGB(X,y)
performkNN(X,y)
performDT(X,y)