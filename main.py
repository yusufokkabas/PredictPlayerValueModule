import pandas as pd
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


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into features (X) and target variable (y)
X = finalData.drop('market_value_in_eur', axis=1)  # Adjust the column name accordingly
y = finalData['market_value_in_eur']  # Adjust the column name accordingly


def performLR(X,y):  

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    linear_regression = LinearRegression()

    # Fit the model to the training data
    linear_regression.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = linear_regression.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

from sklearn.ensemble import GradientBoostingRegressor

def performGB(X,y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Gradient Boosting
    gradient_boosting = GradientBoostingRegressor()
    gradient_boosting.fit(X_train, y_train)
    gb_predictions = gradient_boosting.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_predictions)
    print("Gradient Boosting MSE:", gb_mse)




performGB(X,y)