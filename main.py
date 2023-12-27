from datetime import timezone
import datetime
import pandas as pd
import numpy as np
import pytz
# from src.scripts.dbConn import create_conn
from src.scripts.concatFiles import concatFiles
from sklearn.base import BaseEstimator, TransformerMixin
from src.scripts.filter_csv import filterOutliers
from src.scripts.naValues import naValues
from sklearn.tree import DecisionTreeRegressor
from src.scripts.featureSelection import featureSelection
from src.scripts.featureEncoding import featureEncoding
from src.scripts.normalization import normalizeData
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
mfdf=pd.read_csv('src/data/player_statistics.csv',sep=';') 
mfdf['Age'] = mfdf['Age'].str.split('-').str[0].astype(int)
filterOutliers(mfdf)
df=concatFiles()
cleanedDf=naValues(df)
encodedDf=featureEncoding(cleanedDf)
normalizedDf =normalizeData(encodedDf)
finalData = featureSelection(normalizedDf)



X = finalData.drop('market_value_in_eur', axis=1)  
y = finalData['market_value_in_eur'] 



from sklearn.ensemble import GradientBoostingRegressor
# def age_adjustment(age):
#     if age > 0.7:
#         return 0.6
#     elif age > 0.6:
#         return 0.7
#     elif age > 0.5:
#         return 1.0
#     elif age > 0.4:
#         return 1.2
#     else:
#         return 1.4


def performGB(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    gradient_boosting = GradientBoostingRegressor()
    k_folds = KFold(n_splits = 10)
    scores = cross_val_score(gradient_boosting, X, y, cv=k_folds)
    print("Cross-validation scores(Gradient Boosting):", scores)
    print(scores.mean())
    draw_cross_val(scores, 'Gradient Boosting')
    prediction_extractor = PredictionExtractor(gradient_boosting) 
    prediction_extractor.fit(X_train, y_train)
    X_test_with_predictions = prediction_extractor.transform(X_test)
    #X_test_with_predictions["model_prediction"] = np.array([pred * age_adjustment(age) for pred, age in zip(X_test_with_predictions["model_prediction"], X_test['Age'])])
    gb_mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
    print("Mean Squared Error(Gradient Boosting):", gb_mse)
    r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
    print("R-squared (Gradient Boosting):", r2)
    player_predictions = pd.merge(X_test_with_predictions, cleanedDf[['player', 'market_value_in_eur']], left_index=True, right_index=True)
    player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]
    #float to int model_prediction
    player_predictions['model_prediction'] = player_predictions['model_prediction'].astype(int)
    player_predictions.to_csv('src/data/predictionResults.csv', index=False)
    tz = pytz.timezone('UTC')
    current_datetime = datetime.datetime.now(tz)
    #db insertion
    # connection = create_conn()
    # cursor = connection.cursor()
    # #insert model metrics
    # insertModelMetricsSql = "INSERT INTO [dbo].[model_metrics] ([model_name], [n_fold_score], [mean_squarred_error], [r_squared], [precision], [recall], [f1_score], [createdAt], [updatedAt]) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    # cursor.execute(insertModelMetricsSql, ('Gradient Boosting', scores.mean(), gb_mse, r2, "", "", "", current_datetime,current_datetime))
    # connection.commit()
    # #insert predicted values
    # insertPredictedValueSql = "INSERT INTO [dbo].[prediction_results] ([name], [market_value_in_eur], [model_prediction],[prediction_model],[createdAt], [updatedAt]) VALUES (?, ?, ?, ?, ?, ?)"
    # data = [(row['player'],row['market_value_in_eur'] ,row['model_prediction'],'Gradient Boosting',current_datetime, current_datetime) for index, row in player_predictions.iterrows()]
    # cursor.executemany(insertPredictedValueSql, data)
    # connection.commit()
    

    return player_predictions

def performkNN(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    knn = KNeighborsRegressor(metric="euclidean")
    prediction_extractor = PredictionExtractor(knn)

    prediction_extractor.fit(X_train, y_train)

    X_test_with_predictions = prediction_extractor.transform(X_test)
    k_folds = KFold(n_splits = 10)
    scores = cross_val_score(knn, X, y, cv=k_folds)
    print("Cross-validation scores(kNN):", scores)
    print(scores.mean())
    draw_cross_val(scores, 'K-Nearest Neighbors')
   # X_test_with_predictions["model_prediction"] = np.array([pred * age_adjustment(age) for pred, age in zip(X_test_with_predictions["model_prediction"], X_test['Age'])])
    mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
    print("Mean Squared Error(K-Nearest Neighbors):", mse)
    r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
    print("R-squared (K-Nearest Neighbors):", r2)
    # precision = precision_score(y_test, X_test_with_predictions["model_prediction"])
    # recall = recall_score(y_test, X_test_with_predictions["model_prediction"])
    # f1 = 2 * (precision * recall) / (precision + recall)

    player_predictions = pd.merge(X_test_with_predictions, cleanedDf[['player', 'market_value_in_eur']], left_index=True, right_index=True)
    player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

    return player_predictions

def performDT(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    tree = DecisionTreeRegressor()
    prediction_extractor = PredictionExtractor(tree)

    prediction_extractor.fit(X_train, y_train)

    X_test_with_predictions = prediction_extractor.transform(X_test)
    #X_test_with_predictions["model_prediction"] = np.array([pred * age_adjustment(age) for pred, age in zip(X_test_with_predictions["model_prediction"], X_test['Age'])])
    k_folds = KFold(n_splits = 10)
    scores = cross_val_score(tree, X, y, cv=k_folds)
    print("Cross-validation scores(Decision Tree):", scores)
    print(scores.mean())
    draw_cross_val(scores, 'Decision Tree')
    mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
    print("Mean Squared Error(Decision Tree):", mse)
    r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
    print("R-squared (Decision Tree):", r2)

    player_predictions = pd.merge(X_test_with_predictions, cleanedDf[['player', 'market_value_in_eur']], left_index=True, right_index=True)
    player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

    return player_predictions

def performLR(X,y):  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    linear_regression = LinearRegression()
    prediction_extractor = PredictionExtractor(linear_regression)

    prediction_extractor.fit(X_train, y_train)
 
    X_test_with_predictions = prediction_extractor.transform(X_test)
    #X_test_with_predictions["model_prediction"] = np.array([pred * age_adjustment(age) for pred, age in zip(X_test_with_predictions["model_prediction"], X_test['Age'])])
    k_folds = KFold(n_splits = 10)
    scores = cross_val_score(linear_regression, X, y, cv=k_folds)
    print("Cross-validation scores(Linear Regression):", scores)
    print(scores.mean())
    draw_cross_val(scores, 'Linear Regression')
    mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
    print("Mean Squared Error(Linear Regression):", mse)
    r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
    print("R-squared (Linear Regression):", r2)

    # Return player info along with actual and predicted market values
    player_predictions = pd.merge(X_test_with_predictions, cleanedDf[['player', 'market_value_in_eur']], left_index=True, right_index=True)
    player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

 

    return player_predictions


def performRandomForest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    random_forest = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                           min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                           min_impurity_decrease=0.0, bootstrap=True,
                                           oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
                                           ccp_alpha=0.0, max_samples=None)

    prediction_extractor = PredictionExtractor(random_forest)

    prediction_extractor.fit(X_train, y_train)

    X_test_with_predictions = prediction_extractor.transform(X_test)
   # X_test_with_predictions["model_prediction"] = np.array([pred * age_adjustment(age) for pred, age in zip(X_test_with_predictions["model_prediction"], X_test['Age'])])
    k_folds = KFold(n_splits = 10)
    scores = cross_val_score(random_forest, X, y, cv=k_folds)
    print("Cross-validation scores(Random Forest):", scores)
    print(scores.mean())
    draw_cross_val(scores, 'Random Forest')
    mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
    print("Mean Squared Error(Random Forest):", mse)
    r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
    print("R-squared (Random Forest):", r2)

    player_predictions = pd.merge(X_test_with_predictions, cleanedDf[['player', 'market_value_in_eur']], left_index=True, right_index=True)
    player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

    return player_predictions



def performAdaBoost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    adaboost = AdaBoostRegressor(n_estimators=50, random_state=42)
    prediction_extractor = PredictionExtractor(adaboost)

    prediction_extractor.fit(X_train, y_train)

    X_test_with_predictions = prediction_extractor.transform(X_test)
   # X_test_with_predictions["model_prediction"] = np.array([pred * age_adjustment(age) for pred, age in zip(X_test_with_predictions["model_prediction"], X_test['Age'])])
    k_folds = KFold(n_splits = 10)
    scores = cross_val_score(adaboost, X, y, cv=k_folds)
    draw_cross_val(scores, 'AdaBoost')
    print("Cross-validation scores(AdaBoost):", scores)
    print(scores.mean())
    mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
    print("Mean Squared Error(AdaBoost):", mse)
    r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
    print("R-squared (AdaBoost):", r2)

    player_predictions = pd.merge(X_test_with_predictions, cleanedDf[['player', 'market_value_in_eur']], left_index=True, right_index=True)
    player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

    return player_predictions


finalData = pd.merge(finalData, cleanedDf[['player', 'market_value_in_eur']], left_index=True, right_index=True)
 
def draw_cross_val(scores,algorithmName):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(scores) + 1), scores)
    plt.xlabel('Fold number')
    plt.ylabel('Cross-Validation Score')
    plt.title('Cross-Validation Scores per Fold('+algorithmName+')')
    plt.grid(True)
    plt.show()
class PredictionExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        predictions = self.model.predict(X)
        X["model_prediction"] = predictions
        return X




    

print("Output of Linear Regression model:")
print(performLR(X,y))

print("Output of Gradient Boosting model:")
print(performGB(X,y))

print("Output of K-Nearest Neighbors model:")
print(performkNN(X,y))

print("Output of Decision Tree model:")
print(performDT(X,y))


print("Output of Random Forest model:")
print(performRandomForest(X, y))


print("Output of AdaBoost model:")
print(performAdaBoost(X, y))