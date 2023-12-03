import pandas as pd
import numpy as np
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
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostRegressor


from sqlalchemy import create_engine
import pandas as pd

def create_conn():
    server = 'idsscout.database.windows.net' 
    database = 'idsscout' 
    username = 'idssdb' 
    password = '.99#wA&47R' 
    engine = create_engine(f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=FreeTDS')
    return engine

# Establish connection
engine = create_conn()

# Execute a query
getPlayerSql = "SELECT [general_player_statistic].[id], [general_player_statistic].[season], [general_player_statistic].[name], [general_player_statistic].[first_name], [general_player_statistic].[last_name], [general_player_statistic].[age], [general_player_statistic].[nationality], [general_player_statistic].[height], [general_player_statistic].[weight], [general_player_statistic].[injured], [general_player_statistic].[photo], [general_player_statistic].[market_value_in_eur], [general_player_statistic].[market_value_date], [statistics].[id] AS [statistics.id], [statistics].[createdAt] AS [statistics.createdAt], [statistics].[updatedAt] AS [statistics.updatedAt], [statistics->game_infos].[id] AS [statistics.game_infos.id], [statistics->game_infos].[appearences] AS [statistics.game_infos.appearences], [statistics->game_infos].[lineups] AS [statistics.game_infos.lineups], [statistics->game_infos].[minutes] AS [statistics.game_infos.minutes], [statistics->game_infos].[position] AS [statistics.game_infos.position], [statistics->game_infos].[rating] AS [statistics.game_infos.rating], [statistics->game_infos].[captain] AS [statistics.game_infos.captain], [statistics->duels].[id] AS [statistics.duels.id], [statistics->duels].[total] AS [statistics.duels.total], [statistics->duels].[won] AS [statistics.duels.won], [statistics->fouls].[id] AS [statistics.fouls.id], [statistics->fouls].[drawn] AS [statistics.fouls.drawn], [statistics->fouls].[committed] AS [statistics.fouls.committed], [statistics->cards].[id] AS [statistics.cards.id], [statistics->cards].[yellow] AS [statistics.cards.yellow], [statistics->cards].[yellowred] AS [statistics.cards.yellowred], [statistics->cards].[red] AS [statistics.cards.red], [statistics->dribbles].[id] AS [statistics.dribbles.id], [statistics->dribbles].[attempts] AS [statistics.dribbles.attempts], [statistics->dribbles].[success] AS [statistics.dribbles.success], [statistics->goals].[id] AS [statistics.goals.id], [statistics->goals].[total] AS [statistics.goals.total], [statistics->goals].[conceded] AS [statistics.goals.conceded], [statistics->goals].[assists] AS [statistics.goals.assists], [statistics->goals].[saves] AS [statistics.goals.saves], [statistics->passes].[id] AS [statistics.passes.id], [statistics->passes].[total] AS [statistics.passes.total], [statistics->passes].[key] AS [statistics.passes.key], [statistics->passes].[accuracy] AS [statistics.passes.accuracy], [statistics->penalties].[id] AS [statistics.penalties.id], [statistics->penalties].[won] AS [statistics.penalties.won], [statistics->penalties].[committed] AS [statistics.penalties.committed], [statistics->penalties].[scored] AS [statistics.penalties.scored], [statistics->penalties].[missed] AS [statistics.penalties.missed], [statistics->penalties].[saved] AS [statistics.penalties.saved], [statistics->shots].[id] AS [statistics.shots.id], [statistics->shots].[total] AS [statistics.shots.total], [statistics->shots].[on] AS [statistics.shots.on], [statistics->substitutes].[id] AS [statistics.substitutes.id], [statistics->substitutes].[in] AS [statistics.substitutes.in], [statistics->substitutes].[out] AS [statistics.substitutes.out], [statistics->substitutes].[bench] AS [statistics.substitutes.bench], [statistics->tackles].[id] AS [statistics.tackles.id], [statistics->tackles].[total] AS [statistics.tackles.total], [statistics->tackles].[blocks] AS [statistics.tackles.blocks], [statistics->teams].[id] AS [statistics.teams.id], [statistics->teams].[team_id] AS [statistics.teams.team_id], [statistics->teams].[name] AS [statistics.teams.name], [statistics->teams].[logo] AS [statistics.teams.logo] FROM [general_player_statistics] AS [general_player_statistic] LEFT OUTER JOIN [player_statistics_by_seasons] AS [statistics] ON [general_player_statistic].[statistics_id] = [statistics].[id] LEFT OUTER JOIN [game_infos] AS [statistics->game_infos] ON [statistics].[game_infos_id] = [statistics->game_infos].[id] LEFT OUTER JOIN [duels] AS [statistics->duels] ON [statistics].[duels_id] = [statistics->duels].[id] LEFT OUTER JOIN [fouls] AS [statistics->fouls] ON [statistics].[fouls_id] = [statistics->fouls].[id] LEFT OUTER JOIN [cards] AS [statistics->cards] ON [statistics].[cards_id] = [statistics->cards].[id] LEFT OUTER JOIN [dribbles] AS [statistics->dribbles] ON [statistics].[dribbles_id] = [statistics->dribbles].[id] LEFT OUTER JOIN [goals] AS [statistics->goals] ON [statistics].[goals_id] = [statistics->goals].[id] LEFT OUTER JOIN [passes] AS [statistics->passes] ON [statistics].[passes_id] = [statistics->passes].[id] LEFT OUTER JOIN [penalties] AS [statistics->penalties] ON [statistics].[penalties_id] = [statistics->penalties].[id] LEFT OUTER JOIN [shots] AS [statistics->shots] ON [statistics].[shots_id] = [statistics->shots].[id] LEFT OUTER JOIN [substitutes] AS [statistics->substitutes] ON [statistics].[substitutes_id] = [statistics->substitutes].[id] LEFT OUTER JOIN [tackles] AS [statistics->tackles] ON [statistics].[tackles_id] = [statistics->tackles].[id] LEFT OUTER JOIN [teams] AS [statistics->teams] ON [statistics].[teams_id] = [statistics->teams].[id];"
df = pd.read_sql(getPlayerSql, engine)


print(df)



# mfdf=pd.read_csv('src/data/player_statistics.csv',sep=';') 
df=filterOutliers(df)
#df=concatFiles()
df=naValues(df)
encodedDf=featureEncoding(df)
normalizedDf =normalizeData(encodedDf)
finalData = featureSelection(normalizedDf)

# import matplotlib.pyplot as plt
# import seaborn as sns
# def visualizeData(df):
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

#     plt.figure(figsize=(10, 4))

#     # Histogram
#     plt.subplot(1, 2, 1)
#     sns.histplot(df['market_value_in_eur'], bins=30)
#     plt.title('Histogram')
#     # Boxplot
#     plt.subplot(1, 2, 2)
#     sns.boxplot(x=df['market_value_in_eur'])
#     plt.title('Boxplot')
#     plt.tight_layout()
#     plt.show()

# visualizeData(finalData)

# # Statistical summary
# summary = finalData.describe()
# print("Summary Statistics:")
# print(summary)

# Mode
mode = finalData.mode().loc[0]
print("\nMode:")
print(mode)


# finaldata to csv
finalData.to_csv('src/data/finalData.csv', index=False)



X = finalData.drop('market_value_in_eur', axis=1)  
y = finalData['market_value_in_eur'] 



# from sklearn.ensemble import GradientBoostingRegressor

# def performGB(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     gradient_boosting = GradientBoostingRegressor()
#     prediction_extractor = PredictionExtractor(gradient_boosting)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)

#     gb_mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(Gradient Boosting):", gb_mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (Gradient Boosting):", r2)

#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

#     return player_predictions

# def performkNN(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     knn = KNeighborsRegressor(metric="euclidean")
#     prediction_extractor = PredictionExtractor(knn)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)

#     mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(K-Nearest Neighbors):", mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (K-Nearest Neighbors):", r2)

#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

#     return player_predictions

# def performDT(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     tree = DecisionTreeRegressor()
#     prediction_extractor = PredictionExtractor(tree)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)

#     mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(Decision Tree):", mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (Decision Tree):", r2)

#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

#     return player_predictions

# def performLR(X,y):  
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     linear_regression = LinearRegression()
#     prediction_extractor = PredictionExtractor(linear_regression)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)
#     # now, X_test_with_predictions dataframe includes an additional column "model_prediction" which holds the predicted market value by the model

#     mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(Linear Regression):", mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (Linear Regression):", r2)

#     # Return player info along with actual and predicted market values
#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

 

#     return player_predictions

# def performSVM(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     svm = SVR(C=1.0, gamma='scale', kernel='rbf', degree=3, coef0=0.0, shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1)  
#     prediction_extractor = PredictionExtractor(svm)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)

#     mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(Support Vector Machine):", mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (Support Vector Machine):", r2)

#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

#     return player_predictions

# def performRandomForest(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     random_forest = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
#                                            min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
#                                            min_impurity_decrease=0.0, bootstrap=True,
#                                            oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
#                                            ccp_alpha=0.0, max_samples=None)

#     prediction_extractor = PredictionExtractor(random_forest)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)

#     mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(Random Forest):", mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (Random Forest):", r2)

#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

#     return player_predictions

# def performLDA(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     lda = LinearDiscriminantAnalysis()
#     prediction_extractor = PredictionExtractor(lda)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)

#     mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(Linear Discriminant Analysis):", mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (Linear Discriminant Analysis):", r2)

#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

#     return player_predictions

# def performAdaBoost(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     adaboost = AdaBoostRegressor(n_estimators=50, random_state=42)
#     prediction_extractor = PredictionExtractor(adaboost)

#     prediction_extractor.fit(X_train, y_train)

#     X_test_with_predictions = prediction_extractor.transform(X_test)

#     mse = mean_squared_error(y_test, X_test_with_predictions["model_prediction"])
#     print("Mean Squared Error(AdaBoost):", mse)
#     r2 = r2_score(y_test, X_test_with_predictions["model_prediction"])
#     print("R-squared (AdaBoost):", r2)

#     player_predictions = pd.merge(X_test_with_predictions, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)
#     player_predictions = player_predictions[['player', 'market_value_in_eur', 'model_prediction']]

#     return player_predictions


# players_market_value = pd.read_csv('src/data/cleaned_data.csv') 
# finalData = pd.merge(finalData, players_market_value[['player', 'market_value_in_eur']], left_index=True, right_index=True)

# class PredictionExtractor(BaseEstimator, TransformerMixin):
#     def __init__(self, model):
#         self.model = model

#     def fit(self, X, y=None):
#         self.model.fit(X, y)
#         return self

#     def transform(self, X):
#         predictions = self.model.predict(X)
#         X["model_prediction"] = predictions
#         return X



# performLR(X,y)
# performGB(X,y)
# performkNN(X,y)
# performDT(X,y)
# performSVM(X, y)
# performRandomForest(X, y)
# performLDA(X, y) 
# performAdaBoost(X, y)
# print("Output of Linear Regression model:")
# print(performLR(X,y))

# print("Output of Gradient Boosting model:")
# print(performGB(X,y))

# print("Output of K-Nearest Neighbors model:")
# print(performkNN(X,y))

# print("Output of Decision Tree model:")
# print(performDT(X,y))

# print("Output of Support Vector Machine model:")
# print(performSVM(X, y))

# print("Output of Random Forest model:")
# print(performRandomForest(X, y))

# print("Output of Linear Discriminant Analysis model:")
# print(performLDA(X, y))

# print("Output of AdaBoost model:")
# print(performAdaBoost(X, y))