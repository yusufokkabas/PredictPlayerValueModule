import pandas as pd
from src.scripts.concatFiles import concatFiles
from src.scripts.filter_csv import filterOutliers
from src.scripts.naValues import naValues
from src.scripts.featureSelection import featureSelection
from src.scripts.featureEncoding import featureEncoding
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
mfdf=pd.read_csv('src/data/player_statistics.csv',sep=';') 
filterOutliers(mfdf)
df=concatFiles()
df=naValues(df)
encodedDf=featureEncoding(df)
#featureSelection(encodedDf)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Split the data into input features (X) and target variable (y)
X = encodedDf.drop('market_value_in_eur', axis=1)
y = encodedDf['market_value_in_eur']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
model = RandomForestClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)