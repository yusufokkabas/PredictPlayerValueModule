import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def featureSelection(df):
    print(df)
    X = df.drop('market_value_in_eur', axis=1)  # Features
    y = df['market_value_in_eur']  # Target variable
    # Initialize a random forest classifier as the estimator
    # Create a random forest classifier object
    rf =RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=5)

    # Create the Boruta feature selector object
    boruta_selector = BorutaPy(rf, random_state=10, verbose=2)

    # Fit the selector to your data
    boruta_selector.fit(X.values, y.values)

    # Transform your dataset to keep only the selected features
    X_selected = boruta_selector.transform(X.values)
    # Convert X_selected to a DataFrame using the column names of the selected features
    selected_features = X.columns[boruta_selector.support_]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

    # Concatenate the target variable (y) with the selected features (X_selected_df)
    concatenated_df = pd.concat([y, X_selected_df], axis=1)
    print(concatenated_df.columns)



