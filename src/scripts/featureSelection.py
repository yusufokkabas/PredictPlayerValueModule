import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def featureSelection(df):
    X = df.drop('market_value_in_eur', axis=1)  # Features
    y = df['market_value_in_eur']  # Target variable
    # Initialize a random forest classifier as the estimator
    # Create a random forest classifier object
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

        # Create the Boruta feature selector object
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    # Fit the selector to your data
    boruta_selector.fit(X, y)

    # Get the indices of the selected features
    selected_features = np.where(boruta_selector.support_)[0]

    # Transform your dataset to keep only the selected features
    X_selected = boruta_selector.transform(X)
    print(X_selected)



