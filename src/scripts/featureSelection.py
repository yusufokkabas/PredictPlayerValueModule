import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel


from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def featureSelection(df):
    
    X = df.drop('market_value_in_eur', axis=1)  # Adjust the column name accordingly
    y = df['market_value_in_eur']  # Adjust the column name accordingly

 

    # Create a Lasso regression model for feature selection
    lasso_model = Lasso(alpha=0.1)

    # Fit the Lasso model to the data
    lasso_model.fit(X, y)

    # Select features based on non-zero coefficients from Lasso model
    selected_features_lasso = X.columns[lasso_model.coef_ != 0]

    # Create a SelectFromModel object with linear regression as the base model
    selector = SelectFromModel(LinearRegression(), threshold=0.1)

    # Fit the selector to the data
    selector.fit(X, y)

    # Select features based on the given threshold
    selected_features_linear = X.columns[selector.get_support()]

    # Print the selected features
    print("Selected Features (Lasso):")
    print(selected_features_lasso)

    print("Selected Features (Linear Regression):")
    print(selected_features_linear)
    selected_features_lasso = np.append(selected_features_lasso, 'market_value_in_eur')
    return df[selected_features_lasso]


