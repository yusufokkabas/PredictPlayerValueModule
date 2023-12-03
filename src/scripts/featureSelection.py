import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel




def featureSelection(df):
    df=df.dropna()
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    print(df.columns)
    X = df.drop('market_value_in_eur', axis=1)  
    y = df['market_value_in_eur']  
    
 
    lasso_model = Lasso(alpha=0.1)

    lasso_model.fit(X, y)

    selected_features_lasso = X.columns[lasso_model.coef_ != 0]

    selector = SelectFromModel(LinearRegression(), threshold=0.1)

    selector.fit(X, y)

    # Print the selected features
    print("Selected Features (Lasso):")
    print(selected_features_lasso)

    selected_features_lasso = np.append(selected_features_lasso, 'market_value_in_eur')
    return df[selected_features_lasso]


