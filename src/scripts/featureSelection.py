import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor



def featureSelection(df):
    X = df.drop('market_value_in_eur', axis=1)  
    y = df['market_value_in_eur']  
    
 
    # Define the model
    model = ExtraTreesRegressor()
    
    # Fit the model
    model.fit(X, y)
    
    # Get the importance of the features
    importance = model.feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    
    # Sort the DataFrame by importance
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Print the feature importance
    print(feature_importance)
    
    # Select the top features
    selected_features = feature_importance['Feature'][:15].values
    
    selected_features = np.append(selected_features, 'market_value_in_eur')
    return df[selected_features]


