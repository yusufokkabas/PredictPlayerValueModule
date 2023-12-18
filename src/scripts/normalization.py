from sklearn.preprocessing import MinMaxScaler
import numpy as np


def normalizeData(data):
    # Create the MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Select the integer columns
    selected_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns.remove('market_value_in_eur')
    # Normalize the selected columns
    data[selected_columns] = scaler.fit_transform(data[selected_columns])
    print("Numeric columns has succesfully normalized\n" , data[selected_columns])
    data.to_csv('src/data/normalizedData.csv')
    return data
