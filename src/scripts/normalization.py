from sklearn.preprocessing import MinMaxScaler


def normalizeData(data):
    # Create the MinMaxScaler object
    scaler = MinMaxScaler()
    print(data.columns)
    # Select the columns you want to normalize
    selected_columns = ['PrgP', 'PrgC', 'PrgR']

    # Normalize the selected columns
    data[selected_columns] = scaler.fit_transform(data[selected_columns])
    print("Progress columns has succesfully normalized\n" , data[selected_columns])
    return data
