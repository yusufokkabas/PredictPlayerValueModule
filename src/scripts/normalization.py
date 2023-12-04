from sklearn.preprocessing import MinMaxScaler


def normalizeData(data):
    # Create the MinMaxScaler object
    scaler = MinMaxScaler()
    print(data.columns)
    # Select the columns you want to normalize
    selected_columns = ['statistics.game_infos.minutes']

    # Normalize the selected columns
    data[selected_columns] = scaler.fit_transform(data[selected_columns])
    print("Minutes columns has succesfully normalized\n" , data[selected_columns])
    data = data.dropna()
    print("befor",data.info)
    data = data.apply(lambda row: row * row['statistics.game_infos.minutes'] if row['statistics.game_infos.minutes'] != 0 else row, axis=1)
    print("after",data.info)
    return data
