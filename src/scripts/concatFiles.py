import pandas as pd

def concatFiles():

    player_data = pd.read_csv('src/data/filtered_file.csv')
    market_value = pd.read_csv('src/data/players.csv', sep=';')

    # common property 'player' using a left join
    merged_data = player_data.merge(market_value, on='player', how='left')

    # Drop rows with null values
    merged_data.dropna(subset=['market_value_in_eur'], inplace=True)

    merged_data.to_csv('src/data/cleaned_data.csv', index=False)
    print("Data merged and cleared that has empty target variable column\n", merged_data.info)
    return merged_data
