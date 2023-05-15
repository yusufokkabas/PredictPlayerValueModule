import pandas as pd

def concatFiles():
# Read the CSV files, specifying the correct delimiter for both files
    player_data = pd.read_csv('src/data/player_statistics.csv', sep=';')
    market_value = pd.read_csv('src/data/players.csv', sep=';')

    # Merge the dataframes using the common property 'player' using a left join
    merged_data = player_data.merge(market_value, on='player', how='left')

    # Drop rows with null values in the 'Market value' column
    merged_data.dropna(subset=['market_value_in_eur'], inplace=True)

    # Write the cleaned data to a new CSV file
    merged_data.to_csv('src/data/cleaned_data.csv', index=False)

    return merged_data
