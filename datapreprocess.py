import pandas as pd

# Read the CSV files, specifying the correct delimiter for both files
player_data = pd.read_csv('player_data.csv', sep=';')
market_value = pd.read_csv('market_value.csv', sep=';')

# Merge the dataframes using the common property 'player' using a left join
merged_data = player_data.merge(market_value, on='player', how='left')

# Write the concatenated data to a new CSV file
merged_data.to_csv('concatenated_data.csv', index=False)
