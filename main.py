import pandas as pd
from src.scripts.concatFiles import concatFiles
from src.scripts.filter_csv import filterMFDF
df = pd.read_csv('src/data/cleaned_data.csv')
mfdf=pd.read_csv('src/data/player_data.csv',sep=';')
# Display the first few rows
print(df.head()) 
filterMFDF(mfdf)
# Get the dimensions of the DataFrames
print(df.shape)

# Get a summary of the DataFrame
print(df.info())

# Generate descriptive statistics
print(df.describe())

# Get the column names
print(df.columns)

# Get the data types of the columns
print(df.dtypes)

