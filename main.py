import pandas as pd
from ydata_profiling import ProfileReport
from src.scripts.concatFiles import concatFiles
df = pd.read_csv('src/data/cleaned_data.csv')

# Display the first few rows
print(df.head()) 

# Get the dimensions of the DataFrame
print(df.shape)

# Get a summary of the DataFrame
print(df.info())

# Generate descriptive statistics
print(df.describe())

# Get the column names
print(df.columns)

# Get the data types of the columns
print(df.dtypes)

