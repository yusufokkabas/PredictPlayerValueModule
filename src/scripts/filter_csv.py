import pandas as pd

def filterMFDF(df):
    

    # Specify the column and value to filter
    column_name = 'Pos'  # Replace with the actual column name
    value_to_delete = 'MFDF'  # Replace with the value you want to delete

    # Filter the DataFrame to exclude rows with the specified value in the column
    filtered_df = df[df[column_name] != value_to_delete]

    # Write the filtered data to a new CSV file
    filtered_df.to_csv('src/data/filtered_file.csv', index=False)


