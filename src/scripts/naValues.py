import pandas as pd

def naValues(df):
    # Create a boolean mask for NA values
    na_mask = df.isna().any(axis=1)

    # Filter the DataFrame to show only rows with NA values
    rows_with_na = df[na_mask]

    # Display the rows with NA values
    print(rows_with_na)
    print("Since there is no missing value in our dataset, We do not have to perform handling na values")