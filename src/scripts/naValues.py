import pandas as pd

def naValues(df):
    # Create a boolean mask for NA values
    na_mask = df.isna().any(axis=1)

    # Filter the DataFrame to show only rows with NA values
    rows_with_na = df[na_mask]
    df_dropped = df.dropna()
    # Display the rows with NA values
    print(df_dropped)
    return df_dropped