import pandas as pd

def naValues(df):
    #boolean mask for NA values
    na_mask = df.isna().any(axis=1)
    df_dropped = df.dropna()
    print("Remain NAValue checked and if there is a null value, it is dropped" ,df_dropped.info)
    return df_dropped