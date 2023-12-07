import pandas as pd

def naValues(df):
    # Apply fillna and mean functions to each column
    if 'market_value_in_eur' in df.columns:
        df['market_value_in_eur'] = df['market_value_in_eur'].dropna().astype(int)
    df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
    df_dropped = df #.dropna()
    print("Remain NAValue checked and if there is a null value, it is filled with NA values" ,df_dropped.info)
    return df