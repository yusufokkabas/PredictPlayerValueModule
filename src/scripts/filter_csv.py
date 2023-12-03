from scipy import stats
import numpy as np

def filterOutliers(df):
    column_name = 'Pos' 
    values = ['MF','FW','FWMF','MFFW'] 
    filtered_df = df[df[column_name].isin(values)] 
    print(filtered_df)

    # List of numeric columns
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()


    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(filtered_df[numeric_cols]))

    # Identify outliers
    outliers = filtered_df[(z_scores > 3).any(axis=1)]

    # Filter out outliers
    filtered_df = filtered_df.drop(outliers.index)

    filtered_df.to_csv('src/data/filtered_file.csv', index=False)
    print("Succesfully filtered the outliers!!")
    print(filtered_df)

    # Print outliers
    print("Outliers:")
    print(outliers)

    return filtered_df