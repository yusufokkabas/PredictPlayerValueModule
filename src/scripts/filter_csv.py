from scipy import stats
import numpy as np

def filterOutliers(df):


    # List of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df[numeric_cols]))

    # Identify outliers
    outliers = df[(z_scores > 5).any(axis=1)]

    # Filter out outliers
    df = df.drop(outliers.index)

    df.to_csv('src/data/filtered_file.csv', index=False)
    print("Succesfully filtered the outliers!!")
    print(df)

    # Print outliers
    print("Outliers:")
    print(outliers)

    return df