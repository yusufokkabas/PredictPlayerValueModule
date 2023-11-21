import pandas as pd

def filterOutliers(df):
    
    column_name = 'Pos' 
    values = ['MF','FW','FWMF','MFFW'] 
    filtered_df = df[df[column_name].isin(values)] 
    filtered_df.to_csv('src/data/filtered_file.csv', index=False)
    print("Succesfully filtered the outliers!!")
   # print(filtered_df.info)


