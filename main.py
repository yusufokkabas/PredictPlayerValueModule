import pandas as pd
from src.scripts.concatFiles import concatFiles
from src.scripts.filter_csv import filterMFDF
from src.scripts.naValues import naValues
from src.scripts.featureSelection import featureSelection
#df = concatFiles()
df = pd.read_csv('src/data/cleaned_data.csv')
mfdf=pd.read_csv('src/data/player_data.csv',sep=';') 
filterMFDF(mfdf)
#naValues(df)
featureSelection(df)


