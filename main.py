import pandas as pd
from src.scripts.concatFiles import concatFiles
from src.scripts.naValues import naValues
from src.scripts.featureSelection import featureSelection
#df = concatFiles()
df = pd.read_csv('src/data/cleaned_data.csv')
#naValues(df)
featureSelection(df)


