import pandas as pd
from sklearn.preprocessing import OneHotEncoder
def featureEncoding(data):
    columns_to_drop = ['current_club_name','Age','player']
    data=data.drop(columns_to_drop,axis=1)
    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder()
    one_hot_encoded = encoder.fit_transform(data[categorical_columns])

    one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, one_hot_encoded_df], axis=1)
    data = data.dropna()
    data.to_csv('src/data/numericData.csv')
    print("Data has succesfully encoded(Categorical to numerical). Removed unnecessary columns.", data.info)
    return data

