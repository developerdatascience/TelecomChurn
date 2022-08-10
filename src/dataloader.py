import pandas as pd
from categorical import CategoricalFeatures

from typing import List


df = pd.read_csv("inputs/telecom_raw.csv")
print(df.head())

modified_telecom_data = df[df.TotalCharges != ' ']
print("Missing values in TotalCharges Column:",modified_telecom_data.shape)

modified_telecom_data.TotalCharges = pd.to_numeric(modified_telecom_data.TotalCharges)
modified_telecom_data.drop("customerID", axis=1, inplace=True)

cat_feats: List[str] = modified_telecom_data.columns[modified_telecom_data.dtypes== "object"]

# Label Encoding

cats = CategoricalFeatures(
    df= modified_telecom_data,
    categorical_features= cat_feats,
    encoding_type= 'label',
    handle_na=False
)

full_transformed_data = cats.fit_transform()
print(full_transformed_data.head())

