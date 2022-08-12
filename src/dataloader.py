import pandas as pd
from categorical import CategoricalFeatures
import utils
from typing import List

from sklearn.preprocessing import MinMaxScaler

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

utils.drop_bad_col(df=full_transformed_data)

cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]

# Scaling Tenure, MonhtlyCharges and TotalCharges
scaler= MinMaxScaler()

full_transformed_data[cols_to_scale] = scaler.fit_transform(full_transformed_data[cols_to_scale])

print(full_transformed_data.head())


# Outputting the processed file to output directory for modelling purpose

full_transformed_data.to_csv("outputs/Processed_data.csv", index=False)



