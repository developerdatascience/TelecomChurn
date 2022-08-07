import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler


telecom_data = pd.read_csv("inputs/telecom_raw.csv")

print(telecom_data.shape)

# Dropping the rows having TotalCharges null

modified_telecom_data = telecom_data[telecom_data.TotalCharges != ' ']
print("Missing values in TotalCharges Column:",modified_telecom_data.shape)

modified_telecom_data.TotalCharges = pd.to_numeric(modified_telecom_data.TotalCharges)

tenure_churn_no = modified_telecom_data[modified_telecom_data["Churn"]=='No'].tenure
tenure_churn_yes = modified_telecom_data[modified_telecom_data.Churn =='Yes'].tenure

# plt.hist([tenure_churn_no, tenure_churn_yes],rwidth=.95, color=['red', 'green'], label=['Churn=No', 'Churn=Yes'])
# plt.legend()
# plt.show()

modified_telecom_data.replace('No internet service', 'No', inplace= True)
modified_telecom_data.replace('No phone service', 'No', inplace= True)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService','MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn'
                ]

yes_no_encoding = {'Yes': 1, 'No': 0}

for column in yes_no_columns:
    modified_telecom_data[column].replace(yes_no_encoding, inplace=True)

for col in modified_telecom_data:
    if modified_telecom_data[col].dtypes == 'object':
        print(f'{col}: {modified_telecom_data[col].unique()}')

modified_telecom_data["gender"].replace({"Male":0, "Female":1}, inplace= True)

# Creating new dataframe with encoded categorical variables

df_encoded = pd.get_dummies(data=modified_telecom_data, columns=["InternetService", "Contract", "PaymentMethod"])

def drop_bad_col():
    for col in df_encoded:
        if "Unnamed: 0" in col:
            print("===========Dropping Unnamed: 0 column============")
            return df_encoded.drop("Unnamed: 0", axis=1, inplace=True)
        else:
            print("===========No Unnamed: 0 column found============")
            return df_encoded

drop_bad_col()


cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]


scaler = MinMaxScaler()

df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])


print("=========Exporting the Transformed data to csv format============")
df_encoded.to_csv("inputs/Processed_data.csv", index=False)
print("=========Data Exported to csv format===============")


