import pandas as pd
import utils
from categorical import Categorical
from sklearn.preprocessing import MinMaxScaler

def processing(raw_data_path: str, filename: str)-> pd.DataFrame:
    """
    raw_data_path: path of input file to be processsed eg. inputs/
    filename: name of raw data file in csv format.
    """
    df = pd.read_csv(raw_data_path+filename)
    print(df.shape)
    # Dropping the rows having TotalCharges null

    non_missing_df = df[df.TotalCharges != ' ']
    print("Missing values in TotalCharges Column:",non_missing_df.shape)

    non_missing_df.TotalCharges = pd.to_numeric(non_missing_df.TotalCharges)
    utils.replace_no_internet_service(df= non_missing_df)
    utils.drop_bad_col(df= non_missing_df)

    cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = MinMaxScaler()
    non_missing_df[cols_to_scale] = scaler.fit_transform(non_missing_df[cols_to_scale])

    print("=========Exporting the Transformed data to csv format============")
    non_missing_df.to_csv("inputs/Processed_data.csv", index=False)
    print("==========Data Exported to csv format===============")
    return non_missing_df

if __name__=="__main__":
    processing(raw_data_path="inputs/", filename="telecom_raw.csv")



