from concurrent.futures import process
from operator import mod
from statistics import mode
import pandas as pd
import utils


from sklearn.preprocessing import MinMaxScaler


def processing(raw_data_path: str, filename: str)-> pd.DataFrame:
    """
    raw_data_path: Path of input file path eg, inputs/
    filename: Filename of raw data in csv format
    """
    df = pd.read_csv(raw_data_path+filename)

    # Dropping the rows having TotalCharges null

    mod_df = df[df.TotalCharges != ' ']
    mod_df.TotalCharges = pd.to_numeric(mod_df.TotalCharges)
    utils.drop_bad_col(df=mod_df)
    utils.replace_no_internet_service(df=mod_df)

    cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = MinMaxScaler()

    mod_df[cols_to_scale] = scaler.fit_transform(mod_df[cols_to_scale])


    print("=========Exporting the Transformed data to csv format============")
    mod_df.to_csv("inputs/Processed_data.csv", index=False)
    print("==========Data Exported to csv format===============")

    return mod_df


if __name__=="__main__":
    processing(raw_data_path="inputs/", filename="telecom_raw.csv")

