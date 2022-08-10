import pandas as pd


def replace_no_internet_service(df: pd.DataFrame)-> pd.DataFrame:
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)
    return df


def drop_bad_col(df: pd.DataFrame)-> pd.DataFrame:
    for col in df:
        if "Unnamed: 0" in col:
            print("===========Dropping Unnamed: 0 column============")
            return df.drop("Unnamed: 0", axis=1, inplace=True)
        else:
            print("===========No Unnamed: 0 column found============")
            return df