import pandas as pd
from sklearn import preprocessing

from typing import List


class Categorical:
    def __init__(self, df: pd.DataFrame, categorical_features: List[str], encoding_type: str, handle_na: bool = False)-> None:
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-99999")
        self.output_df = self.df.copy(deep=True)
    

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.fit_transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        else:
            raise Exception("Encoding type not understood")
    


    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-99999")
        
        if self.enc_type=="label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
                return dataframe
        else:
            raise Exception("Encoding type not understood")