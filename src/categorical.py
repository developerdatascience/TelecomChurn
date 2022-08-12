<<<<<<< HEAD
import pandas as pd
from sklearn import preprocessing

from typing import List


class Categorical:
    def __init__(self, df: pd.DataFrame, categorical_features: List[str], encoding_type: str, handle_na: bool = False)-> None:
=======
from sklearn import preprocessing
import pandas as pd

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False ):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
>>>>>>> 70e8e5896243982bb44e15775b30025dfba1c5fe
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()

        if self.handle_na:
            for c in self.cat_feats:
<<<<<<< HEAD
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-99999")
        self.output_df = self.df.copy(deep=True)
    
=======
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep = True)
>>>>>>> 70e8e5896243982bb44e15775b30025dfba1c5fe

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
<<<<<<< HEAD
            self.output_df.loc[:, c] = lbl.fit_transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        else:
            raise Exception("Encoding type not understood")
    


    def transform(self, dataframe):
=======
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    

    def fit_transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        else:
            raise Exception("Encoding Type Not Understood")
    

    def transform(self, dataframe: pd.DataFrame)-> pd.DataFrame:
>>>>>>> 70e8e5896243982bb44e15775b30025dfba1c5fe
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-99999")
        
<<<<<<< HEAD
        if self.enc_type=="label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
                return dataframe
        else:
            raise Exception("Encoding type not understood")
=======
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        else:
            raise Exception("Encoding type not understood")

        

        
    


>>>>>>> 70e8e5896243982bb44e15775b30025dfba1c5fe
