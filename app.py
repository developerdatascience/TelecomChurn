import json
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import src.utils as ut
import src.categorical as cat


from flask import Flask, render_template, jsonify, app, url_for, request

app = Flask(__name__)

clf = joblib.load(os.path.join("models/", "randomforest_0.pkl"))
base_data = pd.read_csv("outputs/base_data.csv")

@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        gender = request.form.get("gender")
        senior_citizen = request.form.get("senior-citizen")
        partner = request.form.get("partner")
        dependents = request.form.get("dependents")
        tenure = request.form.get("tenure")
        phone = request.form.get("phone")
        MultipleLines = request.form.get("multiple-lines")
        InternetService = request.form.get("internet")
        OnlineSecurity = request.form.get("online-security")
        OnlineBackup = request.form.get("online-backup")
        DeviceProtection = request.form.get("device-protection")
        TechSupport = request.form.get("tech-support")
        StreamingTV = request.form.get("streaming-tv")
        StreamingMovies = request.form.get("streaming-movies")
        Contract = request.form.get("contract")
        PaperlessBilling = request.form.get("paperlessbilling")
        PaymentMethod = request.form.get("paymentmethod")
        MonthlyCharges = request.form.get("monthlycharges")
        TotalCharges = request.form.get("totalcharges")
        data = {"gender": gender, "SeniorCitizen": senior_citizen, 
                "Partner": partner, "Dependents": dependents, "tenure": tenure, "PhoneService": phone, 
                "MultipleLines": MultipleLines, "InternetService": InternetService, 'OnlineSecurity': OnlineSecurity,
                "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
                "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies, "Contract": Contract, 
                "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
                }
        form_data = pd.DataFrame(data=data, index=[0])
        ext_df = pd.concat([base_data, form_data], ignore_index=True)
        ext_df = ext_df[ext_df["TotalCharges"] != " "]
        ext_df["SeniorCitizen"] = pd.to_numeric(ext_df["SeniorCitizen"])
        ext_df["tenure"] = pd.to_numeric(ext_df["tenure"])
        ext_df["MonthlyCharges"] = ext_df["MonthlyCharges"].astype("float64")
        ext_df["TotalCharges"] = pd.to_numeric(ext_df["TotalCharges"])
        print(ext_df.dtypes)
        cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_feats = ["gender","Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                    "OnlineBackup", "DeviceProtection", "DeviceProtection", "TechSupport", "StreamingTV", 
                    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
                    ]
        ut.drop_bad_col(ext_df)
        ext_df.to_csv("outputs/base_data.csv", index=False)

        encoder = cat.CategoricalFeatures(
            df= ext_df,
            encoding_type= 'label',
            categorical_features= cat_feats,
            handle_na=True
        )
        enc_df = encoder.fit_transform()
        scaler = MinMaxScaler()
        enc_df[cols_to_scale] = scaler.fit_transform(enc_df[cols_to_scale])

        preds = clf.predict(enc_df.tail(1))[0]
        probability = clf.predict_proba(enc_df.tail(1))[:, 1]

        if preds == "Yes":
            output1 = "This Customer will Churn"
            output2 = f"Confidence: {probability}"
        else:
            output1 = "This Customer will not Churn"
            output2 = f"Confidence: {probability}"
        # return data
        
    return render_template("index.html", prediction_txt = output1)

    


if __name__ == "__main__":
    app.run(debug=True)
