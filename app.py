import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np


app = Flask(__name__)

app.route("/")
def home():
    return render_template("churn_form.html")


app.route('/predict_api', methods=['POST'])
def predict():
    pass