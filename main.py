from flask import Flask
import os

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle
 
app = Flask(__name__)

#df = joblib.load('df_data_dash.joblib')
df = pd.read_pickle("./pickle/df_data_dash.pkl")
#model = joblib.load('gbm.joblib')
model = pickle.load(open('./pickle/gbm.sav', 'rb'))
feats = [f for f in df.columns if f not in ['TARGET','SKIDCURR','index']]


@app.route('/')
def hello():
    return 'calcule du score , /predict/SKIDCURR du client'

 
@app.route('/predict/<prediction>')
def predict(prediction):
    if int(prediction) in df["SKIDCURR"].values:
        data = (df[df['SKIDCURR'] == int(prediction)][feats])
        score ={"SKIDCURR" :prediction , "score" : round(float(model.predict(data)),2)} #calcule du score
    else:
        score ={"SKIDCURR" :prediction , "score" : "Client non trouve"}
    return score
 
if __name__ == '__main__':
      app.run(debug=True, host='127.0.0.1', port=8080)