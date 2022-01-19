
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

import plotly.graph_objects as go
import plotly.express as px

import joblib
import pickle


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st


dossier = 'C:/Users/J45170/Documents/GitHub/git_collab_P7/dashboard/'
PATH = os.getcwd()+'/'


#df = joblib.load('df_data_dash.joblib')
df = pd.read_pickle("./pickle/df_data_dash.pkl")
#model = joblib.load('gbm.joblib')
model = pickle.load(open('./pickle/gbm.sav', 'rb'))
feats = [f for f in df.columns if f not in ['TARGET','SKIDCURR','index']]


st.title('Informations client et score crédit ')

input_client = '407942'


#bandeau de gauche

sb = st.sidebar # add a side bar 
sb.markdown('**Sélection du client:**')
np.random.seed(13) # one major change is that client is directly asked as input since sidebar
client = df['SKIDCURR'].sample(50).sort_values()
radio = sb.radio('', ['Liste', 'Numéro'])
if radio == 'Liste': # Choice choose preselected seed13 or a known client ID
    input_client = sb.selectbox('sélection du client', client)
if radio == 'Numéro':
        input_client = int(sb.text_input('ID', value=407942))



# page pricipale

data = (df[df['SKIDCURR'] == input_client][feats])

st.write('Numero du client : ' , input_client)
st.write('EXTSOURCE1 : ' , float(df[df['SKIDCURR'] == input_client]["EXTSOURCE1"]))
st.write('EXTSOURCE2 : ' , float(df[df['SKIDCURR'] == input_client]["EXTSOURCE2"]))
st.write('EXTSOURCE3 : ' , float(df[df['SKIDCURR'] == input_client]["EXTSOURCE3"]))
st.write('DAYSEMPLOYED : ' , float(df[df['SKIDCURR'] == input_client]["DAYSEMPLOYED"]))


#if df[df['SKIDCURR'] == input_client]["EXTSOURCE1"].isnull().bool():
  #  value_EXTSOURCE1 = 0
#else:
 #  value_EXTSOURCE1 =float(df[df['SKIDCURR'] == input_client]["EXTSOURCE1"])

slide_EXTSOURCE1 = st.slider('EXTSOURCE1', min_value = 0.0 , max_value= 1.0 , value = float(df[df['SKIDCURR'] == input_client]["EXTSOURCE1"]))
st.write("EXTSOURCE1 : ", slide_EXTSOURCE1)

data['EXTSOURCE1'] = slide_EXTSOURCE1

score_sk = float(model.predict(data))


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = score_sk,
    title = {'text': "Score de solvabilité"},
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge = {
        'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "black"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 3,
        'bordercolor': "black",
        'steps': [
            {'range': [0, 0.3], 'color': 'green'},
            {'range': [0.3, 0.6], 'color': 'yellow'},
            {'range': [0.6, 1], 'color': 'red'}
            ],}))

st.plotly_chart(fig, use_container_width=True)    


fig_2 = px.histogram(df,x="EXTSOURCE1",barmode="group",histnorm='percent')
if df[df['SKIDCURR'] == input_client]["EXTSOURCE1"].isnull().bool():
    fig_2.add_annotation(dict(font=dict(color='red',size=20),text="Pas de donnée pour ce client",
                  xref="paper", yref="paper" ))
else:
   fig_2.add_vline(x=float(df[df['SKIDCURR'] == input_client]["EXTSOURCE1"]), line_color="black")
st.plotly_chart(fig_2, use_container_width=True)   


fig_3 = px.histogram(df,x="EXTSOURCE2",barmode="group",histnorm='percent')
if df[df['SKIDCURR'] == input_client]["EXTSOURCE2"].isnull().bool():
    fig_3.add_annotation(dict(font=dict(color='red',size=20),text="Pas de donnée pour ce client",
                  xref="paper", yref="paper" ))
else:
    fig_3.add_vline(x=float(df[df['SKIDCURR'] == input_client]["EXTSOURCE2"]), line_color="black")
st.plotly_chart(fig_3, use_container_width=True)   

fig_4 = px.histogram(df,x="EXTSOURCE3",barmode="group",histnorm='percent')
if df[df['SKIDCURR'] == input_client]["EXTSOURCE3"].isnull().bool():
    fig_4.add_annotation(dict(font=dict(color='red',size=20),text="Pas de donnée pour ce client",
                  xref="paper", yref="paper" ))
else:
    fig_4.add_vline(x=float(df[df['SKIDCURR'] == input_client]["EXTSOURCE3"]), line_color="black")
st.plotly_chart(fig_4, use_container_width=True)   

fig_4 = px.histogram(df,x="DAYSEMPLOYED",barmode="group",histnorm='percent')
if df[df['SKIDCURR'] == input_client]["DAYSEMPLOYED"].isnull().bool():
    fig_4.add_annotation(dict(font=dict(color='red',size=20),text="Pas de donnée pour ce client",
                  xref="paper", yref="paper" ))
else:
    fig_4.add_vline(x=float(df[df['SKIDCURR'] == input_client]["DAYSEMPLOYED"]), line_color="black")
st.plotly_chart(fig_4, use_container_width=True)   
