# General Libraries
import pickle
import pandas as pd

# Model deployment
from flask import Flask
import streamlit as st

model = pickle.load(open('gb_tk.pkl', 'rb'))
PSE = pd.read_csv('PSE.csv', index_col=0)
PSE_transactions = PSE.index.to_list()

st.title("Company Bankruptcy Prediction")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> Company Bankruptcy Detection ML App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

#adding a selectbox
choice = st.selectbox(
    "Select Transaction Number:",
    options = PSE_transactions)


def predict_if_bankrupt(company):    
    transaction = PSE.loc[company].values.reshape(1, -1)
    prediction_num = model.predict(transaction)[0]
    pred_map = {1: 'Bankrupt', 0: 'Not Bankrupt'}
    prediction = pred_map[prediction_num]
    return prediction

if st.button("Predict"):
    output = predict_if_bankrupt(choice)
    
    if output == 'Bankrupt':
        st.error('This Company may BANKRUPT', icon="🚨")
    elif output == 'Not Bankrupt':
        st.success('This Company is not at risk of bankruptcy!', icon="✅")