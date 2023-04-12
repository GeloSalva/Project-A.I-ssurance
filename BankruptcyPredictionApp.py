# General Libraries
import pickle
import pandas as pd
import shap

# Model deployment
from flask import Flask
import streamlit as st

model = pickle.load(open('gb_tk.pkl', 'rb'))
X_holdout = pd.read_csv('PSE.csv', index_col=0)
holdout_transactions = X_holdout.index.to_list()

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
    options = holdout_transactions)


def predict_if_bankrupt(transaction_id):    
    transaction = X_holdout.loc[transaction_id].values.reshape(1, -1)
    prediction_num = model.predict(transaction)[0]
    pred_map = {1: 'Bankrupt', 0: 'Not Bankrupt'}
    prediction = pred_map[prediction_num]
    display_summary(transaction)
    return prediction

def display_summary(transaction):
    explainer = shap.TreeExplainer(model, feature_names=X_holdout.columns)
    shap_values = explainer.shap_values(transaction, check_additivity=False)
    st.write(shap.summary_plot(shap_values, X_holdout.columns, plot_type='bar'), unsafe_allow_html = True)
    st.write('Test')

if st.button("Predict"):
    output = predict_if_bankrupt(choice)
    
    if output == 'Bankrupt':
        st.error('This Company may BANKRUPT', icon="🚨")
    elif output == 'Not Bankrupt':
        st.success('This Company is NOT at risk of bankruptcy!', icon="✅")
