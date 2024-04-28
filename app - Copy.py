import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgbpipe.joblib')

about =  st.sidebar.radio("select your option" , ("MODEL" , "TEAM"))

if about == "MODEL":
 st.title('Titanic Survival Prediction')

 passengerid = st.text_input("Input Passenger ID", '123456') 
 pclass = st.selectbox("Choose class", [1,2,3])
 name  = st.text_input("Input Passenger Name", 'John Smith')
 sex = st.radio("Choose gender", ['male','female'])
 age = st.slider("Choose age",0,100)
 sibsp = st.slider("Choose siblings",0,10)
 parch = st.radio("Choose parch",[0,1,2])
 ticket = st.text_input("Input Ticket Number", "12345") 
 fare = st.number_input("Input Fare Price", 0,1000)
 cabin = st.text_input("Input Cabin", "C52") 
 embarked = st.radio("Did they Embark?", ['S','C','Q'])
 
 def predict(): 
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.info('Passenger Survived')
    else: 
        st.info('Passenger did not Survive') 

 trigger = st.button('Predict', on_click=predict)
 
if about  == "TEAM":
    st.title("TEAM")
    st.subheader("SARTHAK JAISWAL 21BIT0335")
    st.subheader("CHAITANYA VILAS KADAM 21BIT0369")
    st.subheader("AYMAAN SAAMIR PERWEZ 21BIT0388")
    st.subheader("BISWARUP SEN 22BIT0227")
    st.subheader("SHREYANSH SINGH 21BIT0604")
    






