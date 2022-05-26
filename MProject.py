import streamlit as st
import tensorflow as tf
from keras.models import load_model


import numpy as np
def teachable_machine_classification(report1, file):

    # Load the model
    model = load_model(file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1,37), dtype=np.float32)

    # Load the image into the array
    data = report1
   
    # run the inference
    prediction = model.predict(data)
    #print(prediction)
    return np.argmax(prediction)

st.title("Mortality Prediction in ICU")
st.header("Death% vs notDeath%")
st.text("Upload a list of features as report containing 37 variables of patient")
# Take report and handling logic

ALP = st.number_input("ALP value", 0.0, 1500.0, value = 0.0)
ALT = st.number_input("ALT value", 0.0, 10000.0, value = 0.0)
AST = st.number_input("AST value", 0.0, 16000.0, value = 0.0)
Age = st.number_input("Age value", 0.0, 120.0, value = 0.0)
Albumin = st.number_input("Albumin value", 0.0, 10.0, value = 0.0)
BUN = st.number_input("BUN value", 0.0, 200.0, value = 0.0)
Bilirubin = st.number_input("Bilirubin value", 0.0, 50.0, value = 0.0)
Creatinine = st.number_input("Creatinine value", 0.0, 20.0, value = 0.0)
DiasABP = st.number_input("DiasABP value", 0.0, 120.0, value = 0.0)
FiO2 = st.number_input("FiO2 value", 0.0, 150000.0, value = 0.0)
GCS = st.number_input("GCS value", 0.0, 20.0, value = 0.0)
Glucose = st.number_input("Glucose value", 0.0, 800.0, value = 0.0)
HCO3 = st.number_input("HCO3 value", 0.0, 100.0, value = 0.0)
HR = st.number_input("HR value", 0.0, 250.0, value = 0.0)
K = st.number_input("K value", 0.0, 20.0, value = 0.0)
Lactate = st.number_input("Lactate value", 0.0, 100.0, value = 0.0)
MAP = st.number_input("MAP value", 0.0, 300.0, value = 0.0)
st.write("")
MechVent = st.checkbox("Mechanical Ventilator", value=True)
st.write("")
Mg = st.number_input("Mg value", 0.0, 10.0, value = 0.0)
NIDiasABP = st.number_input("NIDiasABP value", 0.0, 110.0, value = 0.0)
NIMap = st.number_input("NIMap value", 0.0, 140.0, value = 0.0)
NISysABP = st.number_input("NISysABP value", 0.0, 250.0, value = 0.0)
Na = st.number_input("Na value", 100.0, 170.0, value = 100.0)
PaCO2 = st.number_input("PaCO2 value", 10.0, 170.0, value = 10.0)
PaO2 = st.number_input("PaO2 value", 10.0, 140000.0, value = 10.0)
Platelets = st.number_input("Platelets value", 0.0, 1000.0, value = 0.0)
RecordID = st.number_input("RecordID value", 132000.0, 142000.0, value = 132000.0)
RespRate = st.number_input("RespRate value", 0.0, 50.0, value = 0.0)
SaO2 = st.number_input("SaO2 value", 30.0, 110.0, value = 30.0)
SysABP = st.number_input("SysABP value", 0.0, 200.0, value = 0.0)
Temp = st.number_input("Temp value", 15.0, 50.0, value = 15.0)
TroponinI = st.number_input("TroponinI value", 0.0, 50.0, value = 0.0)
TroponinT = st.number_input("TroponinT value", 0.0, 30.0, value = 0.0)
Urine = st.number_input("Urine value", 0.0, 3100.0, value = 0.0)
WBC = st.number_input("WBC value", 0.0, 140.0, value = 0.0)
Weight = st.number_input("Weight value", 0.0, 350.0, value = 0.0)
pH = st.number_input("pH value", 0.0, 130.0, value = 0.0)
report = [[ALP,ALT,AST,Age,Albumin,BUN,Bilirubin,Creatinine,DiasABP,FiO2,GCS,Glucose,HCO3,HR,K,Lactate,MAP,1,Mg,NIDiasABP,NIMap,NISysABP,Na,PaCO2,PaO2,Platelets,RecordID,RespRate,SaO2,SysABP,Temp,TroponinI,TroponinT,Urine,WBC,Weight,pH]]

st.write("")
st.write("")


if st.button('Predict!'):
    st.write("Classifying and predicting the Mortality Rate")
    label = teachable_machine_classification(report, r'C:\Users\KIIT\Desktop\model2')
    if label == 1:
        st.write("The patient needs immediate and urgent care and constant observation with specialized support")
    else:
        st.write("Hooray!! the patient is more likely to survive the ordeal but still this person should be kept under observation.")






