import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

 # Load the trained model 
model = tf.keras.models.load_model('model.h5');

### Load the encoder and scaler 
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gendor = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title("Customer churn Prediction")

#user input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gendor.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credict_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0, 10)
num_of_preducts = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


## Prepare the input data 
input_data = {
    'CreditScore': [credict_score],
    'Gender': [label_encoder_gendor.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_preducts],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

## Encode the Geography value 
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

## Combine one-hot encoded with input data 
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale the input data 
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probs = prediction[0][0]

if prediction_probs > 0.5:
    st.write('The Customer is likley to churn');
else:
    st.write('The Customer is not likly to churn.')