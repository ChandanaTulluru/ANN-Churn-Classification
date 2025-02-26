import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle


# load ANN train model, pickle file, scaler, and encoder
model = tf.keras.models.load_model('model.h5')
with open('labelEncoder_gender.pkl', 'rb') as f:
    labelEncoder_gender = pickle.load(f)
with open('ohe_geography.pkl', 'rb') as f:
    ohe_geography = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Streamlit app
st.title('Customer Churn Prediction')
# geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
geography = st.selectbox('Geography', ohe_geography.categories_[0])
gender = st.selectbox('Gender', labelEncoder_gender.classes_)
# age = st.number_input('Age', min_value=18, max_value=100)
age = st.slider('Age', min_value=18, max_value=100)
tenure = st.slider('Tenure', min_value=0, max_value=10)
credit_score = st.number_input('Credit Score')
balance = st.number_input('Balance', min_value=0.0, max_value=100000.0)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [labelEncoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# encode geography
encoded_geo = ohe_geography.transform([[geography]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns=ohe_geography.get_feature_names_out(['Geography']))
# concatenate encoded geography with input data
input_data = pd.concat([input_data, encoded_geo_df], axis=1)
# scale the data
input_data_scaled = scaler.transform(input_data)
# make prediction
prediction = model.predict(input_data_scaled)
# display prediction
if prediction[0][0] > 0.5:
    st.success('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')
st.write(f'Prediction Probability: {prediction[0][0]:.2f}')
st.write(f'Input Data: {input_data}')