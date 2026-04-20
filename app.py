
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
# Assuming 'best_model.pkl' is in the same directory
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the original dataset to get unique categorical values for Label Encoding
# This is crucial because the LabelEncoder in the notebook was overwritten
try:
    original_df = pd.read_csv('/Salary_Dataset_.csv')
except FileNotFoundError:
    st.error("Error: Original dataset '/Salary_Dataset_.csv' not found. Please ensure it's in the correct path.")
    st.stop()

# Handle null values in the original_df for categorical columns to match training preprocessing
for column in original_df.columns:
    if original_df[column].dtype == 'object':
        original_df[column] = original_df[column].fillna(original_df[column].mode()[0])
    else:
        original_df[column] = original_df[column].fillna(original_df[column].mean())


# Create a dictionary to store LabelEncoders for each categorical column
label_encoders = {}
categorical_cols = ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']

for col in categorical_cols:
    if col in original_df.columns:
        le = LabelEncoder()
        le.fit(original_df[col].astype(str).unique()) # Convert to string to handle potential mixed types
        label_encoders[col] = le


st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for features
rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)

# For categorical features, use selectbox and then encode
company_name_options = sorted(original_df['Company Name'].astype(str).unique())
company_name_input = st.selectbox('Company Name', company_name_options)

job_title_options = sorted(original_df['Job Title'].astype(str).unique())
job_title_input = st.selectbox('Job Title', job_title_options)

salaries_reported = st.number_input('Salaries Reported', min_value=1, value=5, step=1)

location_options = sorted(original_df['Location'].astype(str).unique())
location_input = st.selectbox('Location', location_options)

employment_status_options = sorted(original_df['Employment Status'].astype(str).unique())
employment_status_input = st.selectbox('Employment Status', employment_status_options)

job_roles_options = sorted(original_df['Job Roles'].astype(str).unique())
job_roles_input = st.selectbox('Job Roles', job_roles_options)

if st.button('Predict Salary'):
    # Encode categorical inputs
    try:
        encoded_company_name = label_encoders['Company Name'].transform([str(company_name_input)])[0]
        encoded_job_title = label_encoders['Job Title'].transform([str(job_title_input)])[0]
        encoded_location = label_encoders['Location'].transform([str(location_input)])[0]
        encoded_employment_status = label_encoders['Employment Status'].transform([str(employment_status_input)])[0]
        encoded_job_roles = label_encoders['Job Roles'].transform([str(job_roles_input)])[0]
    except ValueError as e:
        st.error(f"Error encoding categorical input: {e}. Please ensure selected categories were present in training data.")
        st.stop()

    # Create a DataFrame for prediction
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': encoded_company_name,
        'Job Title': encoded_job_title,
        'Salaries Reported': salaries_reported,
        'Location': encoded_location,
        'Employment Status': encoded_employment_status,
        'Job Roles': encoded_job_roles
    }])

    # Make prediction
    prediction = model.predict(input_data)

    st.success(f'Predicted Salary: {prediction[0]:,.2f} INR')
