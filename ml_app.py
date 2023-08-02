import streamlit as st
import numpy as np

#load ML package
import joblib
import os


attribute_info = """
                 - Department: Sales & Marketing, Operations, Technology, Analytics, R&D, Procurement, Finance, HR, Legal
                 - Region: region 1 - region 34
                 - Educaiton: Below Secondary, Bachelor's, Master's & above
                 - Gender: Male and Female
                 - Recruitment Channel: Referred, Sourcing, Others
                 - No of Training: 1-10
                 - Age: 10-60
                 - Previous Year Rating: 1-5
                 - Length of Service: 1-37 Month
                 - Awards Won: 1. Yes, 0. No
                 - Avg Training Score: 0-100
                 """


def get_value(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
        
@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model

def predict_cluster(spending_score,annual_income):

    # Data baru yang ingin diprediksi
    data_baru = [[spending_score,annual_income]]  # Ganti data ini dengan data baru yang ingin diprediksi
    scaler_loaded = load_model("standard_scaler.pkl")
    data_baru_scaled = scaler_loaded.transform(data_baru)
    kmeans_loaded = load_model("kmeans_model.pkl")
    # Perform clustering prediction using the loaded K-Means model
    cluster_prediction = kmeans_loaded.predict(data_baru_scaled)

    # Get the cluster number
    cluster_number = int(cluster_prediction[0])

    return cluster_number

def run_ml_app():
    st.subheader("ML section")

    with st.expander("Attribute Info"):
        st.markdown(attribute_info)
        
    # Menampilkan input data dari user menggunakan Streamlit
    st.subheader("Input Your Data")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    annual_income = st.number_input("Annual_Income", 1, 1000000)
    spending_score = st.number_input("Spending Score", 1, 100)
    age = st.number_input("Age", 1, 100)
    Profession = st.selectbox('Profession', ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing'])
    work_experience = st.number_input("Work Experience", 1, 37)
    family_size = st.number_input("Family Size", 1, 10)

    with st.expander("Your Selected Options"):
        result = {
            'Gender': gender,
            'Annual_Income': annual_income,
            'Spending Score': spending_score,
            'Age': age,
            'Profession': Profession,
            'Work Experience': work_experience,
            'Family Size': family_size
        }
        st.write(result)



    # st.write(encoded_result)

    ## prediction section
    st.subheader('Prediction Result')
    # st.write(single_sample)

    cluster_number = predict_cluster(spending_score,annual_income)


    # st.write(prediction)
    # st.write(pred_proba)

    pred_probability_score = {'Promoted':cluster_number}

    if cluster_number == 1:
        st.success("yes nberhasil")
        
        st.write(pred_probability_score)
    if cluster_number == 2:
        st.success("Udin apa kabar")
        
        st.write(pred_probability_score)
    if cluster_number == 3:
        st.success("bolehlah")
        
        st.write(pred_probability_score)
    if cluster_number == 4:
        st.success("mau apa anda")
        
        st.write(pred_probability_score)
    else:
        st.warning("Apa aja boleh")
        st.write(pred_probability_score)
