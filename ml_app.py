import streamlit as st
import numpy as np

#load ML package
import joblib
import os


attribute_info = """
                 - Gender: Male and Female
                 - Age: 0-100
                 - Annual_Income($): Annual Income of Customer
                 - Spending Score: 0-100
                 - Department: Healthcare, Engineer, Lawyer, Entertainment, Artist, Executive, Doctor, Homemaker, Marketing
                 - Work Experience: 0-100 Year
                 - Family Size: 0-10 
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
    annual_income = st.number_input("Annual_Income", 1, 100000000000)
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




    st.subheader('Prediction Result')


    cluster_number = predict_cluster(spending_score,annual_income)


    # st.write(prediction)
    # st.write(pred_proba)


    if cluster_number == 1:
        st.success("Beli Mobil gratis Liburan ke Jepang")
        pred_probability_score = {'Promoted':cluster_number,'Level':"VIP"}
        st.write(pred_probability_score)
    elif cluster_number == 2:
        st.success("Bundling produk A dan produk B dengan harga diskon ")
        pred_probability_score = {'Promoted':cluster_number,'Level':"Premium"}
        st.write(pred_probability_score)
    elif cluster_number == 3:
        st.success("Pemberian diskon produk, seperti Makanan atau Minuman.")
        pred_probability_score = {'Promoted':cluster_number,'Level':"Low"}
        st.write(pred_probability_score)
    else:
        st.warning("Undian kupon berhadiah")
        pred_probability_score = {'Promoted':cluster_number,'Level':"Standart"}
        st.write(pred_probability_score)
