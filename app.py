import streamlit as st
import streamlit.components.v1 as stc


from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Customer Promotion Prediction App </h1>
		    <h4 style="color:white;text-align:center;">Marketing Team</h4>
		    </div>
            """

desc_temp = """
            ### Customer Promotion Prediction App
            the Customer Promotion Prediction App aims to provide businesses with valuable insights into customer behavior and help optimize their promotional strategies.
            #### Data Source
            - https://www.kaggle.com/datasets/datascientistanna/customers-dataset
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """

def main():
    # st.title("Main App")
    stc.html(html_temp)

    menu = ["Home","Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning":
        st.subheader("Machine Learning Section")
        run_ml_app()
    

if __name__ == '__main__':
    main()