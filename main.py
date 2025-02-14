import numpy as np
import pickle
import streamlit as st

#loading the model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

def heart_disease_prediction(input_data):
    numpy_array = np.asarray(input_data)

    numpy_array_reshaped = numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(numpy_array_reshaped)

    if prediction[0]==1:
        return "Presence of Heart Disease"
    elif prediction[0]==0:
        return "Absence of Heart Disease"
    
def main():

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #006400, #228B22);  /* Soothing deep green gradient */
            font-family: 'Helvetica', sans-serif;
            color: #f0f0f0;  /* Light text color for contrast */
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #ffffff;  /* White text color for headings */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    

    st.title("Heart Disease Prediction App")

    Age = st.number_input("Age", min_value=1, max_value=100, value=25)
    sex = st.radio("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    chest_pain_type = st.selectbox("Chest pain type", [1, 2, 3, 4])
    BP = st.number_input("BP", min_value=80, max_value=200, value=120)
    Cholesterol = st.number_input("Cholesterol", min_value=100, max_value=500, value=200)
    FBS_over_120 = st.radio("FBS over 120", [0, 1])
    EKG_results = st.selectbox("EKG results", [0, 1, 2])
    Max_HR = st.number_input("Max HR", min_value=60, max_value=200, value=100)
    Exercise_angina = st.radio("Exercise angina", [0, 1])
    ST_depression = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=0.0)
    Slope_of_ST = st.selectbox("Slope of ST", [1, 2, 3])
    Number_of_vessels_fluro = st.selectbox("Number of vessels fluro", [0, 1, 2, 3])
    Thallium = st.selectbox("Thallium", [3, 6, 7])

    diagnosis = ""

    if st.button("Predict"):
        input_data = [Age, sex, chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, Max_HR, Exercise_angina, ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium]

        diagnosis = heart_disease_prediction(input_data)

    st.success(diagnosis)

if __name__ == "__main__":
    main()