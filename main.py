import numpy as np
import pickle
import streamlit as st

# Load the trained model and scaler
try:
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()


# Function to make predictions
def diabetes_prediction(input_data):
    try:
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Ensure input is numeric
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  # Reshape for model
        std_data = scaler.transform(input_data_reshaped)  # Standardize input
        prediction = loaded_model.predict(std_data)  # Make prediction

        if prediction[0] == 1:
            return "🩸 The person is **diabetic**. Please follow a healthy lifestyle and consult a doctor."
        else:
            return "✅ The person is **not diabetic**. Keep maintaining a healthy routine!"
    except Exception as e:
        return f"⚠️ Error in prediction: {e}"


# Streamlit UI Configuration
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Header with an image
st.image(
    "https://media.istockphoto.com/id/1354249154/photo/diabetic-test-kit.jpg?s=612x612&w=0&k=20&c=Cb6GrkaWqcV1EKVm47Rsi-5Si9HaRjo0aXyiqTlcWoY=",
    use_container_width=True
)

st.title("🩺 DiaGuard: AI-Powered Diabetes Prediction")
st.markdown("### 🔬 AI-Powered Health Assessment")
st.markdown(
    "This tool uses **Machine Learning** to analyze health data and assess the likelihood of diabetes. Enter your details below to get a quick and reliable prediction. 🏥"
)

# Input form for user data
with st.form("diabetes_form"):
    st.markdown("### 📝 Fill in the details below:")
    col1, col2 = st.columns(2)  # Layout with two columns

    with col1:
        Pregnancies = st.number_input("👶 Number of Pregnancies", min_value=0, max_value=20, value=2, step=1)
        Glucose = st.number_input("🩸 Glucose Level (mg/dL)", min_value=0, max_value=300, value=120, step=1)
        BloodPressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=0, max_value=200, value=80, step=1)
        SkinThickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

    with col2:
        Insulin = st.number_input("💉 Insulin Level (IU/mL)", min_value=0, max_value=1000, value=85, step=1)
        BMI = st.number_input("⚖️ BMI (kg/m²)", min_value=0.0, max_value=100.0, value=25.6, step=0.1)
        DiabetesPedigreeFunction = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=2.5,
                                                   value=0.5, step=0.01)
        Age = st.number_input("🎂 Age", min_value=0, max_value=120, value=35, step=1)

    # Submit button
    submit_button = st.form_submit_button("🔍 Predict")

# Prediction logic
if submit_button:
    try:
        # Collect input values
        input_values = [
            Pregnancies, Glucose, BloodPressure,
            SkinThickness, Insulin, BMI,
            DiabetesPedigreeFunction, Age
        ]
        # Make prediction
        result = diabetes_prediction(input_values)
        st.success(result)

        if "diabetic" in result:
            st.warning(
                "⚠️ **Tips for Managing Diabetes:**\n- Maintain a **healthy diet** (low sugar, high fiber).\n- Engage in **regular exercise** (walking, yoga, etc.).\n- Monitor blood sugar levels frequently.\n- Stay hydrated and avoid processed foods.\n- Follow medical advice and take prescribed medications.")

    except ValueError:
        st.error("⚠️ Please enter valid numerical values in all fields.")

# Footer with a positive message
st.markdown("""
    <div style="text-align:center; padding:15px; font-size:16px; color:#333; background-color:#f8f9fa; border-radius:8px;">
        🌟 *Empowering health through technology.* Stay informed, stay healthy! 💙
    </div>
""", unsafe_allow_html=True)
