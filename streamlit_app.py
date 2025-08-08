import streamlit as st
import pandas as pd
import joblib

st.title("Iris Classifier")
st.write("This is a simple Iris Classifier app")

# == Load Model
model = joblib.load("model_joblib")

# == Inference Function
def get_prediction(data: pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# User input
left, right = st.columns(2, gap='medium', border=True)

# -- Sepal input
left.write('Sepal')
sepal_length = left.slider('Sepal Length', min_value=1.0, max_value=10.0, value=5.4, step=0.1)
sepal_width = left.slider('Sepal Width', min_value=1.0, max_value=10.0, value=5.4, step=0.1)

# -- Petal input
right.write('Petal')
petal_length = right.slider('Petal Length', min_value=1.0, max_value=10.0, value=5.4, step=0.1)
petal_width = right.slider('Petal Width', min_value=1.0, max_value=10.0, value=5.4, step=0.1)

# Buat dataframe dari input
data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Show input value
st.dataframe(data, use_container_width=True)

# Prediction Button
button = st.button("Predict", use_container_width=True)

if button:
    st.write("Prediksi Berhasil !")
    pred, pred_proba = get_prediction(data)

    label_map = {0: "Setosa", 1: "Versicolor",2: "Virginica"}
    
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    output = f"Iris Anda diklasifikasikan sebagai {label_proba:.0%} {label_pred}"
    st.write(output)