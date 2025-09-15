import streamlit as st
import numpy as np
import pickle
import joblib
pca = joblib.load("pca_model.pkl")

# ✅ Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

# 🧾 Get all 13 features from user
features = []
feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

st.title("🍷 Wine PCA + Model Predictor")

for name in feature_names:
    value = st.number_input(f"Enter value for {name}", format="%.2f")
    features.append(value)

if st.button("Predict"):
    # Convert to numpy array
    data = np.array([features])

    # Apply PCA
    data_pca = pca.transform(data)

    # 🔮 Make prediction using the ML model
    prediction = model.predict(data_pca)

    st.write("PCA-transformed data:")
    st.write(data_pca)

    st.success(f"Predicted Output: {prediction[0]}")
