import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = fetch_california_housing(as_frame=True)
Xm = df.data
ym = df.target

Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size=0.2, random_state=0)

LRm = LinearRegression()
LRm.fit(Xm_train, ym_train)

st.title("Predicción de precios de casas en California")
st.header("Modelo: Regresión Lineal")

# Imagen
st.image("house.jpeg", caption="Casa en California")

st.write("Este modelo predice el precio medio de casas en California, USA")

MedInc = st.sidebar.slider("Ingreso (MedInc)", float(Xm.MedInc.min()), float(Xm.MedInc.max()))
HouseAge = st.sidebar.slider("Edad de la casa (HouseAge)", float(Xm.HouseAge.min()), float(Xm.HouseAge.max()))
AveRooms = st.sidebar.slider("Número de habitaciones (AveRooms)", float(Xm.AveRooms.min()), float(Xm.AveRooms.max()))
Population = st.sidebar.slider("Población (Population)", float(Xm.Population.min()), float(Xm.Population.max()))

input = pd.DataFrame({
    'MedInc': [MedInc],
    'HouseAge': [HouseAge],
    'AveRooms': [AveRooms],
    'AveBedrms': [0],
    'Population': [Population],
    'AveOccup': [0],
    'Latitude': [0],
    'Longitude': [0]
})


pred = LRm.predict(input)[0]

if st.button("Predecir precio"):
    st.success(f"${pred*100000:,.2f} es el precio calculado de la casa")

st.subheader("Precios reales vs Precios predichos")
y_pred = LRm.predict(Xm_test)

fig, ax = plt.subplots()
ax.scatter(ym_test, y_pred, alpha=0.5)
ax.set_xlabel("Precios reales")
ax.set_ylabel("Precios predichos")
ax.set_title("Evaluación del modelo")
st.pyplot(fig)

