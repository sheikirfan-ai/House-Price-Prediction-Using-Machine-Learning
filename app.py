import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")

model = joblib.load("house_price_model.pkl")

col1, col2 = st.columns(2)
with col1:
    MedInc = st.number_input("Median income (tens of thousands USD)", value=5.0)
    HouseAge = st.number_input("House age (years)", value=20)
    AveRooms = st.number_input("Average rooms per household", value=6)
with col2:
    AveBedrms = st.number_input("Average bedrooms per household", value=1.0)
    Population = st.number_input("Block population", value=1000.0)
    USD_TO_INR = st.number_input("Exchange rate (1 USD in INR)", value=96.5)

if st.button("Predict price"):
    new_data = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
    }])

    raw = model.predict(new_data)[0]
    price_usd = raw * 100000          
    price_inr = price_usd * USD_TO_INR

    if price_usd <= 0:
        st.error("These values are outside what the model can estimate. Try typical values.")
    else:
        st.success(f"Predicted house value: ₹{price_inr:,.2f}")
        st.write(f"(${price_usd:,.2f} at ₹{USD_TO_INR} per USD)")