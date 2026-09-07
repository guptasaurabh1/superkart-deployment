"""
SuperKart Sales Prediction - Streamlit Frontend
------------------------------------------------
A simple UI that lets a business user:
  1. Enter a single product/store record and get an instant sales
     revenue prediction (online inference).
  2. Upload a CSV of many records and get a table of predictions
     back (batch inference).

The frontend never loads the model itself - it always calls the Flask
backend's REST API. When both containers run on the same Docker
network in the Codespace, the backend is reachable at the network
alias set by BACKEND_URL (defaults to http://superkart-backend:7860).
"""

import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://superkart-backend:7860")

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="\U0001F6D2", layout="centered")

st.title("SuperKart Sales Revenue Predictor")
st.write(
    "Estimate `Product_Store_Sales_Total` for a product-store combination using "
    "the tuned Random Forest model trained on SuperKart's historical sales data."
)

with st.sidebar:
    st.header("Backend settings")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    st.caption(
        "This should point at the Flask API container, e.g. "
        "`http://superkart-backend:7860` inside the Docker network, "
        "or the public Codespace forwarded URL for testing from outside."
    )

tab_online, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------------------------------------------------------
# Online (single-record) inference
# ---------------------------------------------------------------
with tab_online:
    st.subheader("Enter product & store details")

    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66, step=0.1)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_area = st.number_input(
            "Product Allocated Area (ratio)", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f"
        )
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08, step=1.0)
        product_id_char = st.selectbox("Product Id Prefix", ["FD", "DR", "NC"])

    with col2:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_city = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox(
            "Store Type",
            ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
        )
        store_age = st.number_input("Store Age (Years)", min_value=0, value=16, step=1)
        product_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales", type="primary"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": product_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_category,
        }
        try:
            with st.spinner("Calling the model API..."):
                response = requests.post(f"{backend_url}/v1/predict", json=payload, timeout=15)
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Predicted Product_Store_Sales_Total: **Rs. {prediction:,.2f}**")
            else:
                st.error(f"API error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend API: {exc}")

# ---------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV for batch prediction")
    st.caption(
        "The file must contain these columns: Product_Weight, Product_Sugar_Content, "
        "Product_Allocated_Area, Product_MRP, Store_Size, Store_Location_City_Type, "
        "Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        preview_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(preview_df.head())
        uploaded_file.seek(0)

        if st.button("Run Batch Prediction"):
            try:
                with st.spinner("Sending file to the model API..."):
                    files = {"file": ("batch.csv", uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(f"{backend_url}/v1/predictbatch", files=files, timeout=30)
                if response.status_code == 200:
                    predictions = pd.Series(response.json(), name="Predicted_Product_Store_Sales_Total")
                    predictions.index = predictions.index.astype(int)
                    predictions = predictions.sort_index()
                    result_df = preview_df.copy()
                    result_df["Predicted_Product_Store_Sales_Total"] = predictions.values
                    st.success("Batch prediction complete!")
                    st.dataframe(result_df)
                    st.download_button(
                        "Download predictions as CSV",
                        result_df.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"API error ({response.status_code}): {response.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the backend API: {exc}")

st.divider()
st.caption("SuperKart Sales Forecasting | Model: Tuned Random Forest Regressor | Served via Flask + Streamlit on Docker")
