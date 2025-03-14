import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO

# Set Streamlit page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
        .stTitle { font-size: 32px; color: #4A90E2; text-align: center; font-weight: bold; }
        .stSidebar { background-color: #1e3d59 !important; color: white !important; }
        .stButton>button { background-color: #4A90E2 !important; color: white !important; font-size: 18px !important; border-radius: 8px !important; }
        .stDataFrame { border-radius: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); }
    </style>
    """,
    unsafe_allow_html=True
)

# Load pickle file from GitHub
def load_pickle_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"❌ Failed to load: {url}")
        return None

# GitHub URLs
df_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/data/df_clean.xlsx"
kmeans_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/kmeans_model_2.pkl"
pca_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/pca_transformer.pkl"
scaler_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/scaler.pkl"

# App title
st.markdown('<h1 class="stTitle">🔹 Customer Segmentation Model Deployment</h1>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header('⚙️ User Input Parameters')

# Function to get user inputs
def user_input_features():
    features = {
        'Income': st.sidebar.number_input("💰 Income ($):", min_value=0, value=50000),
        'Recency': st.sidebar.number_input("⏳ Recency (days):", min_value=0, value=30),
        'Wines': st.sidebar.number_input("🍷 Wines Purchased:", min_value=0, value=10),
        'Fruits': st.sidebar.number_input("🍎 Fruits Purchased:", min_value=0, value=5),
        'Meat': st.sidebar.number_input("🥩 Meat Purchased:", min_value=0, value=8),
        'Fish': st.sidebar.number_input("🐟 Fish Purchased:", min_value=0, value=4),
        'Sweets': st.sidebar.number_input("🍬 Sweets Purchased:", min_value=0, value=3),
        'Gold': st.sidebar.number_input("🛍 Gold Purchased:", min_value=0, value=2),
        'NumDealsPurchases': st.sidebar.slider("🎯 Deals Purchases:", 0, 20, 5),
        'NumWebPurchases': st.sidebar.slider("🛒 Web Purchases:", 0, 20, 5),
        'NumCatalogPurchases': st.sidebar.slider("📖 Catalog Purchases:", 0, 20, 3),
        'NumStorePurchases': st.sidebar.slider("🏬 Store Purchases:", 0, 20, 8),
        'NumWebVisitsMonth': st.sidebar.slider("🌐 Web Visits Last Month:", 0, 30, 10),
        'Complain': st.sidebar.selectbox("❗ Complain (0-No, 1-Yes):", [0, 1]),
        'Response': st.sidebar.selectbox("📩 Response to Campaign (0-No, 1-Yes):", [0, 1]),
        'Duration': st.sidebar.number_input("📆 Engagement Duration (months):", min_value=0, value=12),
        'Age': st.sidebar.number_input("🎂 Age:", min_value=18, value=30),
        'TotalSpent': st.sidebar.number_input("💳 Total Amount Spent ($):", min_value=0, value=1000),
        'TotalPurchases': st.sidebar.number_input("🛍 Total Purchases:", min_value=0, value=20)
    }
    return pd.DataFrame([features])

df = user_input_features()
st.subheader('📌 User Input:')
st.write(df)

# Load models
scaler = load_pickle_from_github(scaler_url)
pca = load_pickle_from_github(pca_url)
kmeans = load_pickle_from_github(kmeans_url)

if scaler and pca and kmeans:
    # Ensure required columns match
    required_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df.columns
    df = df[required_features]

    # Transform input
    df_scaled = scaler.transform(df)
    df_pca = pca.transform(df_scaled)
    prediction = kmeans.predict(df_pca)

    st.subheader("🎯 Cluster Prediction:")
    st.success(f"💡 The customer belongs to **Cluster {prediction[0]}**.")

    # Load dataset
    response = requests.get(df_url)
    if response.status_code == 200:
        data = pd.read_excel(BytesIO(response.content), engine="openpyxl")

        st.subheader("📊 Dataset Overview")
        st.write(data.head())
        st.write(f"**Shape:** {data.shape}")

        # Correlation Heatmap
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(18, 12))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        # Feature Distributions
        st.subheader("📈 Feature Distributions")
        selected_feature = st.selectbox("Select feature:", data.select_dtypes(['float64', 'int64']).columns)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data[selected_feature], kde=True, color='blue', ax=ax)
        st.pyplot(fig)

        # Clusters Visualization
        st.subheader("🌍 Customer Segmentation")
        data_scaled = scaler.transform(data[required_features])
        data_pca = pca.transform(data_scaled)
        data['Cluster'] = kmeans.predict(data_pca)
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)
        ax.set_title("Clusters Visualization")
        st.pyplot(fig)

    else:
        st.error("❌ Failed to load dataset.")
