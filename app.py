import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stTitle {
            font-size: 32px !important;
            color: #4A90E2;
            text-align: center;
            font-weight: bold;
        }
        .stSidebar {
            background-color: #1e3d59 !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #4A90E2 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 8px !important;
        }
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        }
        .stMarkdown {
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load pickle files from GitHub
def load_pickle_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"❌ Failed to load: {url}")
        return None

# GitHub URLs for data and models
df_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/data/df_clean.xlsx"
kmeans_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/kmeans_model_2.pkl"
pca_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/pca_transformer.pkl"
scaler_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/scaler.pkl"

# Title of the App
st.markdown('<h1 class="stTitle">🔹 Customer Segmentation Model Deployment</h1>', unsafe_allow_html=True)

# Sidebar Header for Inputs
st.sidebar.header('⚙️ User Input Parameters')

# User Input Function
def user_input_features():
    with st.sidebar:
        INCOME = st.number_input("💰 Income ($):", min_value=0, value=50000)
        RECENCY = st.number_input("⏳ Recency (days):", min_value=0, value=30)
        WINES = st.number_input("🍷 Wines Purchased:", min_value=0, value=10)
        FRUITS = st.number_input("🍎 Fruits Purchased:", min_value=0, value=5)
        MEAT = st.number_input("🥩 Meat Purchased:", min_value=0, value=8)
        FISH = st.number_input("🐟 Fish Purchased:", min_value=0, value=4)
        SWEETS = st.number_input("🍬 Sweets Purchased:", min_value=0, value=3)
        GOLD = st.number_input("🛍 Gold Purchased:", min_value=0, value=2)
        NUM_DEALS_PURCHASES = st.slider("🎯 Deals Purchases:", 0, 20, 5)
        NUM_WEB_PURCHASES = st.slider("🛒 Web Purchases:", 0, 20, 5)
        NUM_CATALOG_PURCHASES = st.slider("📖 Catalog Purchases:", 0, 20, 3)
        NUM_STORE_PURCHASES = st.slider("🏬 Store Purchases:", 0, 20, 8)
        NUM_WEB_VISITS = st.slider("🌐 Web Visits Last Month:", 0, 30, 10)
        COMPLAIN = st.selectbox("❗ Complain (0-No, 1-Yes):", [0, 1])
        RESPONSE = st.selectbox("📩 Response to Campaign (0-No, 1-Yes):", [0, 1])
        DURATION = st.number_input("📆 Engagement Duration (months):", min_value=0, value=12)
        AGE = st.number_input("🎂 Age:", min_value=18, value=30)
        TOTAL_SPENT = st.number_input("💳 Total Amount Spent ($):", min_value=0, value=1000)
        TOTAL_PURCHASES = st.number_input("🛍 Total Purchases:", min_value=0, value=20)

    return pd.DataFrame([{  
        'Income': INCOME, 'Recency': RECENCY, 'Wines': WINES, 'Fruits': FRUITS, 'Meat': MEAT, 'Fish': FISH,
        'Sweets': SWEETS, 'Gold': GOLD, 'NumDealsPurchases': NUM_DEALS_PURCHASES, 'NumWebPurchases': NUM_WEB_PURCHASES, 
        'NumCatalogPurchases': NUM_CATALOG_PURCHASES, 'NumStorePurchases': NUM_STORE_PURCHASES, 
        'NumWebVisitsMonth': NUM_WEB_VISITS, 'Complain': COMPLAIN, 'Response': RESPONSE, 'Duration': DURATION, 
        'Age': AGE, 'TotalSpent': TOTAL_SPENT, 'TotalPurchases': TOTAL_PURCHASES
    }])

df = user_input_features()
st.subheader('📌 User Input:')
st.write(df)

# Load models
scaler = load_pickle_from_github(scaler_url)
pca = load_pickle_from_github(pca_url)
kmeans = load_pickle_from_github(kmeans_url)

if scaler and pca and kmeans:
    # Ensure feature names match
    df.columns = scaler.feature_names_in_
    
    # Convert data to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

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
        data_scaled = scaler.transform(data)
        data_pca = pca.transform(data_scaled)
        data['Cluster'] = kmeans.predict(data_pca)
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)
        ax.set_title("Clusters Visualization")
        st.pyplot(fig)

    else:
        st.error("❌ Failed to load dataset.")

