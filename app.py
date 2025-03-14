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

# Apply custom CSS for styling
st.markdown(
    """
    <style>
        .stTitle {
            font-size: 32px !important;
            color: #4A90E2;
            text-align: center;
        }
        .stSidebar {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4A90E2 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 8px !important;
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
        st.error(f"Failed to load {url}")
        return None

# GitHub URLs for models
scaler_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/scaler.pkl"
pca_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/pca_transformer.pkl"
kmeans_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/kmeans_model_2.pkl"
df_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/data/customer_data.xlsx"

# Title of the App
st.markdown('<h1 class="stTitle">Customer Segmentation Model Deployment</h1>', unsafe_allow_html=True)

# Sidebar Header for Inputs
st.sidebar.header('🔹 User Input Parameters')

def user_input_features():
    with st.sidebar:
        st.markdown("<h4 style='text-align: center; color: #4A90E2;'>📋 Fill Customer Details</h4>", unsafe_allow_html=True)

        INCOME = st.number_input("💰 Income ($):", min_value=0, value=50000)
        RECENCY = st.number_input("📆 Recency (days):", min_value=0, value=30)
        WINES = st.number_input("🍷 Wines Purchased:", min_value=0, value=10)
        FRUITS = st.number_input("🍎 Fruits Purchased:", min_value=0, value=5)
        MEAT = st.number_input("🥩 Meat Purchased:", min_value=0, value=8)
        FISH = st.number_input("🐟 Fish Purchased:", min_value=0, value=4)
        SWEETS = st.number_input("🍬 Sweets Purchased:", min_value=0, value=3)
        GOLD = st.number_input("🏅 Gold Purchased:", min_value=0, value=2)
        NUM_DEALS_PURCHASES = st.slider("📢 Deals Purchases:", 0, 20, 5)
        NUM_WEB_PURCHASES = st.slider("🖥️ Web Purchases:", 0, 20, 5)
        NUM_CATALOG_PURCHASES = st.slider("📖 Catalog Purchases:", 0, 20, 3)
        NUM_STORE_PURCHASES = st.slider("🏬 Store Purchases:", 0, 20, 8)
        NUM_WEB_VISITS = st.slider("🌐 Web Visits in Last Month:", 0, 30, 10)
        COMPLAIN = st.selectbox("⚠️ Complain (0-No, 1-Yes):", [0, 1])
        RESPONSE = st.selectbox("📩 Response (0-No, 1-Yes):", [0, 1])
        AGE = st.number_input("🎂 Age:", min_value=18, value=30)
        TOTAL_SPENT = st.number_input("💵 Total Amount Spent ($):", min_value=0, value=1000)
        FAMILY_SIZE = st.number_input("👨‍👩‍👧‍👦 Family Size:", min_value=1, value=2)
        TOTAL_CAMPAIGN_RESPONSE = st.number_input("📢 Total Campaign Response:", min_value=0, value=1)
    
    return pd.DataFrame([{  
        'Income': INCOME, 'Recency': RECENCY, 'Wines': WINES, 'Fruits': FRUITS, 'Meat': MEAT, 'Fish': FISH, 'Sweets': SWEETS,
        'Gold': GOLD, 'NumDealsPurchases': NUM_DEALS_PURCHASES, 'NumWebPurchases': NUM_WEB_PURCHASES, 
        'NumCatalogPurchases': NUM_CATALOG_PURCHASES, 'NumStorePurchases': NUM_STORE_PURCHASES, 
        'NumWebVisitsMonth': NUM_WEB_VISITS, 'Complain': COMPLAIN, 'Response': RESPONSE,
        'Age': AGE, 'TotalSpent': TOTAL_SPENT, 'FamilySize': FAMILY_SIZE, 'TotalCampaignResponse': TOTAL_CAMPAIGN_RESPONSE
    }])

df = user_input_features()
st.subheader('📌 User Input:')
st.write(df)

# Load models
scaler = load_pickle_from_github(scaler_url)
pca = load_pickle_from_github(pca_url)
kmeans = load_pickle_from_github(kmeans_url)

# Ensure feature names match before transformation
expected_features = scaler.feature_names_in_  # Get expected column names from the trained scaler
df = df[expected_features]  # Reorder and select only expected columns

# Apply transformations
df_scaled = scaler.transform(df)
df_pca = pca.transform(df_scaled)
prediction = kmeans.predict(df_pca)

st.subheader("🎯 Cluster Prediction:")
st.success(f"💡 The customer belongs to *Cluster {prediction[0]}*.")


# Load dataset from GitHub
response = requests.get(df_url)
if response.status_code == 200:
    data = pd.read_excel(BytesIO(response.content), engine="openpyxl")
    
    st.subheader("📊 Dataset Overview")
    st.write(data.head())
    st.write(f"*Shape:* {data.shape}")
    
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
    
    st.subheader("📈 Feature Distributions")
    selected_feature = st.selectbox("Select feature:", data.select_dtypes(['float64', 'int64']).columns)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(data[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("🌍 Customer Segmentation")
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    data['Cluster'] = kmeans.predict(data_pca)
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis')
    ax.set_title("Clusters Visualization")
    st.pyplot(fig)
else:
    st.error("Failed to load dataset.")
