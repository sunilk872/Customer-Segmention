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

# Custom CSS for improved aesthetics
st.markdown(
    """
    <style>
        body {
            background-color: #eef2f3;
        }
        .stTitle {
            font-size: 36px !important;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
        }
        .stSidebar {
            background-color: #FFD700 !important;
        }
        .stButton>button {
            background-color: #32CD32 !important;
            color: white !important;
            font-size: 20px !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }
        .metric-box {
            background-color: #ADD8E6;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
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
        st.error(f"Failed to load {url}")
        return None

# GitHub URLs for data and models
df_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/data/df_clean.xlsx"
kmeans_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/kmeans_model_2.pkl"
pca_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/pca_transformer.pkl"
scaler_url = "https://github.com/sunilk872/Customer-Segmention/raw/main/pickle/scaler.pkl"

# Title of the App
st.markdown('<h1 class="stTitle">🚀 Customer Segmentation Model</h1>', unsafe_allow_html=True)

# Sidebar Header for Inputs
st.sidebar.header('🔹 Customize User Input')

def user_input_features():
    with st.sidebar:
        INCOME = st.number_input("💰 Income ($):", min_value=0, value=50000)
        RECENCY = st.number_input("📅 Recency (days):", min_value=0, value=30)
        WINES = st.number_input("🍷 Wines Purchased:", min_value=0, value=10)
        FRUITS = st.number_input("🍎 Fruits Purchased:", min_value=0, value=5)
        MEAT = st.number_input("🥩 Meat Purchased:", min_value=0, value=8)
        FISH = st.number_input("🐟 Fish Purchased:", min_value=0, value=4)
        SWEETS = st.number_input("🍬 Sweets Purchased:", min_value=0, value=3)
        GOLD = st.number_input("💎 Gold Purchased:", min_value=0, value=2)
        NUM_DEALS_PURCHASES = st.slider("🔥 Deals Purchases:", 0, 20, 5)
        NUM_WEB_PURCHASES = st.slider("🛒 Web Purchases:", 0, 20, 5)
        NUM_CATALOG_PURCHASES = st.slider("📖 Catalog Purchases:", 0, 20, 3)
        NUM_STORE_PURCHASES = st.slider("🏪 Store Purchases:", 0, 20, 8)
        NUM_WEB_VISITS = st.slider("🌍 Web Visits in Last Month:", 0, 30, 10)
        COMPLAIN = st.selectbox("⚠️ Complain (0-No, 1-Yes):", [0, 1])
        RESPONSE = st.selectbox("🎯 Response (0-No, 1-Yes):", [0, 1])
        AGE = st.number_input("👤 Age:", min_value=18, value=30)
        TOTAL_SPENT = st.number_input("💵 Total Amount Spent ($):", min_value=0, value=1000)
        TOTAL_PURCHASES = st.number_input("📦 Total Purchases:", min_value=0, value=20)
        EDUCATION_LEVEL = st.selectbox("🎓 Education Level (1-5):", [1, 2, 3, 4, 5])
        FAMILY_SIZE = st.number_input("👨‍👩‍👧‍👦 Family Size:", min_value=1, value=2)
    
    return pd.DataFrame([{  
        'Income': INCOME, 'Recency': RECENCY, 'Wines': WINES, 'Fruits': FRUITS, 'Meat': MEAT, 'Fish': FISH, 'Sweets': SWEETS,
        'Gold': GOLD, 'NumDealsPurchases': NUM_DEALS_PURCHASES, 'NumWebPurchases': NUM_WEB_PURCHASES, 
        'NumCatalogPurchases': NUM_CATALOG_PURCHASES, 'NumStorePurchases': NUM_STORE_PURCHASES, 
        'NumWebVisitsMonth': NUM_WEB_VISITS, 'Complain': COMPLAIN, 'Response': RESPONSE, 
        'Age': AGE, 'TotalSpent': TOTAL_SPENT, 'TotalPurchases': TOTAL_PURCHASES, 'EducationLevel': EDUCATION_LEVEL,
        'FamilySize': FAMILY_SIZE
    }])

df = user_input_features()
st.subheader('📌 User Input:')
st.write(df)

# Load models
scaler = load_pickle_from_github(scaler_url)
pca = load_pickle_from_github(pca_url)
kmeans = load_pickle_from_github(kmeans_url)

df_scaled = scaler.transform(df)
df_pca = pca.transform(df_scaled)
prediction = kmeans.predict(df_pca)

st.subheader("🎯 Cluster Prediction:")
st.markdown(f"<div class='metric-box'>💡 The customer belongs to <b>Cluster {prediction[0]}</b></div>", unsafe_allow_html=True)

# Load dataset
response = requests.get(df_url)
if response.status_code == 200:
    data = pd.read_excel(BytesIO(response.content), engine="openpyxl")
    st.subheader("📊 Dataset Overview")
    st.write(data.head())
    
    # Clusters Visualization
    st.subheader("🌍 Customer Segmentation")
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    data['Cluster'] = kmeans.predict(data_pca)
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='coolwarm')
    ax.set_title("Clusters Visualization")
    st.pyplot(fig)
else:
    st.error("Failed to load dataset.")
