import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO

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
st.title('Model Deployment: Customer Segmentation')

# Sidebar Header for Inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    INCOME = st.sidebar.number_input("Insert Income ($):", min_value=0, value=50000)
    RECENCY = st.sidebar.number_input("Insert Recency (days):", min_value=0, value=30)
    WINES = st.sidebar.number_input("Wines Purchased:", min_value=0, value=10)
    FRUITS = st.sidebar.number_input("Fruits Purchased:", min_value=0, value=5)
    MEAT = st.sidebar.number_input("Meat Purchased:", min_value=0, value=8)
    FISH = st.sidebar.number_input("Fish Purchased:", min_value=0, value=4)
    SWEETS = st.sidebar.number_input("Sweets Purchased:", min_value=0, value=3)
    GOLD = st.sidebar.number_input("Gold Purchased:", min_value=0, value=2)
    NUM_DEALS_PURCHASES = st.sidebar.slider("Number of Deals Purchases:", 0, 20, 5)
    NUM_WEB_PURCHASES = st.sidebar.slider("Number of Web Purchases:", 0, 20, 5)
    NUM_CATALOG_PURCHASES = st.sidebar.slider("Number of Catalog Purchases:", 0, 20, 3)
    NUM_STORE_PURCHASES = st.sidebar.slider("Number of Store Purchases:", 0, 20, 8)
    NUM_WEB_VISITS = st.sidebar.slider("Number of Web Visits in Last Month:", 0, 30, 10)
    COMPLAIN = st.sidebar.selectbox("Complain (0-No, 1-Yes):", [0, 1])
    RESPONSE = st.sidebar.selectbox("Response (0-No, 1-Yes):", [0, 1])
    DURATION = st.sidebar.number_input("Duration of Engagement (months):", min_value=0, value=12)
    AGE = st.sidebar.number_input("Insert Age:", min_value=18, value=30)
    TOTAL_SPENT = st.sidebar.number_input("Insert Total Amount Spent ($):", min_value=0, value=1000)
    TOTAL_PURCHASES = st.sidebar.number_input("Total Purchases:", min_value=0, value=20)
    EDUCATION_LEVEL = st.sidebar.selectbox("Education Level (1-5):", [1, 2, 3, 4, 5])
    LIVING_STATUS = st.sidebar.selectbox("Living Status (1-Alone, 2-Partner):", [1, 2])
    CHILDREN = st.sidebar.number_input("Number of Children:", min_value=0, value=0)
    FAMILY_SIZE = st.sidebar.number_input("Family Size:", min_value=1, value=2)
    IS_PARENT = st.sidebar.selectbox("Is Parent (0-No, 1-Yes):", [0, 1])
    TOTAL_CAMPAIGN_RESPONSE = st.sidebar.number_input("Total Campaign Response:", min_value=0, value=1)
    
    return pd.DataFrame([{  
        'Income': INCOME, 'Recency': RECENCY, 'Wines': WINES, 'Fruits': FRUITS, 'Meat': MEAT, 'Fish': FISH, 'Sweets': SWEETS,
        'Gold': GOLD, 'NumDealsPurchases': NUM_DEALS_PURCHASES, 'NumWebPurchases': NUM_WEB_PURCHASES, 'NumCatalogPurchases': NUM_CATALOG_PURCHASES,
        'NumStorePurchases': NUM_STORE_PURCHASES, 'NumWebVisitsMonth': NUM_WEB_VISITS, 'Complain': COMPLAIN, 'Response': RESPONSE,
        'Duration': DURATION, 'Age': AGE, 'TotalSpent': TOTAL_SPENT, 'TotalPurchases': TOTAL_PURCHASES, 'EducationLevel': EDUCATION_LEVEL,
        'LivingStatus': LIVING_STATUS, 'Children': CHILDREN, 'FamilySize': FAMILY_SIZE, 'IsParent': IS_PARENT, 'TotalCampaignResponse': TOTAL_CAMPAIGN_RESPONSE
    }])

# Collect Inputs
df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

# Load models
scaler = load_pickle_from_github(scaler_url)
pca = load_pickle_from_github(pca_url)
kmeans = load_pickle_from_github(kmeans_url)

# Ensure input features match the trained scaler
expected_features = scaler.feature_names_in_
df = df[expected_features]

# Scale input features
df_scaled = scaler.transform(df)

# Apply PCA transformation
df_pca = pca.transform(df_scaled)

# Make Cluster Predictions
prediction = kmeans.predict(df_pca)

# Display Results
st.subheader("Cluster Prediction:")
st.success(f"The customer is segmented into **Cluster {prediction[0]}** based on input data.")

# Read dataset from GitHub
try:
    response = requests.get(df_url)
    if response.status_code == 200:
        data = pd.read_excel(BytesIO(response.content), engine="openpyxl")
        
        # Display Dataset Summary
        st.subheader("Dataset Overview")
        st.write(data.head())
        st.write("Shape of dataset:", data.shape)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(14, 8))  
            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title("Feature Correlation Heatmap", fontsize=14)  
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for correlation heatmap.")

        # Distribution of Numerical Features
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select a feature to view its distribution:", numeric_data.columns)
        fig, ax = plt.subplots()
        sns.histplot(data[selected_feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_feature}")
        st.pyplot(fig)

        # Pairplot of Features
        st.subheader("Pairplot of Selected Features")
        selected_features = st.multiselect("Select features for pairplot:", numeric_data.columns, default=numeric_data.columns[:3])
        if len(selected_features) >= 2:
            fig = sns.pairplot(data[selected_features])
            st.pyplot(fig)
        else:
            st.warning("Please select at least two features for the pairplot.")

        # Clustering Visualization
        st.subheader("Clustering Visualization")
        data_scaled = scaler.transform(data[expected_features])
        data_pca = pca.transform(data_scaled)
        data['Cluster'] = kmeans.predict(data_pca)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis')
        ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.set_title("Customer Segmentation (PCA Reduced)")
        st.pyplot(fig)

    else:
        st.error("Failed to load dataset.")

except Exception as e:
    st.error(f"Error reading dataset: {e}")
