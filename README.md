
# 🛍️ Customer Segmentation using Clustering  

## 📌 Overview  
Understanding customer behavior is key to optimizing marketing strategies and improving business performance. This project applies clustering techniques to segment customers based on their demographics, spending patterns, and interactions with promotions.  

## 🚀 Objective  
The goal is to identify distinct customer segments by analyzing their purchasing behavior and engagement with the company's marketing efforts. These insights can help businesses tailor personalized offers and enhance customer satisfaction.  

---

## 📊 Dataset Description  

The dataset consists of various features categorized into **People**, **Products**, **Promotions**, and **Place**, capturing essential details about customer interactions and spending habits.

### 🧑‍💼 Customer Demographics  
| Feature         | Description  |
|----------------|-------------|
| `ID`           | Unique customer identifier  |
| `Year_Birth`   | Customer's birth year  |
| `Education`    | Education level (e.g., PhD, Master, Graduate)  |
| `Marital_Status` | Marital status (e.g., Single, Married, Divorced)  |
| `Income`       | Yearly household income  |
| `Kidhome`      | Number of children in the household  |
| `Teenhome`     | Number of teenagers in the household  |
| `Dt_Customer`  | Date of enrollment with the company  |
| `Recency`      | Number of days since the last purchase  |
| `Complain`     | 1 if a complaint was filed in the last 2 years, else 0  |

### 🍷 Customer Spending Behavior  
| Feature           | Description  |
|------------------|-------------|
| `MntWines`       | Amount spent on wine in the last 2 years  |
| `MntFruits`      | Amount spent on fruits in the last 2 years  |
| `MntMeatProducts` | Amount spent on meat in the last 2 years  |
| `MntFishProducts` | Amount spent on fish in the last 2 years  |
| `MntSweetProducts` | Amount spent on sweets in the last 2 years  |
| `MntGoldProds`   | Amount spent on gold in the last 2 years  |

### 🎯 Marketing & Promotions  
| Feature            | Description  |
|-------------------|-------------|
| `NumDealsPurchases` | Purchases made using discounts  |
| `AcceptedCmp1-5`   | Whether the customer accepted past campaign offers (1 = Yes, 0 = No)  |
| `Response`        | Accepted the latest campaign (1 = Yes, 0 = No)  |

### 🏬 Purchase Channels  
| Feature               | Description  |
|----------------------|-------------|
| `NumWebPurchases`    | Purchases made through the website  |
| `NumCatalogPurchases` | Purchases made via catalog orders  |
| `NumStorePurchases`  | Purchases made in physical stores  |
| `NumWebVisitsMonth`  | Website visits in the last month  |

---

## 🏆 Project Workflow  

1️⃣ **Data Preprocessing**  
   - Handle missing values & outliers  
   - Standardize numerical features  
   - Encode categorical variables  

2️⃣ **Exploratory Data Analysis (EDA)**  
   - Visualizing spending patterns  
   - Analyzing customer demographics  
   - Correlation analysis  

3️⃣ **Clustering (Segmentation)**  
   - K-Means, Hierarchical Clustering, and DBSCAN  
   - Evaluating clusters using Silhouette Score & Elbow Method  
   - Interpreting customer segments  

4️⃣ **Insights & Business Recommendations**  
   - Identifying high-value customer groups  
   - Strategies for targeted marketing campaigns  
   - Improving customer engagement based on segments  

---

## 🛠️ Technologies Used  
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- **Machine Learning** (Clustering techniques: K-Means, Hierarchical, DBSCAN)  
- **Data Visualization** (Tableau, Matplotlib, Seaborn)  

---

## 📈 Key Insights  
🔹 High-spending customers prefer **wine & meat products**, whereas others focus on discounted purchases.  
🔹 Customers engaging with **multiple marketing campaigns** show increased spending behavior.  
🔹 Online shoppers are more likely to respond to promotional offers compared to in-store buyers.  

---

## 📂 Project Structure  
```
📂 Customer-Segmentation
│── 📁 data               # Raw & processed datasets
│── 📁 notebooks          # Jupyter notebooks for EDA & modeling
│── 📁 Pickle             # Saved Pickle files
│── 📄 README.md          # Project overview & details
│── 📄 requirements.txt   # Dependencies
│── 📄 app.py             # Clustering pipeline(Deployment)
```

---

## 📌 How to Run  
1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/Customer-Segmentation.git
cd Customer-Segmentation
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
3️⃣ Run the clustering script  
```bash
python customer_segmentation.py
```
4️⃣ Analyze insights & visualizations in Jupyter Notebook  
```bash
jupyter notebook
```

---

## 📬 Connect with Me  
🔗 **LinkedIn:** [Sunil Karrenolla](https://www.linkedin.com/in/sunil-karrenolla/)  
💻 **GitHub:** [sunilk872](https://github.com/sunilk872/)  

