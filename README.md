# AI Business Assistant Dashboard

## 📌 Project Description
AI Business Assistant is a Machine Learning based dashboard that helps businesses analyze sales trends, segment customers, and recommend products using historical transaction data.

---

## 🚀 Features

### 1️⃣ Sales Prediction
Predicts future sales based on selected month.

### 2️⃣ Customer Segmentation
Classifies customers into clusters based on spending behaviour.

### 3️⃣ Product Recommendation
Suggests similar products based on selected product.

---

## 🧠 Technologies Used

- Python
- Flask (Backend API)
- Streamlit (Frontend Dashboard)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- Matplotlib (Visualization)

---

## 📂 Project Structure

AI Business Assistant
│
├── backend
│ └── app.py
│
│── data
│ └──online_retail.csv
│
├── frontend
│ └── app.py
│
├── models
│ ├──  customer_cluster.pkl
│ ├── recommendation.pkl
│ └──  sales_model.pkl
│
├── notebooks
│ └── Business assistant.ipynb
│
└── README.md



---

## ⚙️ Installation & Setup

### Step 1: Clone Project

git clone <repository-link>

---

### Step 2: Install Dependencies

pip install flask streamlit pandas scikit-learn matplotlib flask-cors

---

### Step 3: Run Backend

cd backend
python app.py

---

### Step 4: Run Frontend

cd frontend
streamlit run app.py


---

## 📊 Models Used

- Linear Regression → Sales Prediction  
- KMeans Clustering → Customer Segmentation  
- Cosine Similarity → Product Recommendation  

---

## 📈 Dataset
Online retail transaction dataset used for training ML models.

---

## 👨‍💻 Author
Mohit Poonia

---

## 📅 Academic Project
Second Year Machine Learning Based Engineering Project


