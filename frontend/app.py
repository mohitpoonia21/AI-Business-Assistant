import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="AI Business Assistant", layout="wide")

BACKEND_URL = https://ai-business-assistant-1pd5.onrender.com

st.title("AI Business Assistant Dashboard")

# =========================================================
# SALES PREDICTION
# =========================================================
st.header("Sales Prediction")

month = st.slider("Select Month", 1, 12, 3)

if st.button("Predict Sales"):

    res = requests.get(f"{BACKEND_URL}/predict_sales?month={month}")
    data = res.json()

    if "error" in data:
        st.error(data["error"])
    else:
        sales = data["Predicted Sales"]

        st.success(f"Predicted Sales: {sales}")

        # LINE GRAPH
        months = list(range(1, 13))
        predicted = []

        for m in months:
            r = requests.get(f"{BACKEND_URL}/predict_sales?month={m}")
            predicted.append(r.json()["Predicted Sales"])

        fig, ax = plt.subplots()
        ax.plot(months, predicted, marker="o")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        ax.set_title("Sales Prediction Trend")

        st.pyplot(fig)

st.divider()

# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================
st.header("Customer Segmentation")

spending = st.number_input("Total Spending", value=30000.0)

if st.button("Predict Customer Cluster"):

    res = requests.get(f"{BACKEND_URL}/predict_cluster?spending={spending}")
    data = res.json()

    if "error" in data:
        st.error(data["error"])
    else:
        cluster = data["Cluster"]
        st.success(f"Customer belongs to Cluster {cluster}")

        # SCATTER GRAPH
        x = np.random.randint(1000, 60000, 50)
        y = np.random.randint(1, 5, 50)

        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.scatter(spending, cluster, marker="X", s=200)

        ax.set_xlabel("Spending")
        ax.set_ylabel("Cluster")
        ax.set_title("Customer Segmentation")

        st.pyplot(fig)

st.divider()

# =========================================================
# PRODUCT RECOMMENDATION
# =========================================================
st.header("Product Recommendation")

product = st.text_input("Enter Product Name")

if st.button("Recommend Products"):

    res = requests.get(f"{BACKEND_URL}/recommend_products?product={product}")
    data = res.json()

    if "error" in data:
        st.error(data["error"])
    else:
        recs = data["recommendations"]

        st.subheader("Recommended Products")
        for r in recs:
            st.write(r)

        # BAR GRAPH
        fig, ax = plt.subplots()
        ax.barh(recs, list(range(len(recs))))

        ax.set_title("Recommended Products")

        st.pyplot(fig)
