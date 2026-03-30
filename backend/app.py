from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# ===== LOAD MODELS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models")

sales_model = pickle.load(open(os.path.join(MODEL_PATH, "sales_model.pkl"), "rb"))
cluster_model = pickle.load(open(os.path.join(MODEL_PATH, "customer_cluster.pkl"), "rb"))
recommend_model = pickle.load(open(os.path.join(MODEL_PATH, "recommendation.pkl"), "rb"))


# ===== SALES PREDICTION =====
@app.route("/predict_sales")
def predict_sales():
    try:
        month = int(request.args.get("month"))

        prediction = sales_model.predict([[month]])

        return jsonify({
            "Predicted Sales": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ===== CUSTOMER SEGMENTATION =====
@app.route("/predict_cluster")
def predict_cluster():
    try:
        spending = float(request.args.get("spending"))

        # IMPORTANT: KMeans expects 2 features
        cluster = cluster_model.predict([[spending, 1]])

        return jsonify({
            "Cluster": int(cluster[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ===== PRODUCT RECOMMENDATION =====
@app.route("/recommend_products")
def recommend_products():
    try:
        product = request.args.get("product")

        if product not in recommend_model.index:
            return jsonify({"error": "Product not found"})

        similar_products = recommend_model[product].sort_values(ascending=False)[1:6]

        return jsonify({
            "recommendations": similar_products.index.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
