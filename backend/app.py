from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# ===== LOAD MODELS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models")

# Load small models
sales_model = pickle.load(open(os.path.join(MODEL_PATH, "sales_model.pkl"), "rb"))
cluster_model = pickle.load(open(os.path.join(MODEL_PATH, "customer_cluster.pkl"), "rb"))

# Load recommendation model from Hugging Face
rec_path = hf_hub_download(
    repo_id="mohitpoonia21/business-assistant-models",
    filename="recommendation.pkl",
    repo_type="dataset"
)
recommend_model = pickle.load(open(rec_path, "rb"))

# ===== HOME ROUTE =====
@app.route("/")
def home():
    return jsonify({"message": "AI Business Assistant Backend Running 🚀"})


# ===== SALES PREDICTION =====
@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    try:
        data = request.get_json()
        month = int(data['month'])

        prediction = sales_model.predict([[month]])

        return jsonify({
            "predicted_sales": float(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ===== CUSTOMER SEGMENTATION =====
@app.route('/customer_segment', methods=['POST'])
def customer_segment():
    try:
        data = request.get_json()

        recency = float(data['recency'])
        frequency = float(data['frequency'])
        monetary = float(data['monetary'])

        cluster = cluster_model.predict([[recency, frequency, monetary]])

        return jsonify({
            "cluster": int(cluster[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ===== PRODUCT RECOMMENDATION =====
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        product = data['product']

        if product not in recommend_model.index:
            return jsonify({"error": "Product not found"})

        similar_products = recommend_model[product].sort_values(ascending=False)[1:6]

        return jsonify({
            "recommendations": similar_products.index.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ===== RUN SERVER =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
