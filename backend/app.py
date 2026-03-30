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

sales_model = pickle.load(open(os.path.join(MODEL_PATH, "sales_model.pkl"), "rb"))
cluster_model = pickle.load(open(os.path.join(MODEL_PATH, "customer_cluster.pkl"), "rb"))

rec_path = hf_hub_download(
    repo_id="mohitpoonia21/business-assistant-models",
    filename="recommendation.pkl",
    repo_type="dataset"
)
recommend_model = pickle.load(open(rec_path, "rb"))

recommend_model.index = recommend_model.index.str.lower()

# ===== SALES =====
@app.route("/predict_sales")
def predict_sales():
    try:
        month = int(request.args.get("month"))

        # generate trend (realistic visualization)
        months = list(range(1, month + 1))
        predictions = [float(sales_model.predict([[m]])[0]) for m in months]

        return jsonify({
            "current": predictions[-1],
            "months": months,
            "trend": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== SEGMENTATION =====
@app.route("/predict_cluster")
def predict_cluster():
    try:
        spending = float(request.args.get("spending"))

        cluster = cluster_model.predict([[spending]])

        centers = cluster_model.cluster_centers_.tolist()

        return jsonify({
            "cluster": int(cluster[0]),
            "input": spending,
            "centers": centers
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== RECOMMENDATION =====
@app.route("/recommend_products")
def recommend_products():
    try:
        product = request.args.get("product").strip().lower()

        matches = [p for p in recommend_model.index if product in p]

        if not matches:
            return jsonify({"error": "Product not found"})

        product = matches[0]

        similar = recommend_model[product].sort_values(ascending=False)[1:6]

        return jsonify({
            "recommendations": similar.index.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
