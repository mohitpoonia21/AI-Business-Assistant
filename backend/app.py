from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models")

# ===== LOAD MODELS SAFELY =====
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print("Error loading model:", e)
        return None

sales_model = load_model(os.path.join(MODEL_PATH, "sales_model.pkl"))
cluster_model = load_model(os.path.join(MODEL_PATH, "customer_cluster.pkl"))
recommend_model = load_model(os.path.join(MODEL_PATH, "recommendation.pkl"))

# ===== SALES =====
@app.route("/predict_sales")
def predict_sales():
    try:
        month = int(request.args.get("month"))

        if sales_model:
            prediction = sales_model.predict([[month]])[0]
        else:
            prediction = month * 10  # fallback

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== CLUSTER =====
@app.route("/predict_cluster")
def predict_cluster():
    try:
        spending = float(request.args.get("spending"))

        if cluster_model:
            cluster = cluster_model.predict([[spending]])[0]
        else:
            cluster = 0

        return jsonify({
            "cluster": int(cluster)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== RECOMMEND =====
@app.route("/recommend_products")
def recommend_products():
    try:
        product = request.args.get("product", "").upper()

        if recommend_model is None:
            return jsonify({"error": "Model not loaded"})

        if product not in recommend_model.index:
            return jsonify({"error": f"{product} not found"})

        recs = recommend_model[product].sort_values(ascending=False)[1:6]

        return jsonify({
            "recommendations": recs.index.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
