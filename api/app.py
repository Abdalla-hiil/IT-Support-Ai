import os
from flask import Flask, request, jsonify, render_template # Added 'render_template'
from flask_cors import CORS
import joblib
import requests

# '../templates' is the magic path to find index.html
app = Flask(__name__, template_folder='../templates')
CORS(app)

# 1. LOAD MODELS SAFELY (Added '../' to find the folder outside 'api')
# 1. LOAD MODELS SAFELY (Removed the '../' for local testing)
try:
    model = joblib.load('models/it_priority_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    print("✅ AI BRAIN LOADED")
except Exception as e:
    print("❌ ERROR: CHECK YOUR 'models' FOLDER!")
    print(f"Details: {e}")

GOOGLE_URL = "https://script.google.com/macros/s/AKfycbxXcjoftO2G5zqBogZxqOxMP_qbCdYW0dg_3sK9KMX6WybYfRZ7ptr17IhlCvzAnxGM/exec"

# 2. HOME ROUTE (This was missing! It shows your HTML when you open the link)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_issue = data.get('text', '').lower().strip()

        # LOGIC LADDER
        high_keywords = ["emergency", "fire", "server down", "urgent"]
        low_keywords = ["thank you", "thanks", "hello", "hi"]

        if any(word in user_issue for word in high_keywords):
            prio = "High"
        elif any(word in user_issue for word in low_keywords):
            prio = "Low"
        else:
            # Only use AI if keywords don't match
            vector = tfidf.transform([user_issue])
            pred = model.predict(vector)[0]
            label_map = {0: "High", 1: "Medium", 2: "Low"}
            prio = label_map.get(pred, "Medium")

        # SEND TO GOOGLE (with safety timeout)
        try:
            requests.post(GOOGLE_URL, json={"text": user_issue, "priority": prio}, timeout=3)
        except:
            print("⚠️ Google Sheet was slow, but I kept the API alive!")

        return jsonify({"priority": prio, "status": "Success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # debug=False can sometimes prevent random shutdowns during startup
    app.run(debug=False, port=5000)