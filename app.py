from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the serialized model pipeline artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "ExtraaLearn Lead Conversion Prediction API is operational."
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input payload from client request
        data = request.get_json()

        # Convert incoming dictionary payload into DataFrame format expected by preprocessor
        input_df = pd.DataFrame([data])

        # Predict class label and probability using serialized pipeline
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "conversion_probability": round(probability, 4),
            "lead_segment": "High Intent (Target for Sales)" if prediction == 1 else "Low Intent"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
