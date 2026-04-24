from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "FactoryGuard AI is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Expected input features
        features = [
            data['temperature'],
            data['vibration'],
            data['pressure'],
            data['temp_mean_3'],
            data['vib_std_3'],
            data['temp_lag1'],
            data['temp_lag2']
        ]

        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]

        return jsonify({
            "failure_prediction": int(prediction),
            "failure_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)