import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model/svcmodel.pkl', 'rb'))

# StandardScaler instance and fit it to the training data
scaler = StandardScaler()

# Actual mean and standard deviation values from training data
mean_values = np.array([30.0, 0.1, 0.2, 25.0, 5.4, 100.0])
std_values = np.array([5.0, 0.2, 0.4, 4.0, 0.5, 20.0])

scaler.mean_ = mean_values
scaler.scale_ = std_values

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = ['age', 'hypertension', 'heartDisease', 'bmi', 'HbA1c', 'glucose']
        user_input = [float(request.form[feature]) for feature in features]
        input_data = scaler.transform([user_input])
        prediction = model.predict(input_data)[0]
        prediction_label = "Non-Diabetic" if prediction == 0 else "Diabetic"

        # Return the prediction label as JSON
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
