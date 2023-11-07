import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model/svcmodel.pkl', 'rb'))

feature_names = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = [float(request.form.get(field, 0)) for field in feature_names]
    prediction = model.predict([user_input])[0]
    # Interpret the prediction result
    result = 'Not Diabetic' if prediction == 0 else 'Diabetic'
    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
