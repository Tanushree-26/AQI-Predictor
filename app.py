from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and feature names
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form.get(f)) for f in features]
        prediction = model.predict([input_data])[0]
        return render_template('index.html', prediction=f"Predicted AQI: {prediction:.2f}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
