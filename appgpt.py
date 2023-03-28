from flask import Flask, render_template, request
import numpy as np
import joblib
import json

# Load your trained model here
model = joblib.load("Model.pkl")

feat_cols = ['age','workclass','education','marital_status','occupation',
             'relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country']

context_dict = {
    'feats': feat_cols,
    'zip': zip,
    'range': range,
    'len': len,
    'list': list
}

# Initiate the application
app = Flask(__name__)

@app.route('/')
def index():

    return render_template('salpred.html', **context_dict)

@app.route('/predict', methods=['POST'])
def predict_salary():
    age = int(request.form['age'])
    workclass = (request.form['workclass']).encode('utf-8')
    education = (request.form['education']).encode('utf-8')
    marital_status = (request.form['marital_status']).encode('utf-8')
    occupation = (request.form['occupation']).encode('utf-8')
    relationship = (request.form['relationship']).encode('utf-8')
    race = (request.form['race']).encode('utf-8')
    sex = (request.form['sex']).encode('utf-8')
    capital_gain = float(request.form['capital_gain'])
    capital_loss = float(request.form['capital_loss'])
    hours_per_week = int(request.form['hours_per_week'])
    native_country = (request.form['native_country']).encode('utf-8')

    X = np.array([[age,workclass,education,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country]])

    # Make prediction
    pred = model.predict(X)
    print('Prediction:', pred)

    # Return prediction as JSON response
    return json.dumps({'prediction': pred.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
