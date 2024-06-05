from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn
import numpy as np

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
car = pd.read_csv("Cleaned Car.csv")


model2 = pickle.load(open("RidgeModel.pkl",'rb'))
house = pd.read_csv("Cleaned house.csv")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0,"Select Company")
    return render_template('car.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/index2')
def index2():
    locations = sorted(house['location'].unique())

    return render_template('house.html',locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company, car_model, year, fuel_type, kms_driven)
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    print(prediction)
    return str(np.round(prediction[0], 2))


@app.route('/predict2', methods=['POST'])
def predict2():
    locations = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(locations, bhk, bath, sqft)
    prediction = model2.predict(pd.DataFrame([[locations, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk']))*100000
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
