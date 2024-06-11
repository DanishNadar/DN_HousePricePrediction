from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application
app.static_folder = 'static'

# Importing the model object and standard scaler object
cv_model = pickle.load(open('HPP_flaskapp/models/hpp_model.pkl', 'rb'))
scaler = pickle.load(open('HPP_flaskapp/models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET', 'POST']) # GET gets the page, POST permits the posting of information to that page
def predict_datapoint():
    cv_model = pickle.load(open('HPP_flaskapp/models/hpp_model.pkl', 'rb'))
    if request.method == 'POST':
        area_income = float(request.form.get('avg_area_income'))
        house_age = float(request.form.get('avg_area_house_age'))
        number_of_rooms = float(request.form.get('avg_area_number_of_rooms'))
        number_of_bedrooms = float(request.form.get('avg_area_number_of_bedrooms'))
        area_population = float(request.form.get('area_population'))
        
        new_scaled_data = scaler.transform([[area_income, house_age, number_of_rooms, number_of_bedrooms, area_population]])
        result = cv_model.predict(new_scaled_data)
        prediction_value = result[0][0]
        formatted_result = f'{prediction_value:,.2f}'
        return render_template('predict_data.html', prediction=formatted_result)
    else:
        return render_template('predict_data.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0')