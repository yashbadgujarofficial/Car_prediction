import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime

application = Flask(__name__)
app = application


# Load saved files
model = pickle.load(open('models/random_forest.pk1', 'rb'))
preprocessor = pickle.load(open('models/preprocessor.pk1', 'rb'))
label_encoder = pickle.load(open('models/label.pk1', 'rb'))

# Comprehensive car data
companies = [
    'Toyota', 'Hyundai', 'Honda', 'Maruti', 'Mahindra', 'Tata', 'BMW', 'Audi', 'Ford', 'Nissan',
    'Skoda', 'Volkswagen', 'Kia', 'Jeep', 'MG', 'Renault', 'Datsun', 'Force', 'Ambassador', 'Isuzu',
    'Citroen', 'Chevrolet', 'Fiat', 'Mitsubishi', 'Porsche', 'Jaguar', 'Land Rover', 'Mini', 'Volvo', 'Subaru'
]

car_models = ['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'i10', 'Venue',
       'Swift', 'Verna', 'Duster', 'Cooper', 'Ciaz', 'C-Class', 'Innova',
       'Baleno', 'Swift Dzire', 'Vento', 'Creta', 'City', 'Bolero',
       'Fortuner', 'KWID', 'Amaze', 'Santro', 'XUV500', 'KUV100', 'Ignis',
       'RediGO', 'Scorpio', 'Marazzo', 'Aspire', 'Figo', 'Vitara',
       'Tiago', 'Polo', 'Seltos', 'Celerio', 'GO', '5', 'CR-V',
       'Endeavour', 'KUV', 'Jazz', '3', 'A4', 'Tigor', 'Ertiga', 'Safari',
       'Thar', 'Hexa', 'Rover', 'Eeco', 'A6', 'E-Class', 'Q7', 'Z4', '6',
       'XF', 'X5', 'Hector', 'Civic', 'D-Max', 'Cayenne', 'X1', 'Rapid',
       'Freestyle', 'Superb', 'Nexon', 'XUV300', 'Dzire VXI', 'S90',
       'WR-V', 'XL6', 'Triber', 'ES', 'Wrangler', 'Camry', 'Elantra',
       'Yaris', 'GL-Class', '7', 'S-Presso', 'Dzire LXI', 'Aura', 'XC',
       'Ghibli', 'Continental', 'CR', 'Kicks', 'S-Class', 'Tucson',
       'Harrier', 'X3', 'Octavia', 'Compass', 'CLS', 'redi-GO', 'Glanza',
       'Macan', 'X4', 'Dzire ZXI', 'XC90', 'F-PACE', 'A8', 'MUX',
       'GTC4Lusso', 'GLS', 'X-Trail', 'XE', 'XC60', 'Panamera', 'Alturas',
       'Altroz', 'NX', 'Carnival', 'C', 'RX', 'Ghost', 'Quattroporte',
       'Gurkha']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html', car_models=car_models)


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':

        model_name = request.form.get('model')
        year = int(request.form.get('year'))
        kilometers = int(request.form.get('kilometers'))
        fuel_type = request.form.get('fuel_type')
        seller_type = request.form.get('seller_type')
        transmission_type = request.form.get('transmission_type')
        mileage = float(request.form.get('mileage'))
        engine = float(request.form.get('engine'))
        seats = int(request.form.get('seats'))
        max_power = float(request.form.get('max_power'))
        
        # Calculate vehicle age
        current_year = datetime.now().year
        vehicle_age = current_year - year

        # Label encode model column
        model_encoded = label_encoder.transform([model_name])[0]

        # DataFrame input - Include vehicle_age
        input_data = pd.DataFrame([{
            'model': model_encoded,
            'year': year,
            'vehicle_age': vehicle_age,
            'km_driven': kilometers,
            'seller_type': seller_type,
            'fuel_type': fuel_type,
            'transmission_type': transmission_type,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats
        }])

        # Preprocess
        transformed_data = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(transformed_data)
        predicted_value = round(prediction[0], 2)

        # If request came from AJAX (fetch/XHR), return JSON so client can update without reload
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'result': predicted_value})

        return render_template(
            'home.html',
            car_models=car_models,
            result=predicted_value
        )

    else:
        return render_template('home.html', car_models=car_models)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)