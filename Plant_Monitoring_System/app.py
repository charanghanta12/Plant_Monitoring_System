from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load plant disease model
plant_disease_model = tf.keras.models.load_model("models/plant_disease_model.keras")

# Load plant watering model and scaler
plant_watering_model = joblib.load("models/plant_watering_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Disease class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']


def model_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = plant_disease_model.predict(input_arr)
    return np.argmax(predictions)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('disease.html', error='No file uploaded')

        file = request.files['image']

        if file.filename == '':
            return render_template('disease.html', error='No file selected')

        if file:
            try:
                # Save the uploaded file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Make prediction
                result_index = model_prediction(filepath)
                prediction = class_names[result_index]

                # Format the prediction for better readability
                parts = prediction.split('___')
                plant_type = parts[0].replace('_', ' ')
                condition = parts[1].replace('_', ' ')

                formatted_prediction = f"{plant_type}: {condition}"

                # Pass both the prediction and the image path to the template
                return render_template('disease.html',
                                       prediction=formatted_prediction,
                                       image_path=os.path.join('uploads', file.filename))
            except Exception as e:
                return render_template('disease.html', error=f"Error: {str(e)}")

    return render_template('disease.html')


@app.route('/predict_watering', methods=['GET', 'POST'])
def predict_watering():
    if request.method == 'POST':
        try:
            # Get form data for all 14 input features
            input_data = (
                float(request.form['light_intensity']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['soil_moisture']),
                float(request.form['soil_temperature']),
                float(request.form['soil_ph']),
                float(request.form['soil_ec']),
                float(request.form['leaf_temperature']),
                float(request.form['atmospheric_pressure']),
                float(request.form['vapor_density']),
                float(request.form['heat_index']),
                float(request.form['rain_status']),
                float(request.form['cloud_status']),
                float(request.form['wind_status'])
            )

            # Reshape and scale the input data
            input_data_as_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_array.reshape(1, -1)
            std_data = scaler.transform(input_data_reshaped)

            # Make prediction
            prediction = plant_watering_model.predict(std_data)[0]

            # Determine watering recommendation based on prediction
            if prediction == 1:
                recommendation = "Your plant needs watering!"
                status = "needs_watering"
            else:
                recommendation = "Your plant does not need watering at this time."
                status = "no_watering"

            return render_template('watering.html', prediction=recommendation, status=status)
        except Exception as e:
            return render_template('watering.html', error=f"Error in prediction: {str(e)}")

    return render_template('watering.html')


if __name__ == '__main__':
    app.run(debug=True)