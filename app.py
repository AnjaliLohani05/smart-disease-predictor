import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# =========================
# Prediction Function
# =========================
def predict(values, dic):

    # ---------------- DIABETES ----------------
    if len(values) == 8:

        dic2 = {
            'NewBMI_Obesity 1': 0,
            'NewBMI_Obesity 2': 0,
            'NewBMI_Obesity 3': 0,
            'NewBMI_Overweight': 0,
            'NewBMI_Underweight': 0,
            'NewInsulinScore_Normal': 0,
            'NewGlucose_Low': 0,
            'NewGlucose_Normal': 0,
            'NewGlucose_Overweight': 0,
            'NewGlucose_Secret': 0
        }

        # BMI Categories
        if dic['BMI'] <= 18.5:
            dic2['NewBMI_Underweight'] = 1
        elif 24.9 < dic['BMI'] <= 29.9:
            dic2['NewBMI_Overweight'] = 1
        elif 29.9 < dic['BMI'] <= 34.9:
            dic2['NewBMI_Obesity 1'] = 1
        elif 34.9 < dic['BMI'] <= 39.9:
            dic2['NewBMI_Obesity 2'] = 1
        elif dic['BMI'] > 39.9:
            dic2['NewBMI_Obesity 3'] = 1

        # Insulin Category
        if 16 <= dic['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        # Glucose Category
        if dic['Glucose'] <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < dic['Glucose'] <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < dic['Glucose'] <= 126:
            dic2['NewGlucose_Overweight'] = 1
        elif dic['Glucose'] > 126:
            dic2['NewGlucose_Secret'] = 1

        dic.update(dic2)

        model = pickle.load(open('models/diabetes.pkl', 'rb'))
        values = np.asarray(list(dic.values()))
        return model.predict(values.reshape(1, -1))[0]

    # ---------------- BREAST CANCER ----------------
    elif len(values) == 22:
        model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # ---------------- HEART ----------------
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # ---------------- KIDNEY ----------------
    elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # ---------------- LIVER ----------------
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    else:
        raise ValueError("Invalid number of inputs")


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes")
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer")
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart")
def heartPage():
    return render_template('heart.html')

@app.route("/kidney")
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver")
def liverPage():
    return render_template('liver.html')

@app.route("/malaria")
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia")
def pneumoniaPage():
    return render_template('pneumonia.html')


# =========================
# GENERAL PREDICT ROUTE
# =========================

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()

        # Convert all values to float safely
        for key in to_predict_dict:
            to_predict_dict[key] = float(to_predict_dict[key])

        to_predict_list = list(to_predict_dict.values())

        print("Received Data:", to_predict_list)
        print("Length:", len(to_predict_list))

        pred = predict(to_predict_list, to_predict_dict)

        return render_template('predict.html', pred=pred)

    except Exception as e:
        print("ERROR:", e)
        message = "Please enter valid data"
        return render_template("diabetes.html", message=message)


# =========================
# MALARIA IMAGE PREDICT
# =========================

@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        img = Image.open(request.files['image'])
        img.save("uploads/image.jpg")

        img = tf.keras.utils.load_img("uploads/image.jpg", target_size=(128, 128))
        img = tf.keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        model = tf.keras.models.load_model("models/malaria.h5")
        pred = np.argmax(model.predict(img))

        return render_template('malaria_predict.html', pred=pred)

    except:
        message = "Please upload an image"
        return render_template('malaria.html', message=message)


# =========================
# PNEUMONIA IMAGE PREDICT
# =========================

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        img = Image.open(request.files['image']).convert('L')
        img.save("uploads/image.jpg")

        img = tf.keras.utils.load_img("uploads/image.jpg", target_size=(128, 128))
        img = tf.keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        model = tf.keras.models.load_model("models/pneumonia.h5")
        pred = np.argmax(model.predict(img))

        return render_template('pneumonia_predict.html', pred=pred)

    except:
        message = "Please upload an image"
        return render_template('pneumonia.html', message=message)


# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(debug=True) 