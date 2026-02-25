import os
import pickle
import numpy as np
import torch
from flask import Flask, render_template, request
from src.utils.preprocess import preprocess_single_sample, preprocess_image_pytorch, get_image_model

app = Flask(__name__)

# =========================
# Model Caching
# =========================
MODELS = {}
IMAGE_MODELS = {}
CLASS_MAPPINGS = {}

def load_models():
    """Load all available models into memory at startup."""
    models_dir = "models"
    
    # Tabular Models (Pickle)
    diseases = ["diabetes", "breast_cancer", "heart", "kidney", "liver"]
    for disease in diseases:
        model_path = os.path.join(models_dir, f"{disease}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                MODELS[disease] = pickle.load(f)
            print(f"Loaded {disease} model.")

    # Image Models (PyTorch)
    image_diseases = ["malaria", "pneumonia"]
    for disease in image_diseases:
        model_path = os.path.join(models_dir, f"{disease}.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                model = get_image_model(disease)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                IMAGE_MODELS[disease] = model
                CLASS_MAPPINGS[disease] = checkpoint.get('class_to_idx', {0: 'Healthy', 1: 'Diseased'})
                print(f"Loaded {disease} PyTorch model.")
            except Exception as e:
                print(f"Error loading {disease} model: {e}")

# Load models on import/startup
load_models()

# =========================
# Prediction Logic
# =========================

def predict_tabular(values, dic):
    disease_map = {
        8: "diabetes",
        20: "breast_cancer",
        13: "heart",
        24: "kidney",
        10: "liver"
    }
    
    disease = disease_map.get(len(values))
    if not disease:
        raise ValueError(f"Invalid number of inputs: {len(values)}")
        
    if disease not in MODELS:
        # Try reloading once in case it was just trained
        load_models()
        if disease not in MODELS:
            raise FileNotFoundError(f"Model for {disease} not found.")

    X = preprocess_single_sample(disease, dic)
    return MODELS[disease].predict(X)[0]

def predict_image(disease, image_file):
    if disease not in IMAGE_MODELS:
        load_models()
        if disease not in IMAGE_MODELS:
            raise FileNotFoundError(f"PyTorch model for {disease} not found.")

    model = IMAGE_MODELS[disease]
    X = preprocess_image_pytorch(image_file, target_size=(128, 128))
    
    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
        
    return int(preds[0])

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
# PREDICTION ENDPOINTS
# =========================

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        clean_dict = {}
        for k, v in to_predict_dict.items():
            val = v.strip()
            if val == "":
                clean_dict[k] = 0.0 # Default to 0.0 for empty inputs
            else:
                try:
                    clean_dict[k] = float(val)
                except ValueError:
                    clean_dict[k] = val # Keep as string for categorical handling in preprocess.py
        
        to_predict_list = list(clean_dict.values())

        pred = predict_tabular(to_predict_list, clean_dict)
        return render_template('predict.html', pred=pred)

    except Exception as e:
        print("ERROR:", e)
        return render_template("home.html", message="Error processing request. Please try again.")

@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        if 'image' not in request.files:
            return render_template('malaria.html', message="Please upload an image")
            
        file = request.files['image']
        pred = predict_image("malaria", file)
        return render_template('malaria_predict.html', pred=pred)

    except Exception as e:
        print("ERROR:", e)
        return render_template('malaria.html', message="Prediction failed. Ensure model is trained.")

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' not in request.files:
            return render_template('pneumonia.html', message="Please upload an image")
            
        file = request.files['image']
        pred = predict_image("pneumonia", file)
        return render_template('pneumonia_predict.html', pred=pred)

    except Exception as e:
        print("ERROR:", e)
        return render_template('pneumonia.html', message="Prediction failed. Ensure model is trained.")

if __name__ == '__main__':
    app.run(debug=True)
 