import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler


# ──────────────────────────────── DIABETES ────────────────────────────────────

def _engineer_diabetes_features(df):
    """Internal helper to apply the same feature engineering to train and test/inference."""
    # BMI category
    df["NewBMI"] = "Normal"
    df.loc[df["BMI"] <= 18.5, "NewBMI"] = "Underweight"
    df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = "Overweight"
    df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = "Obesity 1"
    df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = "Obesity 2"
    df.loc[df["BMI"] > 39.9, "NewBMI"] = "Obesity 3"

    # Insulin score
    df["NewInsulinScore"] = "Abnormal"
    df.loc[(df["Insulin"] >= 16) & (df["Insulin"] <= 166), "NewInsulinScore"] = "Normal"

    # Glucose category
    df["NewGlucose"] = "Normal"
    df.loc[df["Glucose"] <= 70, "NewGlucose"] = "Low"
    df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = "Normal"
    df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = "Overweight"
    df.loc[df["Glucose"] > 126, "NewGlucose"] = "Secret"

    return df


def preprocess_diabetes(csv_path: str, test_size: float = 0.30, random_state: int = 0):
    df = pd.read_csv(csv_path)

    # Replace 0 with NaN for columns where 0 is physiologically impossible
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    # Fill NaN with per-outcome median
    def median_target(col):
        return df.groupby("Outcome")[col].median()

    for col in zero_cols:
        medians = median_target(col)
        for outcome_val in df["Outcome"].unique():
            df.loc[(df["Outcome"] == outcome_val) & df[col].isnull(), col] = medians[outcome_val]

    df = _engineer_diabetes_features(df)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

    y = df["Outcome"]
    X = df.drop(["Outcome"], axis=1)

    # RobustScaler on raw numeric columns
    scaler_r = RobustScaler()
    X_scaled = pd.DataFrame(scaler_r.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # StandardScaler
    scaler_s = StandardScaler()
    X_train_final = scaler_s.fit_transform(X_train)
    X_test_final = scaler_s.transform(X_test)

    # Return scalers so they can be saved
    return X_train_final, X_test_final, y_train, y_test, scaler_r, scaler_s, X.columns.tolist()


# ──────────────────────────────── HEART ───────────────────────────────────────

def preprocess_heart(csv_path: str, test_size: float = 0.30, random_state: int = 0):
    df = pd.read_csv(csv_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ──────────────────────────────── BREAST CANCER ───────────────────────────────

def preprocess_breast_cancer(csv_path: str, test_size: float = 0.30, random_state: int = 0):
    df = pd.read_csv(csv_path)
    
    # 20 features present in the current breast_cancer.csv (Mean and SE only)
    feature_cols = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", 
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se"
    ]

    y = df["diagnosis"].map({"M": 1, "B": 0})
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train)
    X_test_final = scaler.transform(X_test)

    # Return scaler and feature order
    return X_train_final, X_test_final, y_train, y_test, scaler, feature_cols


# ──────────────────────────────── KIDNEY ──────────────────────────────────────

def preprocess_kidney(csv_path: str, test_size: float = 0.30, random_state: int = 0):
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    if "classification" in df.columns:
        target_col = "classification"
    elif "class" in df.columns:
        target_col = "class"
    else:
        target_col = df.columns[-1]

    # Target Mapping: ckd -> 1, notckd -> 0 (for UI Positive/Negative)
    df[target_col] = df[target_col].str.strip().replace({
        "ckd": 1, "ckd\t": 1, "notckd": 0, "not ckd": 0, "\tno": 0
    })
    
    # Manual Mapping for categorical features to match HTML form (0/1)
    # rbc, pc: abnormal -> 0, normal -> 1
    # pcc, ba: notpresent -> 0, present -> 1
    # htn, dm, cad, pe, ane: no -> 0, yes -> 1
    # appet: good -> 0, poor -> 1
    
    binary_maps = {
        'rbc': {'abnormal': 0, 'normal': 1},
        'pc': {'abnormal': 0, 'normal': 1},
        'pcc': {'notpresent': 0, 'present': 1},
        'ba': {'notpresent': 0, 'present': 1},
        'htn': {'no': 0, 'yes': 1},
        'dm': {'no': 0, '\tno': 0, 'yes': 1, '\tyes': 1, ' yes': 1},
        'cad': {'no': 0, '\tno': 0, 'yes': 1},
        'pe': {'no': 0, 'yes': 1},
        'ane': {'no': 0, 'yes': 1},
        'appet': {'good': 0, 'poor': 1}
    }

    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].str.strip().replace(mapping)

    feature_cols = [c for c in df.columns if c != target_col]
    num_cols, cat_cols = [], []
    for col in feature_cols:
        # After mapping, many are now numeric (0/1)
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except (ValueError, TypeError):
                cat_cols.append(col)
        else:
            num_cols.append(col)

    def random_value_imputation(feature):
        n_miss = df[feature].isna().sum()
        if n_miss == 0: return
        pool = df[feature].dropna()
        if pool.empty:
            df[feature] = df[feature].fillna(0)
            return
        random_sample = pool.sample(n_miss, random_state=42, replace=True)
        random_sample.index = df[df[feature].isnull()].index
        df.loc[df[feature].isnull(), feature] = random_sample

    for col in num_cols:
        if df[col].isna().sum() > 0:
            if df[col].isna().sum() / len(df) > 0.1:
                random_value_imputation(col)
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    modes = {}
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            modes[col] = mode_val
        else:
            modes[col] = df[col].mode()[0] if not df[col].empty else "unknown"

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Ensure we fit on string representations
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    # If already numeric (after our mapping), fit_transform might still be needed if there are NaNs or weird types
    df[target_col] = df[target_col].fillna(0).astype(int)

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state), encoders, X.columns.tolist(), modes


# ──────────────────────────────── LIVER ───────────────────────────────────────

def preprocess_liver(csv_path: str, test_size: float = 0.30, random_state: int = 0):
    ILPD_COLS = [
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Protiens",
        "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"
    ]

    df = pd.read_csv(csv_path, header=None)
    df.columns = ILPD_COLS
    df = df.drop_duplicates().dropna(how="any")

    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0]).astype(int)

    y = df["Dataset"]
    X = df.drop("Dataset", axis=1)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ──────────────────────────────── INFERENCE ───────────────────────────────────

def preprocess_single_sample(disease: str, data: dict, models_dir: str = "models"):
    """
    Applies the same preprocessing to a single sample for inference.
    `data` should be a dictionary of raw inputs from the form.
    """
    if disease == "diabetes":
        df = pd.DataFrame([data])
        # Expected features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        df = _engineer_diabetes_features(df)
        
        # Load column order and dummy structure
        with open(os.path.join(models_dir, "diabetes_features.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
            
        # One-hot encode with full categories to match training
        df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"])
        
        # Reindex to ensure same columns as training (fills missing dummies with 0)
        df = df.reindex(columns=feature_cols, fill_value=0)
        
        # Scale
        with open(os.path.join(models_dir, "diabetes_scaler_r.pkl"), "rb") as f:
            scaler_r = pickle.load(f)
        with open(os.path.join(models_dir, "diabetes_scaler_s.pkl"), "rb") as f:
            scaler_s = pickle.load(f)
            
        X = scaler_r.transform(df)
        X = scaler_s.transform(X)
        return X

    elif disease == "breast_cancer":
        df = pd.DataFrame([data])
        with open(os.path.join(models_dir, "breast_cancer_features.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
        X = df[feature_cols]
        with open(os.path.join(models_dir, "breast_cancer_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return scaler.transform(X)

    elif disease == "kidney":
        # Data keys: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
        # The form now sends numeric 0/1 for categoricals, so we just need correct order and float conversion
        df = pd.DataFrame([data])
        
        with open(os.path.join(models_dir, "kidney_features.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
            
        # Ensure numeric conversion for all fields
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Ensure column order matches training
        df = df.reindex(columns=feature_cols, fill_value=0)
        
        return df.values

    elif disease == "liver":
        # Data keys: Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio
        df = pd.DataFrame([data])
        # Force Age to float if it came as string
        df["Age"] = pd.to_numeric(df["Age"])
        # Handle gender
        if "Gender" in df.columns:
            if isinstance(df["Gender"].iloc[0], str):
                df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
        return df.values

    elif disease == "heart":
        df = pd.DataFrame([data])
        return df.values

    return None


# ────────────────────────────── IMAGE INFERENCE ───────────────────────────────

def preprocess_image(image_file, target_size=(128, 128), grayscale=False):
    """
    Common preprocessing for image models (Numpy version).
    """
    from PIL import Image
    
    img = Image.open(image_file).convert('RGB')
    if grayscale:
        img = img.convert('L')
        
    img = img.resize(target_size)
    img_array = np.array(img)
    
    if grayscale:
        img_array = np.expand_dims(img_array, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def preprocess_image_pytorch(image_file, target_size=(128, 128), grayscale=False):
    """
    PyTorch specific preprocessing.
    """
    from PIL import Image
    import torch
    from torchvision import transforms

    img = Image.open(image_file).convert('RGB')
    
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if grayscale:
        # Pretrained models expect 3-channel input even for grayscale
        pass # Already converted to RGB above
        
    transform = transforms.Compose(transform_list)
    tensor = transform(img).unsqueeze(0) # (1, 3, H, W)
    return tensor

def get_image_model(disease, num_classes=2):
    """
    Returns the PyTorch model architecture for the given disease.
    """
    import torch.nn as nn
    from torchvision import models
    
    if disease == "malaria":
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
        return model
    elif disease == "pneumonia":
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        return model
    return None
