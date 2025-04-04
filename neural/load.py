# Import necessary libraries
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys
import json

MODEL_PATH = r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\neural\disease_classification_model.keras'
SCALER_PATH = r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\neural\standard_scaler.pkl'
MINMAX_PATH = r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\neural\min_max_scaler.pkl'
LABEL_ENCODER_PATH = r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\neural\label_encoder.pkl'

model = keras.models.load_model(MODEL_PATH)
standard_scaler = joblib.load(SCALER_PATH)
min_max_scaler = joblib.load(MINMAX_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def preprocess_data(new_data):
    
    scaled_data = standard_scaler.transform(new_data)
    normalized_data = min_max_scaler.transform(scaled_data)
    return normalized_data

def predict_disease(new_data):
    
    processed_data = preprocess_data(new_data)
    prediction_probs = model.predict(processed_data)
    predicted_class_indices = np.argmax(prediction_probs, axis=1)
    predicted_diseases = label_encoder.inverse_transform(predicted_class_indices)
    return predicted_diseases, prediction_probs

def get_confidence_levels(prediction_probabilities):
    
    confidence_levels = np.max(prediction_probabilities, axis=1)
    return confidence_levels


df=pd.read_csv(r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\neural\Final_Augmented_dataset_Diseases_and_Symptoms.csv')
available_side_effects = [col for col in df.columns if col != 'diseases']

with open(r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\input.txt","r") as file:
    user_input = file.read()
user_side_effects = [effect.strip().lower() for effect in user_input.split(',')]
        
new_df = pd.DataFrame(0, index=[0], columns=available_side_effects)

for effect in user_side_effects:
    if effect in available_side_effects: 
        new_df.loc[0, effect] = 1  
        print(f"Recorded symptom: {effect}")
    else:
        print(f"Warning: '{effect}' is not a recognized symptom and will be ignored.")


predicted_diseases, prediction_probs = predict_disease(new_df)
confidence_levels = get_confidence_levels(prediction_probs)
        
results = pd.DataFrame({
            'Predicted_Disease': predicted_diseases,
            'Confidence': confidence_levels
        })
        

        
       
disease = str(predicted_diseases[0])
confidence = confidence_levels[0]
result_data = {
    "disease": disease,
    "confidence": round(float(confidence),2)  
}


with open(r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\result.txt", "w") as file:
    json.dump(result_data, file)