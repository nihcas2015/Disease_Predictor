import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

model=joblib.load(r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\random_forest\random_forest_model.joblib')
le=joblib.load(r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\random_forest\label_encoder.joblib')
scaler=joblib.load(r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\random_forest\scaler.joblib')
df=pd.read_csv(r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\random_forest\Final_Augmented_dataset_Diseases_and_Symptoms.csv')

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

scaled_data = scaler.transform(new_df)
prediction_probs = model.predict_proba(scaled_data)
predicted_class_indices = np.argsort(prediction_probs[0])[-3:][::-1] 
predicted_diseases = le.inverse_transform(predicted_class_indices)
top_probabilities = [float(prediction_probs[0][idx]) for idx in predicted_class_indices]

result_data = {
    "disease": str(predicted_diseases[0]), 
    "confidence": round(float(top_probabilities[0]) ,2)}

with open(r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\result.txt", "w") as file:
    json.dump(result_data, file)