import joblib
import numpy as np
import pandas as pd
import json


if __name__ == "__main__":
   
    model_path = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\log\logistic_model.pkl"
    preprocessor_path = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\log\preprocessor.pkl"
    label_encoder_path = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\log\label_encoder.pkl"
    
   
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)
    
   
    df=pd.DataFrame(pd.read_csv(r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\log\dataset.csv'))
    df=df[df["Disease"].notna()]
    all_symptoms = set()

    symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]

    for col in symptom_cols:
        symptoms = df[col].dropna().unique()
        all_symptoms.update(symptoms)



 
    with open(r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\input.txt", "r") as file:
        user_input = file.read()
    
 
    user_symptoms = [s.strip().lower() for s in user_input.split(',')]
    
 
    input_data = {symptom: 0 for symptom in all_symptoms}
    
  
    for symptom in user_symptoms:
        normalized_symptom = f" {symptom}" if not symptom.startswith(" ") else symptom
        if normalized_symptom in all_symptoms:
            input_data[normalized_symptom] = 1
            print(f"Recorded symptom: {normalized_symptom}")
        else:
            print(f"Warning: '{symptom}' is not recognized and will be ignored.")
    
  
    input_df = pd.DataFrame([input_data])
    
   
    processed_data = preprocessor.transform(input_df)
    
   
    prediction = model.predict(processed_data)
    prediction_probs = model.predict_proba(processed_data)
    
  
    max_prob = np.max(prediction_probs)
    
   
    disease = label_encoder.inverse_transform(prediction)[0]
    
  
    result_data = {
        "disease": str(disease),
        "confidence": round(float(max_prob),2)
    }
    
 
    with open(r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\result.txt", "w") as file:
        json.dump(result_data, file)
