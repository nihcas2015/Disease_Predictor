import os
import json
import subprocess
from flask import Flask, render_template, request, jsonify
import pandas as pd

INPUT_FILE_PATH = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\input.txt"
OUTPUT_FILE_PATH = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\result.txt"
NN_SCRIPT_PATH = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\neural\load.py"
RF_SCRIPT_PATH = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\random_forest\load.py"
LR_SCRIPT_PATH = r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\log\load.py"

app = Flask(__name__)

df= pd.read_csv(r"C:\Users\nihca\OneDrive\Documents\vscode\edaproject\random_forest\Final_Augmented_dataset_Diseases_and_Symptoms.csv")
available_side_effects = [col for col in df.columns if col != 'diseases']


df=pd.DataFrame(pd.read_csv(r'C:\Users\nihca\OneDrive\Documents\vscode\edaproject\log\dataset.csv'))
df=df[df["Disease"].notna()]
all_symptoms = set()
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]

for col in symptom_cols:
    symptoms = df[col].dropna().unique()
    all_symptoms.update(symptoms)
all_symptoms = list(all_symptoms)



ALL_SYMPTOMS_NN = available_side_effects 
ALL_SYMPTOMS_RF = available_side_effects 
ALL_SYMPTOMS_LR = all_symptoms 

def run_model_script(script_path, symptoms_string):
    try:
       
        with open(INPUT_FILE_PATH, 'w') as f:
            f.write(symptoms_string)
        
       
        subprocess.run(["python", script_path], check=True)
        
      
        try:
            with open(OUTPUT_FILE_PATH, 'r') as f:
                result = json.load(f)
            return result
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {"error": f"Error reading model output: {str(e)}"}
            
    except subprocess.CalledProcessError as e:
        return {"error": f"Error running model script: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<model_type>')
def model_page(model_type):

    symptoms_list = []
    title = ""
    if model_type == 'nn':
        title = "Neural Network Predictor"
        symptoms_list = ALL_SYMPTOMS_NN
    elif model_type == 'rf':
        title = "Random Forest Predictor"
        symptoms_list = ALL_SYMPTOMS_RF
    elif model_type == 'lr':
        title = "Logistic Regression Predictor"
        symptoms_list = ALL_SYMPTOMS_LR
    else:
        return "Invalid model type", 404

    display_symptoms = [symptoms_list]

    return render_template('model_page.html',
                          model_type=model_type,
                          title=title,
                          symptoms=display_symptoms)

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
   
    try:
        data = request.get_json()
        symptoms_string = data.get('symptoms')

        if not symptoms_string:
            return jsonify({'error': 'No symptoms provided.'}), 400
        
      
        if model_type == 'nn':
            script_path = NN_SCRIPT_PATH
        elif model_type == 'rf':
            script_path = RF_SCRIPT_PATH
        elif model_type == 'lr':
            script_path = LR_SCRIPT_PATH
        else:
            return jsonify({'error': 'Invalid model type specified.'}), 400
        
      
        result = run_model_script(script_path, symptoms_string)
        
    
        if "error" in result:
            return jsonify(result), 500
        
      
        return jsonify(result)

    except KeyError:
        return jsonify({'error': 'Missing "symptoms" key in request data.'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)