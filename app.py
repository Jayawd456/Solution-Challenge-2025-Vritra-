'''
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder="C:/Users/anura/OneDrive/Desktop/python/disease/templates/index.html")

# Load the trained model
model = joblib.load("disease_model.pkl")
symptom_list = joblib.load("symptom_list.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Loads frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data["symptoms"].split(", ")  # Convert to list
        input_features = np.zeros(len(symptom_list))  # One-hot encoding

        for symptom in symptoms:
            if symptom in symptom_list:
                index = symptom_list.index(symptom)
                input_features[index] = 1  

        prediction = model.predict([input_features])[0]
        return jsonify({"predicted_disease": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
'''
import gradio as gr
import joblib
import numpy as np
import os

# Load the trained model and symptom list
def load_model():
    try:
        model = joblib.load("disease_model.pkl")
        symptom_list = joblib.load("symptom_list.pkl")
        print(f"✅ Model loaded successfully! Total symptoms: {len(symptom_list)}")
        return model, symptom_list
    except FileNotFoundError as e:
        print(f"❌ File Not Found: {e}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    return None, []

model, symptom_list = load_model()

# Function to make predictions
def predict_disease(symptoms):
    if not model:
        return "❌ Error: Model not loaded. Please check the backend."

    if not symptoms:
        return "⚠️ Please select at least one symptom."

    input_features = np.zeros(len(symptom_list))  # Create input feature array

    for symptom in symptoms:
        if symptom in symptom_list:
            index = symptom_list.index(symptom)
            input_features[index] = 1

    # Predict disease
    prediction = model.predict([input_features])[0]
    return f"🩺 Predicted Disease: {prediction}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.CheckboxGroup(choices=symptom_list, label="Select Symptoms"),
    outputs="text",
    title="Disease Prediction System",
    description="Select your symptoms to get a disease prediction.",
)

# Launch the Gradio app (dynamic port selection)
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=True)
