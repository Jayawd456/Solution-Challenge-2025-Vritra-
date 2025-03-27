'''
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

# Print current working directory for debugging
print("Current working directory:", os.getcwd())

# Print the absolute path to templates folder
template_dir = os.path.join(os.getcwd(), 'templates')
print("Templates directory:", template_dir)
print("Does templates directory exist?", os.path.exists(template_dir))

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and symptom list
try:
    model = joblib.load("disease_model.pkl")
    symptom_list = joblib.load("symptom_list.pkl")
    expected_features = len(symptom_list)
    print(f"✅ Model loaded successfully! Expected features: {expected_features}")
    # Print first 5 symptoms for reference
    print(f"Sample symptoms: {symptom_list[:5]}")
except Exception as e:
    print(f"❌ Error loading model or symptom list: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api_test")
def api_test():
    return jsonify({"status": "API is working", "message": "If you see this, the API is running correctly"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Print request data for debugging
        print("Request received at /predict endpoint")
        data = request.get_json()
        print(f"Request data: {data}")
        
        if not data or 'symptoms' not in data:
            return jsonify({"error": "No symptoms provided"}), 400
            
        symptoms_input = data.get("symptoms", "").lower()
        symptoms = [s.strip() for s in symptoms_input.split(",")]
        
        print(f"📩 Received symptoms: {symptoms}")
        
        # Create feature vector
        input_features = np.zeros(len(symptom_list))
        
        # Check if symptoms are valid
        found_symptoms = []
        not_found_symptoms = []
        
        for symptom in symptoms:
            if symptom in symptom_list:
                index = symptom_list.index(symptom)
                input_features[index] = 1
                found_symptoms.append(symptom)
            else:
                not_found_symptoms.append(symptom)
        
        if not found_symptoms:
            return jsonify({
                "error": f"None of the provided symptoms were recognized. Try using symptoms like: {', '.join(symptom_list[:5])}"
            }), 400
            
        # Predict disease
        prediction = model.predict([input_features])[0]
        
        result = {
            "disease": prediction,
            "found_symptoms": found_symptoms
        }
        
        if not_found_symptoms:
            result["warning"] = f"Some symptoms were not recognized: {', '.join(not_found_symptoms)}"
            
        print(f"🩺 Predicted Disease: {prediction}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route("/test")
def test_page():
    return render_template("test.html")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    #app.run(debug=True, port=5000)
    '''
    
import gradio as gr
import joblib
import numpy as np

# Load model and symptom list
model = None
symptom_list = []

try:
    model = joblib.load("disease_model.pkl")
    symptom_list = joblib.load("symptom_list.pkl")
    print(f"✅ Model loaded successfully! Total symptoms: {len(symptom_list)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Prediction function
def predict_disease(symptoms):
    if model is None or not symptom_list:
        return "❌ Model or symptom list not loaded."

    input_features = np.zeros(len(symptom_list))

    # Validate and process symptoms
    found_symptoms = [s for s in symptoms if s in symptom_list]
    not_found_symptoms = [s for s in symptoms if s not in symptom_list]

    if not found_symptoms:
        return "⚠️ None of the selected symptoms were recognized. Please select valid symptoms."

    for symptom in found_symptoms:
        index = symptom_list.index(symptom)
        input_features[index] = 1

    # Predict disease
    try:
        prediction = model.predict([input_features])[0]
        result = f"🩺 **Predicted Disease:** {prediction}"

        if not_found_symptoms:
            result += f"\n⚠️ Some symptoms were not recognized: {', '.join(not_found_symptoms)}"

        return result
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.CheckboxGroup(choices=symptom_list, label="Select Symptoms"),
    outputs="text",
    title="🩺 Disease Predictor",
    description="Select symptoms to predict a possible disease.",
    examples=[["fever", "headache"], ["cough", "sore throat"]],
)

# Launch Gradio app for Hugging Face
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
