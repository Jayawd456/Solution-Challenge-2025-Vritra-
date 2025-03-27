import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Get the absolute directory of the current script
base_dir = os.path.dirname(__file__)  

# Dataset path
dataset_path = os.path.join(base_dir, "dataset.csv")

# Check if dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

# Load the dataset
df = pd.read_csv(dataset_path)

# Fill missing values
df.fillna("None", inplace=True)

# Define the correct list of 105 symptoms
symptom_list = [
    'abdominal_pain', 'acidity', 'altered_sensorium', 'anxiety', 'back_pain', 
    'blackheads', 'bladder_discomfort', 'blister', 'bloody_stool', 'blurred_and_distorted_vision', 
    'breathlessness', 'bruising', 'burning_micturition', 'chest_pain', 'chills', 
    'cold_hands_and_feets', 'constipation', 'continuous_feel_of_urine', 'continuous_sneezing', 'cough', 
    'cramps', 'dark_urine', 'dehydration', 'depression', 'diarrhoea', 
    'dischromic_patches', 'distention_of_abdomen', 'dizziness', 'enlarged_thyroid', 'excessive_hunger', 
    'extra_marital_contacts', 'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 
    'foul_smell_of_urine', 'headache', 'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 
    'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level', 'irritability', 
    'irritation_in_anus', 'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 
    'loss_of_appetite', 'loss_of_balance', 'malaise', 'mild_fever', 'mood_swings', 
    'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 
    'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_during_bowel_movements', 
    'pain_in_anal_region', 'painful_walking', 'passage_of_gases', 'patches_in_throat', 'phlegm', 
    'prominent_veins_on_calf', 'puffy_face_and_eyes', 'pus_filled_pimples', 'red_sore_around_nose', 
    'restlessness', 'scurring', 'shivering', 'silver_like_dusting', 'skin_peeling', 
    'skin_rash', 'small_dents_in_nails', 'spinning_movements', 'spotting_urination', 'stiff_neck', 
    'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 
    'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_legs', 'ulcers_on_tongue', 'unsteadiness', 
    'vomiting', 'watering_from_eyes', 'weakness_in_limbs', 'weakness_of_one_body_side', 'weight_gain', 
    'weight_loss', 'yellow_crust_ooze', 'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin', 'itching'
]

# Selecting only the symptom columns from dataset
symptom_columns = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5", "Symptom_6", "Symptom_7"]

# Convert symptoms to lowercase and remove extra spaces
df[symptom_columns] = df[symptom_columns].apply(lambda x: x.str.strip().str.lower())

# Convert the dataframe into a list of symptom sets (rows)
X_raw = df[symptom_columns].values.tolist()

# Convert symptoms into one-hot encoding manually
X = []
for row in X_raw:
    row_vector = [1 if symptom in row else 0 for symptom in symptom_list]
    X.append(row_vector)

X = pd.DataFrame(X, columns=symptom_list)  # Convert to DataFrame for better readability

# Target variable (Disease)
y = df["Disease"]

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Checking model accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

# Paths to save files
model_path = os.path.join(base_dir, "disease_model.pkl")
symptom_path = os.path.join(base_dir, "symptom_list.pkl")

# Saving the model and symptom list
joblib.dump(model, model_path)
joblib.dump(symptom_list, symptom_path)

# Debugging prints
print(f"✅ Model saved at: {model_path}")
print(f"✅ Symptom list saved at: {symptom_path}")
