import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import json

# Medical training dataset
training_data = [
    # Fever/Flu cases
    {"symptoms": "fever high temperature chills body ache", "age": 35, "gender": "male", "diagnosis": "Acute Febrile Illness", "category": "fever"},
    {"symptoms": "fever headache fatigue weakness", "age": 28, "gender": "female", "diagnosis": "Acute Febrile Illness", "category": "fever"},
    {"symptoms": "high fever sweating body pain", "age": 42, "gender": "male", "diagnosis": "Acute Febrile Illness", "category": "fever"},
    {"symptoms": "temperature pyrexia malaise", "age": 25, "gender": "female", "diagnosis": "Acute Febrile Illness", "category": "fever"},
    
    # Diabetes cases
    {"symptoms": "high blood sugar frequent urination thirst", "age": 55, "gender": "male", "diagnosis": "Type 2 Diabetes Mellitus", "category": "diabetes"},
    {"symptoms": "diabetes hyperglycemia fatigue blurred vision", "age": 48, "gender": "female", "diagnosis": "Type 2 Diabetes Mellitus", "category": "diabetes"},
    {"symptoms": "excessive thirst hunger weight loss", "age": 52, "gender": "male", "diagnosis": "Type 2 Diabetes Mellitus", "category": "diabetes"},
    {"symptoms": "high glucose slow healing frequent infections", "age": 60, "gender": "female", "diagnosis": "Type 2 Diabetes Mellitus", "category": "diabetes"},
    
    # Cold/URTI cases
    {"symptoms": "cough cold sneezing runny nose", "age": 30, "gender": "female", "diagnosis": "Upper Respiratory Tract Infection", "category": "cold"},
    {"symptoms": "sore throat cough congestion", "age": 22, "gender": "male", "diagnosis": "Upper Respiratory Tract Infection", "category": "cold"},
    {"symptoms": "nasal congestion sneezing watery eyes", "age": 35, "gender": "female", "diagnosis": "Upper Respiratory Tract Infection", "category": "cold"},
    {"symptoms": "cold symptoms mucus cough", "age": 28, "gender": "male", "diagnosis": "Upper Respiratory Tract Infection", "category": "cold"},
    
    # Headache/Migraine cases
    {"symptoms": "severe headache pain sensitivity light", "age": 32, "gender": "female", "diagnosis": "Migraine", "category": "headache"},
    {"symptoms": "headache tension stress neck pain", "age": 40, "gender": "male", "diagnosis": "Tension Headache", "category": "headache"},
    {"symptoms": "migraine nausea visual disturbance", "age": 29, "gender": "female", "diagnosis": "Migraine", "category": "headache"},
    {"symptoms": "persistent headache dizziness", "age": 45, "gender": "male", "diagnosis": "Tension Headache", "category": "headache"},
    
    # Hypertension cases
    {"symptoms": "high blood pressure headache dizziness", "age": 58, "gender": "male", "diagnosis": "Hypertension", "category": "hypertension"},
    {"symptoms": "hypertension chest discomfort shortness breath", "age": 62, "gender": "female", "diagnosis": "Hypertension", "category": "hypertension"},
    {"symptoms": "elevated bp nosebleed fatigue", "age": 55, "gender": "male", "diagnosis": "Hypertension", "category": "hypertension"},
    
    # Asthma cases
    {"symptoms": "wheezing difficulty breathing chest tightness", "age": 25, "gender": "female", "diagnosis": "Asthma", "category": "asthma"},
    {"symptoms": "asthma attack shortness breath cough", "age": 32, "gender": "male", "diagnosis": "Asthma", "category": "asthma"},
    {"symptoms": "breathing difficulty wheezing night", "age": 28, "gender": "female", "diagnosis": "Asthma", "category": "asthma"},
    
    # Gastritis/Stomach cases
    {"symptoms": "stomach pain burning sensation nausea", "age": 38, "gender": "male", "diagnosis": "Gastritis", "category": "gastric"},
    {"symptoms": "acidity heartburn indigestion", "age": 42, "gender": "female", "diagnosis": "Gastritis", "category": "gastric"},
    {"symptoms": "abdominal pain bloating gas", "age": 35, "gender": "male", "diagnosis": "Gastritis", "category": "gastric"},
    
    # Allergy cases
    {"symptoms": "skin rash itching hives", "age": 27, "gender": "female", "diagnosis": "Allergic Reaction", "category": "allergy"},
    {"symptoms": "allergic reaction swelling face", "age": 33, "gender": "male", "diagnosis": "Allergic Reaction", "category": "allergy"},
    {"symptoms": "itchy eyes sneezing seasonal allergy", "age": 29, "gender": "female", "diagnosis": "Allergic Rhinitis", "category": "allergy"},
    
    # Arthritis cases
    {"symptoms": "joint pain swelling stiffness morning", "age": 65, "gender": "female", "diagnosis": "Osteoarthritis", "category": "arthritis"},
    {"symptoms": "arthritis knee pain difficulty walking", "age": 58, "gender": "male", "diagnosis": "Osteoarthritis", "category": "arthritis"},
    
    # Anxiety/Depression cases
    {"symptoms": "anxiety worry nervousness panic", "age": 30, "gender": "female", "diagnosis": "Anxiety Disorder", "category": "mental_health"},
    {"symptoms": "depression sadness loss interest fatigue", "age": 35, "gender": "male", "diagnosis": "Depression", "category": "mental_health"},
]

# Treatment recommendations database
treatment_db = {
    "fever": {
        "Medicine": "- Paracetamol 500 mg: Every 6 hours for fever\n- Ibuprofen 400 mg: Alternative if no contraindications",
        "Alternative": "- Cold compress on forehead\n- Sponge bath with lukewarm water",
        "Lifestyle": "Hydration (8-10 glasses water), rest, light diet, avoid cold beverages.",
        "Red Flags": "Fever >3 days, severe headache, rash, difficulty breathing, confusion.",
        "Follow-Up": "Review in 48 hours if fever persists or worsens."
    },
    "diabetes": {
        "Medicine": "- Metformin 500 mg: BD with meals\n- Glimepiride 1 mg: OD before breakfast (if needed)",
        "Alternative": "- DPP-4 inhibitors (Sitagliptin) if metformin not tolerated\n- Insulin therapy for uncontrolled cases",
        "Lifestyle": "Low glycemic diet, regular exercise (30 min daily), monitor blood glucose, weight management.",
        "Red Flags": "Glucose >400 mg/dL, confusion, chest pain, excessive thirst, diabetic ketoacidosis signs.",
        "Follow-Up": "HbA1c every 3 months, fasting glucose weekly, regular eye and foot checks."
    },
    "cold": {
        "Medicine": "- Cetirizine 10 mg: Once daily for runny nose\n- Paracetamol 500 mg: For fever/body ache\n- Cough syrup: As needed",
        "Alternative": "- Steam inhalation 3x daily\n- Ginger-honey tea\n- Saline nasal drops",
        "Lifestyle": "Rest, hydration, warm fluids, avoid cold beverages, vitamin C rich foods.",
        "Red Flags": "High fever, chest pain, difficulty breathing, symptoms >7 days.",
        "Follow-Up": "Review if symptoms persist beyond 5 days or worsen."
    },
    "headache": {
        "Medicine": "- Paracetamol 500 mg: Every 6-8 hours as needed\n- For migraine: Sumatriptan 50 mg at onset",
        "Alternative": "- Ibuprofen 400 mg\n- Cold/warm compress\n- Rest in dark quiet room",
        "Lifestyle": "Stress management, regular sleep schedule, hydration, avoid triggers (caffeine, bright lights).",
        "Red Flags": "Sudden severe headache (thunderclap), vision changes, confusion, neck stiffness, seizures.",
        "Follow-Up": "Review if headaches increase in frequency or severity. Consider CT/MRI if red flags present."
    },
    "hypertension": {
        "Medicine": "- Amlodipine 5 mg: Once daily\n- Losartan 50 mg: Once daily (if needed)\n- Aspirin 75 mg: For cardiovascular protection",
        "Alternative": "- Beta-blockers (Metoprolol) if tachycardia present\n- Diuretics (Hydrochlorothiazide) for fluid retention",
        "Lifestyle": "Low salt diet (<5g/day), DASH diet, regular exercise, stress reduction, weight loss, limit alcohol.",
        "Red Flags": "BP >180/120, severe headache, chest pain, shortness of breath, visual changes.",
        "Follow-Up": "BP monitoring weekly, review medication in 4 weeks, annual cardiac assessment."
    },
    "asthma": {
        "Medicine": "- Salbutamol inhaler: 2 puffs PRN for breathlessness\n- Beclomethasone inhaler: 2 puffs BD (controller)",
        "Alternative": "- Montelukast 10 mg: Once daily\n- Nebulization with salbutamol in acute attack",
        "Lifestyle": "Avoid triggers (dust, smoke, allergens), regular inhaler use, breathing exercises.",
        "Red Flags": "Severe breathlessness, unable to speak, blue lips, no response to inhaler.",
        "Follow-Up": "Peak flow monitoring, review in 2 weeks, pulmonary function tests annually."
    },
    "gastric": {
        "Medicine": "- Pantoprazole 40 mg: Once daily before breakfast\n- Antacids: As needed for immediate relief",
        "Alternative": "- Ranitidine 150 mg: BD if PPI not tolerated\n- Sucralfate for mucosal protection",
        "Lifestyle": "Small frequent meals, avoid spicy/oily foods, no late night meals, reduce stress, avoid NSAIDs.",
        "Red Flags": "Severe abdominal pain, vomiting blood, black stools, unexplained weight loss.",
        "Follow-Up": "Review in 4 weeks. Consider endoscopy if symptoms persist or red flags present."
    },
    "allergy": {
        "Medicine": "- Cetirizine 10 mg: Once daily\n- Hydrocortisone cream: For skin rash BD\n- In severe reaction: Epinephrine auto-injector",
        "Alternative": "- Loratadine 10 mg if cetirizine causes drowsiness\n- Calamine lotion for itching",
        "Lifestyle": "Identify and avoid allergen, keep antihistamines handy, wear medical alert bracelet if severe.",
        "Red Flags": "Swelling of face/tongue, difficulty breathing, rapid pulse, anaphylaxis.",
        "Follow-Up": "Allergy testing if recurrent. Emergency plan for severe allergies."
    },
    "arthritis": {
        "Medicine": "- Ibuprofen 400 mg: TDS with food\n- Glucosamine supplements: For joint health",
        "Alternative": "- Paracetamol for mild pain\n- Topical NSAIDs (Diclofenac gel)",
        "Lifestyle": "Regular low-impact exercise (swimming, walking), weight management, physiotherapy, hot/cold therapy.",
        "Red Flags": "Severe joint swelling, redness, warmth, inability to move joint, fever.",
        "Follow-Up": "Review in 4-6 weeks. X-ray if severe. Consider rheumatologist referral."
    },
    "mental_health": {
        "Medicine": "- For anxiety: Escitalopram 10 mg OD or Alprazolam 0.25 mg SOS\n- For depression: Sertraline 50 mg OD",
        "Alternative": "- Cognitive Behavioral Therapy (CBT)\n- Mindfulness and meditation\n- Support groups",
        "Lifestyle": "Regular exercise, good sleep hygiene, social support, stress management, avoid alcohol/drugs.",
        "Red Flags": "Suicidal thoughts, self-harm, severe agitation, psychotic symptoms.",
        "Follow-Up": "Weekly counseling sessions, medication review in 2 weeks. Psychiatric referral if severe."
    },
    "general": {
        "Medicine": "- Symptomatic treatment based on presentation\n- Paracetamol 500 mg: As needed for pain/fever",
        "Alternative": "- Rest and hydration\n- Monitor symptoms",
        "Lifestyle": "Healthy diet, adequate rest, hydration, stress management.",
        "Red Flags": "Persistent symptoms >3 days, worsening condition, new concerning symptoms.",
        "Follow-Up": "Review in 48-72 hours if no improvement."
    }
}

print("="*60)
print("Med4Me ML Model Training Script")
print("="*60)

# Create DataFrame
df = pd.DataFrame(training_data)
print(f"\n✓ Loaded {len(df)} training samples")

# Feature engineering
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['child', 'young_adult', 'middle_age', 'senior', 'elderly'])
df['gender_encoded'] = df['gender'].map({'male': 0, 'female': 1})

# Prepare features
X_text = df['symptoms']
X_age = df['age'].values.reshape(-1, 1)
X_gender = df['gender_encoded'].values.reshape(-1, 1)
y = df['category']

# Text vectorization
print("\n✓ Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X_text_vectors = vectorizer.fit_transform(X_text)

# Combine features
X_combined = np.hstack([
    X_text_vectors.toarray(),
    X_age,
    X_gender
])

print(f"✓ Feature matrix shape: {X_combined.shape}")

# Train model
print("\n✓ Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_combined, y)

# Calculate training accuracy
train_accuracy = model.score(X_combined, y)
print(f"✓ Training accuracy: {train_accuracy*100:.2f}%")

# Feature importance
feature_names = vectorizer.get_feature_names_out().tolist() + ['age', 'gender']
importances = model.feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
print("\n✓ Top 10 important features:")
for feat, imp in top_features:
    print(f"  - {feat}: {imp:.4f}")

# Save model and vectorizer
print("\n✓ Saving model files...")
with open('ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('treatment_db.json', 'w') as f:
    json.dump(treatment_db, f, indent=2)

print("\n✓ Model saved as: ml_model.pkl")
print("✓ Vectorizer saved as: vectorizer.pkl")
print("✓ Treatment database saved as: treatment_db.json")

# Test prediction
print("\n" + "="*60)
print("Testing Model with Sample Cases")
print("="*60)

test_cases = [
    {"symptoms": "high fever and body ache", "age": 30, "gender": "male"},
    {"symptoms": "frequent urination and excessive thirst", "age": 55, "gender": "female"},
    {"symptoms": "severe headache with nausea", "age": 35, "gender": "female"},
]

for i, test in enumerate(test_cases, 1):
    # Prepare test features
    test_vector = vectorizer.transform([test['symptoms']])
    test_age = np.array([[test['age']]])
    test_gender = np.array([[0 if test['gender'] == 'male' else 1]])
    test_combined = np.hstack([test_vector.toarray(), test_age, test_gender])
    
    # Predict
    prediction = model.predict(test_combined)[0]
    probabilities = model.predict_proba(test_combined)[0]
    confidence = max(probabilities) * 100
    
    print(f"\nTest Case {i}:")
    print(f"  Symptoms: {test['symptoms']}")
    print(f"  Age: {test['age']}, Gender: {test['gender']}")
    print(f"  Predicted: {prediction} (Confidence: {confidence:.1f}%)")

print("\n" + "="*60)
print("✓ Model training completed successfully!")
print("="*60)
print("\nNext step: Run the Flask app with 'python app.py'")
print("The app will automatically use the ML model.")