from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import re, json, sqlite3, os, pickle
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

# config
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret_for_prod")
basedir = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(basedir, 'med4me.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------- Load ML Model ----------
ML_MODEL = None
VECTORIZER = None
TREATMENT_DB = {}
USE_ML = False

def load_ml_model():
    global ML_MODEL, VECTORIZER, TREATMENT_DB, USE_ML
    try:
        model_path = os.path.join(basedir, 'ml_model.pkl')
        vectorizer_path = os.path.join(basedir, 'vectorizer.pkl')
        treatment_path = os.path.join(basedir, 'treatment_db.json')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as f:
                ML_MODEL = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                VECTORIZER = pickle.load(f)
            if os.path.exists(treatment_path):
                with open(treatment_path, 'r') as f:
                    TREATMENT_DB = json.load(f)
            USE_ML = True
            print("✓ ML Model loaded successfully")
            return True
    except Exception as e:
        print(f"⚠ ML Model not found: {e}")
        print("⚠ Using rule-based fallback system")
    return False

# ---------- Models ----------
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class DoctorPatient(db.Model):
    __tablename__ = 'doctor_patient'
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_id = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('doctor_id', 'patient_id', name='_doctor_patient_uc'),)
    doctor = db.relationship('User', backref=db.backref('doctor_patients', lazy=True))

class Visit(db.Model):
    __tablename__ = 'visit'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(120), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    symptoms = db.Column(db.Text)
    age = db.Column(db.String(50))
    gender = db.Column(db.String(50))
    genetic_history = db.Column(db.Text)
    medicine = db.Column(db.Text)
    diagnosis = db.Column(db.Text)
    lifestyle = db.Column(db.Text)
    follow_up = db.Column(db.Text)
    ml_prediction = db.Column(db.String(100))
    ml_confidence = db.Column(db.Float)
    creator = db.relationship('User', backref=db.backref('visits', lazy=True))

# ---------- ML Prediction Function ----------
def ml_recommendation(symptoms, age, gender, genetic_history=None):
    """Use ML model for prediction if available, otherwise fallback to rules"""
    
    if USE_ML and ML_MODEL and VECTORIZER:
        try:
            # Prepare features
            text_vector = VECTORIZER.transform([symptoms.lower()])
            age_val = int(age) if str(age).isdigit() else 30
            gender_val = 0 if gender.lower() in ['male', 'm'] else 1
            
            # Combine features
            X = np.hstack([
                text_vector.toarray(),
                np.array([[age_val]]),
                np.array([[gender_val]])
            ])
            
            # Predict
            prediction = ML_MODEL.predict(X)[0]
            probabilities = ML_MODEL.predict_proba(X)[0]
            confidence = float(max(probabilities))
            
            # Get treatment from database
            treatment = TREATMENT_DB.get(prediction, TREATMENT_DB.get('general', {}))
            
            # Map prediction to diagnosis
            diagnosis_map = {
                'fever': 'Acute Febrile Illness',
                'diabetes': 'Type 2 Diabetes Mellitus',
                'cold': 'Upper Respiratory Tract Infection (URTI)',
                'headache': 'Tension Headache / Migraine',
                'hypertension': 'Hypertension',
                'asthma': 'Asthma',
                'gastric': 'Gastritis',
                'allergy': 'Allergic Reaction',
                'arthritis': 'Osteoarthritis',
                'mental_health': 'Anxiety/Depression - Requires Specialist',
                'general': 'General Symptomatic Care'
            }
            
            diagnosis = diagnosis_map.get(prediction, 'Condition Requiring Further Assessment')
            
            rec = {
                "Diagnosis": diagnosis,
                "Medicine": treatment.get("Medicine", "Symptomatic treatment recommended"),
                "Alternative": treatment.get("Alternative", "Consult specialist for alternatives"),
                "Lifestyle": treatment.get("Lifestyle", "Healthy lifestyle, adequate rest"),
                "Red Flags": treatment.get("Red Flags", "Worsening symptoms, no improvement in 3 days"),
                "Follow-Up": treatment.get("Follow-Up", "Review in 48-72 hours"),
                "Notes": f"ML Model Prediction: {prediction} (Confidence: {confidence*100:.1f}%). This is an AI-assisted recommendation. Final prescription authority lies with the licensed physician.",
                "ml_prediction": prediction,
                "ml_confidence": confidence
            }
            
            return rec
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            # Fall through to rule-based system
    
    # Fallback to rule-based system
    return fallback_recommendation(symptoms, age, gender, genetic_history)

# ---------- Rule-based Fallback ----------
def fallback_recommendation(symptoms, age, gender, genetic_history=None):
    s = (symptoms or "").lower()
    notes_suffix = "This is a rule-based recommendation. Final prescription authority lies with the licensed physician."

    def bullet(line):
        return "- " + line + "\n"

    rec = {
        "Diagnosis": "General symptomatic care",
        "Medicine": bullet("Paracetamol 500 mg: Every 6 hours as needed"),
        "Alternative": bullet("Ibuprofen 200 mg: Every 6-8 hours if no contraindications"),
        "Lifestyle": "Hydration, rest, monitor symptoms.",
        "Red Flags": "Persistent symptoms more than 3 days, high fever, severe pain.",
        "Follow-Up": "Review in 48-72 hours if no improvement.",
        "Notes": notes_suffix,
        "ml_prediction": None,
        "ml_confidence": None
    }

    if re.search(r"\bfever\b|\btemperature\b|\bpyrexia\b", s):
        rec.update({
            "Diagnosis": "Acute febrile illness",
            "Medicine": bullet("Paracetamol 500 mg: Every 6 hours for fever"),
            "Alternative": bullet("Ibuprofen 400 mg: Every 8 hours if no contraindications"),
            "Lifestyle": "Fluids, rest, monitor temperature.",
            "Red Flags": "Fever more than 3 days, severe headache, rash, breathing difficulty.",
            "Follow-Up": "Review in 48 hours if fever persists.",
            "Notes": notes_suffix
        })
        return rec

    if re.search(r"\bdiabetes\b|\bhigh.*sugar\b|\bhyperglycemi\b", s):
        rec.update({
            "Diagnosis": "Type 2 Diabetes Mellitus",
            "Medicine": bullet("Metformin 500 mg: BD with meals"),
            "Alternative": bullet("DPP-4 inhibitors if metformin not tolerated"),
            "Lifestyle": "Diet control, exercise, glucose monitoring, weight loss.",
            "Red Flags": "Glucose more than 400, confusion, chest pain, excessive thirst.",
            "Follow-Up": "HbA1c every 3 months.",
            "Notes": notes_suffix
        })
        return rec

    if re.search(r"\bcough\b|\bcold\b|\bsneez\b", s):
        rec.update({
            "Diagnosis": "Upper Respiratory Tract Infection (URTI)",
            "Medicine": bullet("Antihistamine (Cetirizine 10 mg): Once daily") + bullet("Cough syrup: As needed"),
            "Alternative": bullet("Steam inhalation") + bullet("Loratadine 10 mg if drowsiness is a concern"),
            "Lifestyle": "Rest, hydration, avoid cold beverages.",
            "Red Flags": "High fever, chest pain, difficulty breathing.",
            "Follow-Up": "Review if symptoms persist beyond 5 days.",
            "Notes": notes_suffix
        })
        return rec

    if re.search(r"\bheadache\b|\bmigrain\b", s):
        rec.update({
            "Diagnosis": "Tension headache / Migraine",
            "Medicine": bullet("Paracetamol 500 mg: Every 6-8 hours") + bullet("For migraine: Sumatriptan 50 mg as needed"),
            "Alternative": bullet("Ibuprofen 400 mg") + bullet("Rest in dark room"),
            "Lifestyle": "Stress management, regular sleep, hydration.",
            "Red Flags": "Sudden severe headache, vision changes, confusion, neck stiffness.",
            "Follow-Up": "Review if headaches increase in frequency.",
            "Notes": notes_suffix
        })
        return rec

    return rec

# ---------- DB initialization ----------
def ensure_db_and_migrate():
    with app.app_context():
        db.create_all()
        
        if not os.path.exists(DB_PATH):
            return

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(visit);")
            cols = cur.fetchall()
            col_names = [row[1] for row in cols]
            
            if 'doctor_id' not in col_names:
                cur.execute("ALTER TABLE visit ADD COLUMN doctor_id INTEGER;")
                conn.commit()
            if 'ml_prediction' not in col_names:
                cur.execute("ALTER TABLE visit ADD COLUMN ml_prediction TEXT;")
                conn.commit()
            if 'ml_confidence' not in col_names:
                cur.execute("ALTER TABLE visit ADD COLUMN ml_confidence REAL;")
                conn.commit()
            
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[migration] warning: {e}")

    with app.app_context():
        admin = User.query.filter_by(username="admin").first()
        if not admin:
            admin = User(username="admin")
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()
            print("✓ Default admin user created")

ensure_db_and_migrate()
load_ml_model()

# ---------- Auth endpoints ----------
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.json or {}
        username = (data.get('username') or "").strip()
        password = data.get('password') or ""
        if not username or not password:
            return jsonify({'success': False, 'message': 'username and password required'}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 409
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User registered', 'username': user.username, 'user_id': user.id}), 201
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json or {}
        username = (data.get('username') or "").strip()
        password = data.get('password') or ""
        if not username or not password:
            return jsonify({'success': False, 'message': 'username and password required'}), 400

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            return jsonify({'success': True, 'message': 'Login successful', 'username': username, 'user_id': user.id})
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/my/patients', methods=['GET', 'POST'])
def my_patients():
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            if not user_id:
                return jsonify({'success': False, 'message': 'user_id required'}), 400
            uid = int(user_id)
            doctor = User.query.get(uid)
            if not doctor:
                return jsonify({'success': False, 'message': 'Doctor not found'}), 404
            mappings = DoctorPatient.query.filter_by(doctor_id=uid).order_by(DoctorPatient.created_at.desc()).all()
            out = []
            for mp in mappings:
                last_visit = Visit.query.filter_by(doctor_id=uid, patient_id=mp.patient_id).order_by(Visit.date.desc()).first()
                out.append({
                    "patient_id": mp.patient_id,
                    "first_seen": mp.created_at.strftime("%Y-%m-%d %H:%M"),
                    "last_visit": last_visit.date.strftime("%Y-%m-%d %H:%M") if last_visit else None,
                    "last_symptoms": last_visit.symptoms if last_visit else None,
                    "visits_count": Visit.query.filter_by(doctor_id=uid, patient_id=mp.patient_id).count()
                })
            return jsonify({'success': True, 'doctor_id': uid, 'patients': out, 'total': len(out)})
        else:
            data = request.json or {}
            user_id = data.get('user_id')
            patient_id = data.get('patient_id')
            if not user_id or not patient_id:
                return jsonify({'success': False, 'message': 'user_id and patient_id required'}), 400
            uid = int(user_id)
            doctor = User.query.get(uid)
            if not doctor:
                return jsonify({'success': False, 'message': 'Doctor not found'}), 404
            exists = DoctorPatient.query.filter_by(doctor_id=uid, patient_id=patient_id).first()
            if exists:
                return jsonify({'success': True, 'message': 'Mapping exists', 'patient_id': patient_id})
            mapping = DoctorPatient(doctor_id=uid, patient_id=patient_id)
            db.session.add(mapping)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Mapping created', 'patient_id': patient_id})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/patients/<patient_id>/history', methods=['GET'])
def get_history(patient_id):
    try:
        doctor_id = request.args.get('doctor_id')
        query = Visit.query.filter_by(patient_id=patient_id)
        if doctor_id:
            try:
                did = int(doctor_id)
                query = query.filter_by(doctor_id=did)
            except:
                pass
        visits = query.order_by(Visit.date.asc()).all()
        out = []
        for v in visits:
            out.append({
                "id": v.id,
                "date": v.date.strftime("%Y-%m-%d %H:%M"),
                "symptoms": v.symptoms,
                "age": v.age,
                "gender": v.gender,
                "genetic_history": v.genetic_history,
                "medicine": v.medicine,
                "diagnosis": v.diagnosis,
                "lifestyle": v.lifestyle,
                "follow_up": v.follow_up,
                "created_by": v.creator.username if v.creator else None,
                "doctor_id": v.doctor_id,
                "ml_prediction": v.ml_prediction,
                "ml_confidence": v.ml_confidence
            })
        return jsonify({'success': True, 'patient_id': patient_id, 'history': out, 'total_visits': len(out)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json or {}
        patient_id = str(data.get('patient_id', '')).strip()
        symptoms = data.get('symptoms')
        age = data.get('age')
        gender = data.get('gender')
        genetic_history = data.get('genetic_history', "")
        creator = data.get('created_by')

        if not all([patient_id, symptoms, age, gender]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400

        doctor = None
        if creator is not None:
            try:
                maybe = int(creator)
                doctor = User.query.get(maybe)
            except:
                doctor = User.query.filter_by(username=str(creator)).first()

        # Use ML model for recommendation
        recommendation = ml_recommendation(symptoms, age, gender, genetic_history)
        
        keys = ["Diagnosis", "Medicine", "Alternative", "Lifestyle", "Red Flags", "Follow-Up", "Notes"]
        normalized = {k: recommendation.get(k, "") for k in keys}
        
        doctor_id = doctor.id if doctor else None
        if doctor_id:
            mp = DoctorPatient.query.filter_by(doctor_id=doctor_id, patient_id=patient_id).first()
            if not mp:
                mp = DoctorPatient(doctor_id=doctor_id, patient_id=patient_id)
                db.session.add(mp)
                db.session.commit()

        visit = Visit(
            patient_id=patient_id,
            doctor_id=doctor_id,
            date=datetime.now(),
            symptoms=symptoms,
            age=str(age),
            gender=gender,
            genetic_history=genetic_history,
            medicine=normalized.get("Medicine", ""),
            diagnosis=normalized.get("Diagnosis", ""),
            lifestyle=normalized.get("Lifestyle", ""),
            follow_up=normalized.get("Follow-Up", ""),
            ml_prediction=recommendation.get("ml_prediction"),
            ml_confidence=recommendation.get("ml_confidence")
        )
        if doctor:
            visit.creator = doctor

        db.session.add(visit)
        db.session.commit()

        visit_date = visit.date.strftime("%Y-%m-%d %H:%M")
        response_visit = {
            "id": visit.id,
            "date": visit_date,
            "symptoms": visit.symptoms,
            "age": visit.age,
            "gender": visit.gender,
            "genetic_history": visit.genetic_history,
            "medicine": visit.medicine,
            "diagnosis": visit.diagnosis,
            "lifestyle": visit.lifestyle,
            "follow_up": visit.follow_up,
            "created_by": doctor.username if doctor else None,
            "doctor_id": doctor_id,
            "ml_prediction": visit.ml_prediction,
            "ml_confidence": visit.ml_confidence
        }

        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'recommendation': normalized,
            'timestamp': visit_date,
            'visit': response_visit,
            'using_ml': USE_ML
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/patients', methods=['GET'])
def list_patients():
    try:
        patients = db.session.query(Visit.patient_id).distinct().all()
        out = [p[0] for p in patients]
        return jsonify({"success": True, "patients": out, "total_patients": len(out)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Med4Me Backend with ML",
        "version": "8.0.0-ML",
        "ml_enabled": USE_ML,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

if __name__ == "__main__":
    print("="*60)
    print("Med4Me Backend v8.0 - ML Enhanced")
    print("="*60)
    print("ML Model Status:", "✓ ENABLED" if USE_ML else "✗ Using Rule-based Fallback")
    print("DB file:", DB_PATH)
    print("Starting on http://0.0.0.0:5000")
    print("="*60)
    app.run(debug=True, host="0.0.0.0", port=5000)