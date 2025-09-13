import os
import re
import io
import json
import math
import cv2
import numpy as np
import datetime
import base64
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Load secrets and config from environment variables
MONGO_URI = os.getenv('MONGO_URI')
SECRET_KEY = os.getenv('SECRET_KEY')
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')

# Set Flask secret key
app.secret_key = SECRET_KEY

# Initialize OAuth for Google login
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    client_kwargs={'scope': 'openid email profile'}
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Connect to MongoDB using MONGO_URI from env
client = MongoClient(MONGO_URI)
db = client['face_attendance']
users_col = db['users']
attendance_col = db['attendance']

# Authorized location (example)
AUTHORIZED_LAT = float(os.getenv('AUTHORIZED_LAT', '12.9716'))
AUTHORIZED_LON = float(os.getenv('AUTHORIZED_LON', '77.5946'))
ALLOWED_RADIUS = int(os.getenv('ALLOWED_RADIUS', '100'))

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_images_and_labels():
    image_paths = []
    labels = []
    label_map = {}
    for i, user in enumerate(users_col.find()):
        user_id = str(user['_id'])
        user_folder = os.path.join(DATASET_FOLDER, user_id)
        if os.path.exists(user_folder):
            label_map[i] = user['username']
            for filename in os.listdir(user_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(user_folder, filename))
                    labels.append(i)
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images, labels, label_map

def train_model():
    images, labels, label_map = get_images_and_labels()
    if len(images) == 0:
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer, label_map

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if not email.lower().endswith("@gmail.com"):
            return render_template('login.html', error="Only @gmail.com addresses allowed")
        user = users_col.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html', error=None)

@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize_google', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize/google')
def authorize_google():
    token = google.authorize_access_token()
    userinfo = google.parse_id_token(token)
    email = userinfo.get('email', '')
    if not email.lower().endswith('@gmail.com'):
        return "Only @gmail.com addresses allowed for login."
    user = users_col.find_one({'email': email})
    if user:
        session['user_id'] = str(user['_id'])
        return redirect(url_for('dashboard'))
    user_id = users_col.insert_one({
        'email': email,
        'username': userinfo.get('name', 'Google User'),
        'password': '',
        'google': True
    }).inserted_id
    session['user_id'] = str(user_id)
    return redirect(url_for('dashboard'))

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html', error=None)

@app.route('/register-cam', methods=['POST'])
def register_cam():
    email = request.form['email']
    username = request.form['username']
    password = generate_password_hash(request.form['password'])
    face_images_json = request.form['face_images']
    if not email.lower().endswith("@gmail.com"):
        return render_template('register.html', error="Only @gmail.com addresses allowed")
    if users_col.find_one({'email': email}):
        return render_template('register.html', error="Email already exists")
    if not face_images_json:
        return render_template('register.html', error="No face images captured")
    try:
        face_images = json.loads(face_images_json)
    except:
        return render_template('register.html', error="Invalid face images data")
    user_id = users_col.insert_one({'email': email, 'username': username, 'password': password}).inserted_id
    user_folder = os.path.join(DATASET_FOLDER, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    for idx, data_url in enumerate(face_images):
        img_str = re.search(r'base64,(.*)', data_url).group(1)
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        filepath = os.path.join(user_folder, f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{idx}.png')
        img.save(filepath)
    train_model()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = users_col.find_one({'_id': ObjectId(session['user_id'])})
    username = user['username'] if user else 'User'
    return render_template('dashboard.html', username=username)

@app.route('/attendance', methods=['GET'])
def attendance():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('attendance.html')

@app.route('/attendance-cam', methods=['POST'])
def attendance_cam():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    face_data_url = request.form.get('face_image')
    latitude = request.form.get('latitude', type=float)
    longitude = request.form.get('longitude', type=float)
    if not face_data_url:
        return render_template('attendance.html', error="No face image provided")
    if latitude is None or longitude is None:
        return render_template('attendance.html', error="Location data missing. Please allow location access.")
    distance = haversine(latitude, longitude, AUTHORIZED_LAT, AUTHORIZED_LON)
    if distance > ALLOWED_RADIUS:
        return render_template('attendance.html', error=f"You are far away from the authorized location ({distance:.1f} meters). Attendance not marked.")
    img_str = re.search(r'base64,(.*)', face_data_url).group(1)
    img_bytes = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    recognizer, label_map = train_model()
    if recognizer is None:
        return render_template('attendance.html', error="No users registered yet")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return render_template('attendance.html', error="No face detected")
    attendance_marked = []
    threshold = 90
    users_list = list(users_col.find())
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(roi_gray)
        if confidence < threshold:
            username = label_map.get(label)
            if username:
                attendance_col.insert_one({
                    'user_id': session['user_id'],
                    'username': username,
                    'date': datetime.datetime.now(),
                    'latitude': latitude,
                    'longitude': longitude
                })
                attendance_marked.append(username)
    if attendance_marked:
        return render_template('attendance.html', message=f"Attendance marked for: {', '.join(attendance_marked)}")
    else:
        return render_template('attendance.html', error="No matching face recognized")

@app.route('/attendance-records')
def attendance_records():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    records = list(attendance_col.find({'user_id': user_id}).sort('date', -1))
    for r in records:
        r['date_str'] = r['date'].strftime('%Y-%m-%d %H:%M:%S') if 'date' in r else "N/A"
    return render_template('attendance_records.html', records=records)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    user = users_col.find_one({'_id': ObjectId(user_id)})
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            users_col.update_one({'_id': ObjectId(user_id)}, {'$set': {'username': username}})
            user = users_col.find_one({'_id': ObjectId(user_id)})
    return render_template('profile.html', user=user)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
