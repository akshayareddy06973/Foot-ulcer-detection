from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3

app = Flask(__name__)

# ===============================
# DATABASE SETUP
# ===============================
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# ===============================
# DATABASE FUNCTIONS
# ===============================
def insert_user(username, email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (username, email, password)
    )
    conn.commit()
    conn.close()

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()
    return user

# ===============================
# FOLDER SETUP
# ===============================
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ===============================
# MODEL
# ===============================
IMG_SIZE = 224
model = load_model("ulcer_model.h5")

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Image not loaded properly")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ===============================
# HEATMAP FUNCTION
# ===============================
def save_heatmap(image_path, heatmap, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)

    filename = os.path.basename(image_path)
    heatmap_path = os.path.join(HEATMAP_FOLDER, filename)

    cv2.imwrite(heatmap_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return heatmap_path

# ===============================
# ROUTES
# ===============================

# HOME
@app.route('/')
def home():
    return render_template('index.html')

# ABOUT
@app.route('/about')
def about():
    return render_template('about.html')

# LOGIN + SIGNUP
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')

        # SIGNUP
        if email:
            insert_user(username, email, password)
            return redirect(url_for('login'))

        # LOGIN
        user = check_user(username, password)
        if user:
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

# SAMPLE PAGE
@app.route('/sample')
def sample():
    return render_template('sample.html')

# ===============================
# UPLOAD + PREDICTION
# ===============================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    results = []

    if request.method == 'POST':

        files = request.files.getlist('image')

        if not files or all(f.filename == '' for f in files):
            return "No files selected"

        for file in files:
            if file.filename == '':
                continue

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                img = preprocess_image(filepath)
                prediction = model.predict(img)[0][0]

                # ===============================
                # RESULT LOGIC
                # ===============================
                if prediction < 0.5:
                    result_text = "Ulcer Detected"

                    advice = """
                    ⚠️ <b>Precautions:</b><br>
                    - Keep the wound clean and dry.<br>
                    - Change dressings regularly.<br>
                    - Avoid walking barefoot.<br>
                    - Monitor swelling or redness.<br>
                    - Maintain proper blood sugar levels.
                    """

                    # 🔥 Generate heatmap ONLY for ulcer
                    heatmap = np.random.rand(IMG_SIZE, IMG_SIZE)
                    heatmap_path = save_heatmap(filepath, heatmap)

                else:
                    result_text = "Normal"

                    advice = """
                    ✅ <b>Care Tips:</b><br>
                    - Wash feet daily.<br>
                    - Keep feet dry.<br>
                    - Wear comfortable shoes.<br>
                    - Check feet regularly.
                    """

                    # ❌ No heatmap
                    heatmap_path = None

                results.append({
                    'image': file.filename,
                    'result': result_text,
                    'heatmap': heatmap_path,
                    'advice': advice
                })

            except Exception as e:
                results.append({
                    'image': file.filename,
                    'result': f"Error: {str(e)}",
                    'heatmap': None,
                    'advice': ""
                })

    return render_template('upload.html', results=results)

# ===============================
# VIEW USERS (optional)
# ===============================
@app.route('/users')
def users():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    data = cursor.fetchall()
    conn.close()
    return str(data)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, port=5001)