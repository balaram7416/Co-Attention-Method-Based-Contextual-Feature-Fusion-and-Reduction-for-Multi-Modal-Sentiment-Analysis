import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from model import ContextualFusionCoAttention
from preprocess import preprocess_text, preprocess_audio, preprocess_video

# ✅ Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")

# ✅ Secret key for session management
app.secret_key = "supersecretkey"  # Change this for production

# ✅ Database Configuration (SQLite)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ✅ User Model (Database Table)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ✅ Constants for Model
TEXT_DIM = 768
AUDIO_DIM = 40
VIDEO_DIM = 512
HIDDEN_DIM = 128
MODEL_PATH = "C:/Users/HP/Downloads/dataset/contextual_fusion_best.pth"

# ✅ Set Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Pretrained Model
try:
    print("\n🔹 Initializing Model...")
    model = ContextualFusionCoAttention(TEXT_DIM, AUDIO_DIM, VIDEO_DIM, HIDDEN_DIM)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
    model.to(device)
    model.eval()
    print("\n✅ Model loaded successfully.")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    sys.exit(1)

# ✅ Home Page
@app.route("/")
def home():
    user_name = session.get("user_name")
    return render_template("home.html", user_name=user_name)

# ✅ About Page
@app.route("/about")
def about():
    return render_template("about.html")

# ✅ Contact Page
@app.route("/contact")
def contact():
    return render_template("contact.html")

# ✅ Contact Form Submission
@app.route("/contact_submit", methods=["POST"])
def contact_submit():
    name = request.form["name"]
    email = request.form["email"]
    message = request.form["message"]
    
    print(f"📩 Received message from {name} ({email}): {message}")
    flash("Message sent successfully!", "success")
    return redirect(url_for("contact"))

# ✅ Signup Page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match! Please try again.", "error")
            return redirect(url_for("signup"))

        # Check if email is already registered
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Try logging in.", "error")
            return redirect(url_for("signup"))

        # Hash the password for security
        hashed_password = generate_password_hash(password)

        # Save user data in the database
        new_user = User(name=username, email=email, phone=phone, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! Please log in.", "success")
        return redirect(url_for("login"))  

    return render_template("signup.html")

# ✅ Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id  
            session["user_name"] = user.name  
            flash(f"Welcome {user.name} to My Emotion AI!", "success")
            return redirect(url_for("home"))  
        else:
            flash("Invalid email or password!", "danger")

    return render_template("login.html")

# ✅ Logout Route
@app.route("/logout")
def logout():
    session.pop("user_id", None)  
    session.pop("user_name", None)  
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))

# ✅ Forgot Password Page
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        user = User.query.filter_by(email=email).first()
        
        if user:
            flash("Password reset link sent to your email!", "success")
        else:
            flash("Email not found. Try again!", "danger")
    
    return render_template("forgot.html")

# ✅ Prediction Page
@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# ✅ API Endpoint for Predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("\n🔹 Received request to /predict")
        
        text_file = request.files.get("text")
        audio_file = request.files.get("audio")
        video_file = request.files.get("video")
        
        if not text_file or not audio_file or not video_file:
            return jsonify({"error": "Missing input files"}), 400
        
        text_features = preprocess_text(text_file)
        audio_features = preprocess_audio(audio_file)
        video_features = preprocess_video(video_file)
        
        text_tensor = torch.tensor(text_features, dtype=torch.float32).view(1, -1).to(device)
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32).view(1, -1).to(device)
        video_tensor = torch.tensor(video_features, dtype=torch.float32).view(1, -1).to(device)

        # ✅ Debug Tensor Shapes
        print(f"Text Tensor Shape: {text_tensor.shape}")
        print(f"Audio Tensor Shape: {audio_tensor.shape}")
        print(f"Video Tensor Shape: {video_tensor.shape}")

        # ✅ Run Prediction
        with torch.no_grad():
            prediction = model(text_tensor, audio_tensor, video_tensor)
        
        sentiment = torch.argmax(prediction, dim=1).item()
        sentiment_label = ["Negative", "Neutral", "Positive", "Strong Negative", "Strong Positive"][sentiment]
        
        print(f"✅ Prediction: {sentiment_label}")
        return jsonify({"sentiment": sentiment_label})

    except Exception as e:
        print(f"\n❌ Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ 404 Error Handling
@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404

# ✅ Run Flask Server
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables if they don’t exist
    app.run(debug=True)
