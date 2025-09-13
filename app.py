from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
import json
from PIL import Image
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================
# Load Crop Recommendation Model
# ==========================

model = pickle.load(open("model.pkl", "rb"))
ms = pickle.load(open("minmaxscaler.pkl", "rb"))

CROP_DICT = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee",
}


def humanize_label(raw):
    """Converts raw class names like 'Apple___Apple_scab' or 'Apple__Healthy'
    into readable labels like 'Apple - Apple Scab'."""
    text = raw.replace(",", "")
    text = text.replace("___", " - ").replace("__", " - ")
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text.title()


# ---------------- Disease Detection Model ---------------- #

interpreter = tf.lite.Interpreter(
    model_path="models/plant_disease_model.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("disease_data.json", "r") as f:
    disease_info = json.load(f)

disease_classes = list(disease_info.keys())


# ---------------- Fruit Identification Model ---------------- #

fruit_interpreter = tf.lite.Interpreter(
    model_path="models/fruit_vegetable_classifier_quantized.tflite"
)
fruit_interpreter.allocate_tensors()

fruit_input_details = fruit_interpreter.get_input_details()
fruit_output_details = fruit_interpreter.get_output_details()

with open("labels.txt", "r") as f:
    fruit_labels = [line.strip() for line in f if line.strip()]

with open("fruit_data.json", "r") as f:
    fruit_data = json.load(f)

# Known spelling mismatches between labels.txt and fruit_data.json.
# labels.txt is left untouched since its order must match the trained model exactly.
FRUIT_NAME_ALIASES = {
    "raddish": "radish",
}


# ---------------- Health Classifier Model ---------------- #

health_interpreter = tf.lite.Interpreter(
    model_path="models/fruit_vegetable_healty_classifier.tflite"
)
health_interpreter.allocate_tensors()

health_input_details = health_interpreter.get_input_details()
health_output_details = health_interpreter.get_output_details()

with open("class_names.txt", "r") as f:
    health_labels = [line.strip() for line in f if line.strip()]


# ---------------- Gemini Chatbot ---------------- #

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini Chatbot Configured")
    except Exception as e:
        print(f"⚠️  Gemini setup failed: {e}")
else:
    print("⚠️  GEMINI_API_KEY not set — chatbot disabled")

CHATBOT_SYSTEM_INSTRUCTION = """You are AgroBot, the assistant for AgrowAssist, an AI agriculture platform.

Only answer questions about: farming, crops, soil, plant diseases, produce, nutrition, and how to use AgrowAssist's tools (crop recommendation, disease detection, fruit identification, health classifier, analytics).

If asked something unrelated to agriculture, politely say you're focused on farming topics and redirect back.

When relevant, point users to the right tool:
- Soil/climate questions -> suggest /recommendation
- A sick-looking plant -> suggest /disease
- Identifying produce -> suggest /identify
- Checking if produce is spoiled -> suggest /health

Keep answers short — 2 to 4 sentences unless the user asks for more detail."""


print("✅ Crop Recommendation Model Loaded")
print("✅ Disease Detection Model Loaded")
print(f"✅ Disease Classes Loaded: {len(disease_info)}")
print("✅ Fruit Identification Model Loaded")
print(f"✅ Fruit Labels Loaded: {len(fruit_labels)}")
print("✅ Health Classifier Model Loaded")
print(f"✅ Health Classes Loaded: {len(health_labels)}")


# ==========================
# Disease Prediction Function
# ==========================

def predict_disease(image_path):

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None, 0

    input_shape = input_details[0]["shape"]
    height = input_shape[1]
    width = input_shape[2]

    img = img.resize((width, height))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = int(np.argmax(output))
    confidence = float(np.max(output) * 100)
    class_name = disease_classes[predicted_index]

    return class_name, confidence


# ==========================
# Fruit Identification Function
# ==========================

FRUIT_CONFIDENCE_THRESHOLD = 50  # matches Flutter app


def predict_fruit(image_path):

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None, 0

    input_shape = fruit_input_details[0]["shape"]
    height = input_shape[1]
    width = input_shape[2]

    img = img.resize((width, height))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    fruit_interpreter.set_tensor(fruit_input_details[0]["index"], img)
    fruit_interpreter.invoke()

    output = fruit_interpreter.get_tensor(fruit_output_details[0]["index"])[0]

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index] * 100)

    if confidence < FRUIT_CONFIDENCE_THRESHOLD:
        return "Low confidence prediction", 0

    class_name = fruit_labels[predicted_index]
    return class_name, confidence


def get_fruit_info(fruit_name):
    """Case-insensitive lookup by 'name' field, matches Dart _getFruitInfo()."""
    if not fruit_name:
        return None

    normalized = fruit_name.lower().strip()
    normalized = FRUIT_NAME_ALIASES.get(normalized, normalized)

    for fruit in fruit_data:
        if str(fruit.get("name", "")).lower() == normalized:
            return fruit

    return None


# ==========================
# Health Classification Function
# ==========================

HEALTH_CONFIDENCE_THRESHOLD = 50  # matches Flutter app


def predict_health(image_path):

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None, 0

    input_shape = health_input_details[0]["shape"]
    height = input_shape[1]
    width = input_shape[2]

    img = img.resize((width, height))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    health_interpreter.set_tensor(health_input_details[0]["index"], img)
    health_interpreter.invoke()

    output = health_interpreter.get_tensor(health_output_details[0]["index"])[0]

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index] * 100)
    class_name = health_labels[predicted_index]

    # confidence is returned as-is even when low, matching the Flutter app
    return class_name, confidence


# ==========================
# HOME
# ==========================

@app.route("/")
def home():
    return render_template("home.html")


# ==========================
# CROP RECOMMENDATION
# ==========================

@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():

    result = None
    crop = None          
    N = P = K = temp = humidity = ph = rainfall = None

    if request.method == "POST":

        N = float(request.form["Nitrogen"])
        P = float(request.form["Phosporus"])
        K = float(request.form["Potassium"])
        temp = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        ph = float(request.form["Ph"])
        rainfall = float(request.form["Rainfall"])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Only MinMax Scaling (matches training notebook)
        scaled_features = ms.transform(single_pred)
        prediction = model.predict(scaled_features)

        if prediction[0] in CROP_DICT:
            crop = CROP_DICT[prediction[0]]
            result = f"{crop} is the best crop to be cultivated."
        else:
            result = "Sorry, we could not determine the best crop."

    return render_template(
        "recommendation.html",
        result=result,
        crop=crop,  
        N=N, P=P, K=K, temp=temp, humidity=humidity, ph=ph, rainfall=rainfall,
    )


# ==========================
# DISEASE DETECTION
# ==========================

@app.route("/disease")
def disease():
    return render_template("disease.html")


@app.route("/predict-disease", methods=["POST"])
def predict_disease_route():

    if "image" not in request.files:
        return render_template("disease.html", error="Please select an image.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("disease.html", error="No image selected.")

    if not allowed_file(file.filename):
        return render_template("disease.html", error="Only PNG, JPG and JPEG files are allowed.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    disease_name, confidence = predict_disease(filepath)

    if disease_name is None:
        return render_template("disease.html", error="Unable to process image.")

    disease_details = disease_info.get(disease_name, {})

    return render_template(
        "disease.html",
        image_path=filepath.replace("\\", "/"),
        disease=disease_name,
        confidence=round(confidence, 2),
        details=disease_details,
    )


# ==========================
# FRUIT IDENTIFICATION
# ==========================

@app.route("/identify")
def identify():
    return render_template("identify.html")


@app.route("/predict-fruit", methods=["POST"])
def predict_fruit_route():

    if "image" not in request.files:
        return render_template("identify.html", error="Please select an image.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("identify.html", error="No image selected.")

    if not allowed_file(file.filename):
        return render_template("identify.html", error="Only PNG, JPG and JPEG files are allowed.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    fruit_name, confidence = predict_fruit(filepath)

    if fruit_name is None:
        return render_template("identify.html", error="Unable to process image.")

    if fruit_name == "Low confidence prediction":
        return render_template(
            "identify.html",
            image_path=filepath.replace("\\", "/"),
            low_confidence=True,
        )

    info = get_fruit_info(fruit_name)

    return render_template(
        "identify.html",
        image_path=filepath.replace("\\", "/"),
        fruit=fruit_name,
        confidence=round(confidence, 2),
        info=info,
    )


# ==========================
# HEALTH CLASSIFIER
# ==========================

@app.route("/health")
def health():
    return render_template("health.html")


@app.route("/predict-health", methods=["POST"])
def predict_health_route():

    if "image" not in request.files:
        return render_template("health.html", error="Please select an image.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("health.html", error="No image selected.")

    if not allowed_file(file.filename):
        return render_template("health.html", error="Only PNG, JPG and JPEG files are allowed.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    class_name, confidence = predict_health(filepath)

    if class_name is None:
        return render_template("health.html", error="Unable to process image.")

    low_confidence = confidence < HEALTH_CONFIDENCE_THRESHOLD

    fruit_name, status = (
        class_name.split("__", 1) if "__" in class_name else (class_name, None)
    )

    return render_template(
        "health.html",
        image_path=filepath.replace("\\", "/"),
        fruit_name=fruit_name,
        status=status,
        confidence=round(confidence, 2),
        low_confidence=low_confidence,
    )


# ==========================
# ANALYTICS
# ==========================

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ==========================
# INFORMATION
# ==========================

@app.route("/information")
def information():
    fruit_list = sorted(label.title() for label in fruit_labels)
    disease_list = sorted(humanize_label(c) for c in disease_classes)
    crop_list = [CROP_DICT[k] for k in sorted(CROP_DICT)]
    health_list = sorted(humanize_label(c) for c in health_labels)

    return render_template(
        "information.html",
        fruit_list=fruit_list,
        disease_list=disease_list,
        crop_list=crop_list,
        health_list=health_list,
    )


# ==========================
# ABOUT
# ==========================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# CHATBOT
# ==========================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    if len(user_message) > 500:
        return jsonify({"error": "Message too long."}), 400

    if gemini_client is None:
        return jsonify({"error": "Chatbot is not configured."}), 503

    try:
        gemini_history = []
        for turn in history[-10:]:  # cap context sent per request
            role = "user" if turn.get("role") == "user" else "model"
            gemini_history.append(
                types.Content(role=role, parts=[types.Part(text=turn.get("text", ""))])
            )

        chat_session = gemini_client.chats.create(
            model="gemini-3.6-flash",
            config=types.GenerateContentConfig(
                system_instruction=CHATBOT_SYSTEM_INSTRUCTION,
                max_output_tokens=300,
            ),
            history=gemini_history,
        )

        response = chat_session.send_message(user_message)
        reply = response.text

    except Exception as e:
        print("Gemini error:", e)
        return jsonify({"error": "Something went wrong. Please try again."}), 500

    return jsonify({"reply": reply})


# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":
    app.run(debug=True)