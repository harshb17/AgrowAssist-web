from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
import json
from PIL import Image
import os
from werkzeug.utils import secure_filename

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


print("✅ Crop Recommendation Model Loaded")
print("✅ Disease Detection Model Loaded")
print(f"✅ Disease Classes Loaded: {len(disease_info)}")
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
# RUN APP
# ==========================

if __name__ == "__main__":
    app.run(debug=True)
