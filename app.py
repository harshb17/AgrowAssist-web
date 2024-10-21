from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

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

print("✅ Crop Recommendation Model Loaded")

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
# RUN APP
# ==========================

if __name__ == "__main__":
    app.run(debug=True)
