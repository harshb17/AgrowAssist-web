<div align="center">

# 🌱 AgrowAssist

### AI-Powered Smart Agricultural Assistance Platform

*Four models. One platform. Zero guesswork.*

[**🚀 Live Demo**](https://gkc55-agrowassist.hf.space) · [**📱 Android App**](https://github.com/Gunjankumar55/Agroassist----Smart-Agricultural-Assistance-System-App)

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_API-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

---

## The Problem

Farmers make high-stakes decisions with incomplete information. *What should I plant in this soil? Is this leaf diseased or just stressed? Is this produce still sellable?* Getting these wrong costs a season.

AgrowAssist puts four trained models behind those decisions — each one answering a question a farmer actually asks, in seconds, from a browser.

---

## What It Does

| Module | Input | Output | Classes |
|---|---|---|---|
| 🌾 **Crop Recommendation** | 7 soil & climate readings | Best-fit crop | 22 |
| 🐛 **Disease Detection** | Leaf photo | Diagnosis + treatment plan | 39 |
| 🍎 **Produce Identification** | Photo | Name + nutrition data | 41 |
| 🩺 **Freshness Check** | Photo | Fresh or spoiled | 28 |
| 📊 **Analytics** | — | Interactive Power BI dashboard | — |
| 🤖 **AgroBot** | Natural language | Agriculture-scoped answers | — |

---

## How It Works

### Crop Recommendation

~~~
N, P, K, temperature, humidity, pH, rainfall
        ↓
   MinMaxScaler
        ↓
 RandomForestClassifier
        ↓
   Predicted crop
~~~

Trained on a 22-crop dataset. The scaling step matters — the model was trained on MinMax-scaled features, so inference has to match exactly or predictions silently drift.

### Vision Modules

All three image models share one pipeline:

~~~
Upload → RGB → resize 224×224 → normalize /255.0 → TFLite interpreter → argmax + confidence
~~~

Each has a **confidence floor**. Below threshold, the app says *"no confident prediction"* and suggests how to retake the photo — rather than showing a wrong answer with a real-looking percentage next to it.

### AgroBot

Gemini with a scoped system instruction — answers agriculture questions, redirects off-topic ones, and routes users to the right tool. Chat history persists in `localStorage` and is capped before being sent back as context.

---

## Tech Stack

**Backend** — Flask · Python 3.9 · Gunicorn  
**ML** — scikit-learn (Random Forest) · TensorFlow Lite (3 CNN classifiers)  
**AI** — Google Gemini API  
**Frontend** — HTML5 · CSS3 · Vanilla JS · Bootstrap 5  
**Analytics** — Power BI (embedded)  
**Deploy** — Docker · Hugging Face Spaces

No frontend framework. Every component, animation, and interaction is hand-written CSS and vanilla JS.

---

## Running Locally

~~~bash
git clone https://github.com/Gunjankumar55/AgrowAssist-Web.git
cd AgrowAssist-Web

pip install -r requirements.txt
~~~

Create a `.env` file in the project root:

~~~
GEMINI_API_KEY=your_key_here
~~~

~~~bash
python app.py
~~~

Open `http://127.0.0.1:5000`

> **Note:** model files are stored with Git LFS. Run `git lfs install` before cloning, or the `.tflite` / `.pkl` files will download as pointer text instead of actual models.

### With Docker

~~~bash
docker build -t agrowassist .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key agrowassist
~~~

---

## Project Structure

~~~
├── app.py                 # Flask routes + all four inference pipelines
├── models/                # TFLite classifiers
├── templates/             # Jinja templates, all extending base.html
├── static/
│   ├── css/style.css      # Single design system
│   ├── js/main.js         # Scroll reveal, upload zones, filters
│   └── js/chatbot.js      # AgroBot widget
├── model.pkl              # Random Forest
├── minmaxscaler.pkl       # Fitted scaler
├── disease_data.json      # Symptoms, treatments, prevention
├── fruit_data.json        # Nutrition data
└── Dockerfile
~~~

---

## Notes From the Build

A few things worth recording, since they weren't obvious going in:

- **The original Flask app loaded a `StandardScaler` that never existed.** Tracing back to the training notebook showed the pipeline only ever used `MinMaxScaler` — the app had been carrying a phantom dependency that crashed on startup.
- **Label order is load-bearing.** Class names have to come from the training pipeline, not from whatever order a JSON file happens to be in. Getting this wrong doesn't error — it just returns confidently wrong answers.
- **`labels.txt` says `raddish`, `fruit_data.json` says `Radish`.** One character, and that class silently returns no nutrition data. Handled with an alias map rather than editing `labels.txt`, whose order must match the model.

---



<div align="center">

**Built by [Gunjankumar Choudhari](https://gunjankumar55.github.io/Gunjan_Portfolio/)**

Computer Engineering Graduate 

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Gunjankumar55)
[![Portfolio](https://img.shields.io/badge/Portfolio-52B788?style=flat-square&logo=googlechrome&logoColor=white)](https://gunjankumar55.github.io/Gunjan_Portfolio/)

</div>
