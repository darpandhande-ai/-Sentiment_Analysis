from flask import Flask, render_template_string, request, jsonify
import pickle
import os
import re

app = Flask(__name__)

# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

MODEL_PATH = "model (1).pkl"
VECTORIZER_PATH = "vectorizer.pkl"

try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("✅ Model and Vectorizer loaded successfully!")
    print("Model:", type(model).__name__)
    print("Vectorizer:", type(vectorizer).__name__)

except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    vectorizer = None
    model = None


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Basic text cleaning.
    IMPORTANT:
    If your model was trained with a specific preprocessing
    pipeline, keep preprocessing consistent with training.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):

    if model is None or vectorizer is None:
        return {
            "sentiment": "Error",
            "confidence": 0,
            "icon": "⚠️",
            "description": "Model could not be loaded."
        }

    cleaned = clean_text(text)

    # Convert text into TF-IDF features
    transformed_text = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(transformed_text)[0]

    # Confidence
    confidence = 0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = float(max(probabilities)) * 100

    # Convert prediction to string
    sentiment = str(prediction)

    # Normalize common sentiment names
    sentiment_lower = sentiment.lower()

    if "positive" in sentiment_lower or sentiment_lower in ["1", "pos"]:
        result = {
            "sentiment": "Positive",
            "confidence": confidence,
            "icon": "😊",
            "description": "This text expresses a positive feeling."
        }

    elif "negative" in sentiment_lower or sentiment_lower in ["0", "neg"]:
        result = {
            "sentiment": "Negative",
            "confidence": confidence,
            "icon": "😞",
            "description": "This text expresses a negative feeling."
        }

    elif "neutral" in sentiment_lower or sentiment_lower in ["2", "neu"]:
        result = {
            "sentiment": "Neutral",
            "confidence": confidence,
            "icon": "😐",
            "description": "This text appears to have a neutral tone."
        }

    else:
        # If your dataset uses other labels
        result = {
            "sentiment": sentiment.title(),
            "confidence": confidence,
            "icon": "🔍",
            "description": "Sentiment detected from the trained model."
        }

    return result


# ============================================================
# MAIN PAGE
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Sentiment Analysis</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {

    --bg: #070b14;
    --bg2: #0c1220;
    --card: rgba(18, 25, 40, 0.72);
    --card-border: rgba(255,255,255,0.09);

    --text: #f7f9fc;
    --muted: #9ba7ba;

    --primary: #7c5cff;
    --primary2: #a855f7;

    --green: #22c55e;
    --red: #ef4444;
    --yellow: #f59e0b;

    --shadow:
        0 25px 80px rgba(0,0,0,0.45);
}

body.light {

    --bg: #f4f7fb;
    --bg2: #ffffff;

    --card: rgba(255,255,255,0.82);
    --card-border: rgba(20,30,50,0.09);

    --text: #101827;
    --muted: #64748b;

    --shadow:
        0 25px 70px rgba(40,50,80,0.15);
}

body.blue {

    --primary: #3b82f6;
    --primary2: #06b6d4;
}

body.green {

    --primary: #10b981;
    --primary2: #22c55e;
}

body.pink {

    --primary: #ec4899;
    --primary2: #8b5cf6;
}

body {

    font-family: "Inter", sans-serif;

    color: var(--text);

    min-height: 100vh;

    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(124,92,255,0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(168,85,247,0.14),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            var(--bg),
            var(--bg2)
        );

    transition: 0.35s ease;

    overflow-x: hidden;
}


/* =========================================================
   BACKGROUND
========================================================= */

.background {

    position: fixed;

    inset: 0;

    pointer-events: none;

    overflow: hidden;

    z-index: -1;
}

.orb {

    position: absolute;

    width: 350px;
    height: 350px;

    border-radius: 50%;

    filter: blur(100px);

    opacity: 0.12;
}

.orb.one {

    background: var(--primary);

    top: -120px;
    left: -100px;
}

.orb.two {

    background: var(--primary2);

    bottom: -150px;
    right: -100px;
}


/* =========================================================
   NAVBAR
========================================================= */

.navbar {

    width: min(1180px, 92%);

    margin: 22px auto;

    padding: 15px 20px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    background: var(--card);

    border: 1px solid var(--card-border);

    backdrop-filter: blur(25px);

    border-radius: 20px;

    box-shadow: var(--shadow);
}

.logo {

    display: flex;

    align-items: center;

    gap: 12px;

    font-family: "Outfit", sans-serif;

    font-size: 20px;

    font-weight: 800;
}

.logo-icon {

    width: 42px;
    height: 42px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 13px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary2)
        );

    box-shadow:
        0 8px 25px rgba(124,92,255,0.35);

    font-size: 21px;
}

.nav-right {

    display: flex;

    gap: 10px;

    align-items: center;
}

.theme-btn {

    width: 42px;
    height: 42px;

    border: 1px solid var(--card-border);

    background: rgba(255,255,255,0.04);

    color: var(--text);

    border-radius: 12px;

    cursor: pointer;

    font-size: 18px;

    transition: 0.2s;
}

.theme-btn:hover {

    transform: translateY(-2px);

    border-color: var(--primary);
}


/* =========================================================
   HERO
========================================================= */

.container {

    width: min(1080px, 92%);

    margin: auto;
}

.hero {

    text-align: center;

    padding: 55px 0 35px;
}

.badge {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 8px 14px;

    border-radius: 50px;

    background: rgba(124,92,255,0.10);

    border: 1px solid rgba(124,92,255,0.25);

    color: #b9aaff;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 0.7px;

    text-transform: uppercase;

    margin-bottom: 20px;
}

.hero h1 {

    font-family: "Outfit", sans-serif;

    font-size: clamp(42px, 7vw, 76px);

    line-height: 1;

    letter-spacing: -3px;

    margin-bottom: 18px;
}

.gradient-text {

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--primary2),
            #22d3ee
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero p {

    color: var(--muted);

    max-width: 650px;

    margin: auto;

    line-height: 1.7;

    font-size: 15px;
}


/* =========================================================
   MAIN CARD
========================================================= */

.main-card {

    background: var(--card);

    border: 1px solid var(--card-border);

    backdrop-filter: blur(30px);

    border-radius: 28px;

    padding: 28px;

    box-shadow: var(--shadow);

    position: relative;

    overflow: hidden;
}

.main-card::before {

    content: "";

    position: absolute;

    top: 0;
    left: 10%;

    width: 80%;
    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            var(--primary),
            transparent
        );
}

.input-header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 14px;
}

.input-title {

    font-size: 14px;

    font-weight: 700;
}

.counter {

    font-size: 12px;

    color: var(--muted);
}

textarea {

    width: 100%;

    min-height: 190px;

    resize: vertical;

    padding: 20px;

    border-radius: 18px;

    border: 1px solid var(--card-border);

    outline: none;

    background: rgba(0,0,0,0.17);

    color: var(--text);

    font-family: inherit;

    font-size: 15px;

    line-height: 1.7;

    transition: 0.25s;
}

textarea:focus {

    border-color: var(--primary);

    box-shadow:
        0 0 0 4px rgba(124,92,255,0.08);
}

textarea::placeholder {

    color: #69768b;
}


/* =========================================================
   BUTTONS
========================================================= */

.action-row {

    display: flex;

    justify-content: space-between;

    gap: 15px;

    margin-top: 18px;

    flex-wrap: wrap;
}

.examples {

    display: flex;

    gap: 8px;

    flex-wrap: wrap;
}

.example-btn {

    border: 1px solid var(--card-border);

    background: rgba(255,255,255,0.035);

    color: var(--muted);

    padding: 10px 13px;

    border-radius: 11px;

    cursor: pointer;

    font-size: 12px;

    transition: 0.2s;
}

.example-btn:hover {

    color: var(--text);

    border-color: var(--primary);

    transform: translateY(-2px);
}

.analyze-btn {

    border: none;

    padding: 13px 25px;

    border-radius: 13px;

    cursor: pointer;

    color: white;

    font-weight: 800;

    font-size: 14px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary2)
        );

    box-shadow:
        0 10px 30px rgba(124,92,255,0.3);

    transition: 0.25s;

    min-width: 170px;
}

.analyze-btn:hover {

    transform: translateY(-3px);

    box-shadow:
        0 15px 40px rgba(124,92,255,0.4);
}

.analyze-btn:active {

    transform: translateY(0);
}


/* =========================================================
   RESULT
========================================================= */

.result {

    display: none;

    margin-top: 25px;

    padding: 25px;

    border-radius: 20px;

    border: 1px solid var(--card-border);

    background: rgba(255,255,255,0.025);

    animation: slideUp 0.45s ease;
}

@keyframes slideUp {

    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result.show {

    display: block;
}

.result-top {

    display: flex;

    align-items: center;

    gap: 18px;
}

.result-icon {

    width: 64px;
    height: 64px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 18px;

    font-size: 30px;

    background: rgba(124,92,255,0.12);

    border: 1px solid var(--card-border);
}

.result-label {

    color: var(--muted);

    font-size: 12px;

    margin-bottom: 4px;
}

.result-sentiment {

    font-family: "Outfit", sans-serif;

    font-size: 30px;

    font-weight: 800;
}

.description {

    color: var(--muted);

    font-size: 13px;

    margin-top: 5px;
}

.confidence-section {

    margin-top: 25px;
}

.confidence-row {

    display: flex;

    justify-content: space-between;

    font-size: 12px;

    margin-bottom: 9px;
}

.confidence-value {

    font-weight: 800;
}

.progress {

    width: 100%;

    height: 9px;

    border-radius: 50px;

    background: rgba(255,255,255,0.07);

    overflow: hidden;
}

.progress-bar {

    height: 100%;

    width: 0%;

    border-radius: inherit;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--primary2)
        );

    transition: width 1s ease;
}


/* =========================================================
   FEATURE CARDS
========================================================= */

.features {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 15px;

    margin-top: 20px;

    margin-bottom: 45px;
}

.feature {

    padding: 20px;

    border-radius: 18px;

    background: var(--card);

    border: 1px solid var(--card-border);

    backdrop-filter: blur(20px);

    transition: 0.25s;
}

.feature:hover {

    transform: translateY(-5px);

    border-color: rgba(124,92,255,0.35);
}

.feature-icon {

    font-size: 25px;

    margin-bottom: 12px;
}

.feature h3 {

    font-family: "Outfit", sans-serif;

    font-size: 16px;

    margin-bottom: 6px;
}

.feature p {

    color: var(--muted);

    font-size: 12px;

    line-height: 1.6;
}


/* =========================================================
   LOADER
========================================================= */

.loader {

    display: none;

    width: 17px;
    height: 17px;

    border: 2px solid rgba(255,255,255,0.3);

    border-top-color: white;

    border-radius: 50%;

    animation: spin 0.7s linear infinite;

    margin-right: 8px;

    vertical-align: middle;
}

@keyframes spin {

    to {
        transform: rotate(360deg);
    }
}


/* =========================================================
   THEME PANEL
========================================================= */

.theme-panel {

    position: fixed;

    top: 80px;
    right: 25px;

    width: 190px;

    padding: 15px;

    background: var(--card);

    border: 1px solid var(--card-border);

    backdrop-filter: blur(25px);

    border-radius: 16px;

    box-shadow: var(--shadow);

    display: none;

    z-index: 20;
}

.theme-panel.show {

    display: block;

    animation: fadeIn 0.25s ease;
}

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(-8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.theme-panel p {

    color: var(--muted);

    font-size: 11px;

    margin-bottom: 10px;
}

.theme-option {

    width: 100%;

    border: 1px solid var(--card-border);

    background: transparent;

    color: var(--text);

    padding: 9px;

    border-radius: 9px;

    margin-bottom: 7px;

    cursor: pointer;

    text-align: left;
}

.theme-option:hover {

    border-color: var(--primary);
}


/* =========================================================
   FOOTER
========================================================= */

footer {

    text-align: center;

    color: var(--muted);

    font-size: 12px;

    padding-bottom: 25px;
}


/* =========================================================
   RESPONSIVE
========================================================= */

@media(max-width: 700px) {

    .features {

        grid-template-columns: 1fr;
    }

    .hero {

        padding-top: 35px;
    }

    .main-card {

        padding: 18px;
    }

    .action-row {

        flex-direction: column;
    }

    .examples {

        width: 100%;
    }

    .analyze-btn {

        width: 100%;
    }

    .navbar {

        margin-top: 12px;
    }
}

</style>

</head>


<body>

<div class="background">

    <div class="orb one"></div>
    <div class="orb two"></div>

</div>


<!-- ======================================================
     NAVBAR
====================================================== -->

<nav class="navbar">

    <div class="logo">

        <div class="logo-icon">
            ✦
        </div>

        <span>Sentiment Analysis</span>

    </div>


    <div class="nav-right">

        <button
            class="theme-btn"
            onclick="toggleThemePanel()"
            title="Change Theme">

            🎨

        </button>

    </div>

</nav>


<!-- ======================================================
     THEME PANEL
====================================================== -->

<div class="theme-panel" id="themePanel">

    <p>CHOOSE INTERFACE THEME</p>

    <button
        class="theme-option"
        onclick="setTheme('')">

        🟣 Purple Premium

    </button>

    <button
        class="theme-option"
        onclick="setTheme('blue')">

        🔵 Ocean Blue

    </button>

    <button
        class="theme-option"
        onclick="setTheme('green')">

        🟢 Emerald

    </button>

    <button
        class="theme-option"
        onclick="setTheme('pink')">

        🌸 Royal Pink

    </button>

    <button
        class="theme-option"
        onclick="setTheme('light')">

        ☀️ Light Premium

    </button>

</div>


<main class="container">


<!-- ======================================================
     HERO
====================================================== -->

<section class="hero">

    <div class="badge">

        ✨ AI POWERED • NLP

    </div>

    <h1>

        Understand the
        <span class="gradient-text">
            Emotion
        </span>

    </h1>

    <p>

        Analyze text instantly with our machine-learning
        sentiment engine. Discover whether your message
        expresses a positive, negative, or neutral feeling.

    </p>

</section>


<!-- ======================================================
     MAIN ANALYZER
====================================================== -->

<section class="main-card">

    <div class="input-header">

        <div class="input-title">

            ✍️ Enter your text

        </div>

        <div
            class="counter"
            id="counter">

            0 / 1000

        </div>

    </div>


    <textarea
        id="textInput"
        maxlength="1000"
        placeholder="Type or paste your review, comment, feedback, or message here..."></textarea>


    <div class="action-row">

        <div class="examples">

            <button
                class="example-btn"
                onclick="useExample('I absolutely love this product! It is amazing and works perfectly.')">

                😊 Positive

            </button>

            <button
                class="example-btn"
                onclick="useExample('This product is terrible. I am very disappointed with it.')">

                😞 Negative

            </button>

            <button
                class="example-btn"
                onclick="useExample('The product arrived today and it is okay.')">

                😐 Neutral

            </button>

        </div>


        <button
            class="analyze-btn"
            id="analyzeBtn"
            onclick="analyzeSentiment()">

            <span
                class="loader"
                id="loader"></span>

            Analyze Sentiment

        </button>

    </div>


    <!-- RESULT -->

    <div
        class="result"
        id="result">

        <div class="result-top">

            <div
                class="result-icon"
                id="resultIcon">

                😊

            </div>

            <div>

                <div class="result-label">
                    DETECTED SENTIMENT
                </div>

                <div
                    class="result-sentiment"
                    id="sentiment">

                    Positive

                </div>

                <div
                    class="description"
                    id="description">

                    This text expresses a positive feeling.

                </div>

            </div>

        </div>


        <div class="confidence-section">

            <div class="confidence-row">

                <span>
                    Model Confidence
                </span>

                <span
                    class="confidence-value"
                    id="confidence">

                    0%

                </span>

            </div>

            <div class="progress">

                <div
                    class="progress-bar"
                    id="progressBar">

                </div>

            </div>

        </div>

    </div>

</section>


<!-- ======================================================
     FEATURES
====================================================== -->

<section class="features">

    <div class="feature">

        <div class="feature-icon">
            ⚡
        </div>

        <h3>
            Instant Analysis
        </h3>

        <p>
            Get sentiment predictions within seconds
            using your trained machine-learning model.
        </p>

    </div>


    <div class="feature">

        <div class="feature-icon">
            🧠
        </div>

        <h3>
            NLP Powered
        </h3>

        <p>
            Text is transformed into TF-IDF features
            before being processed by Naive Bayes.
        </p>

    </div>


    <div class="feature">

        <div class="feature-icon">
            📊
        </div>

        <h3>
            Confidence Score
        </h3>

        <p>
            View the model's prediction confidence
            for every sentiment analysis.
        </p>

    </div>

</section>


<footer>

    Sentiment Analysis • Machine Learning • NLP

</footer>

</main>


<script>

const textarea = document.getElementById("textInput");

const counter = document.getElementById("counter");

const result = document.getElementById("result");

const sentiment = document.getElementById("sentiment");

const confidence = document.getElementById("confidence");

const progressBar = document.getElementById("progressBar");

const resultIcon = document.getElementById("resultIcon");

const description = document.getElementById("description");

const loader = document.getElementById("loader");

const analyzeBtn = document.getElementById("analyzeBtn");


/* ========================================================
   CHARACTER COUNTER
======================================================== */

textarea.addEventListener("input", function() {

    counter.textContent =
        this.value.length + " / 1000";

});


/* ========================================================
   EXAMPLE BUTTON
======================================================== */

function useExample(text) {

    textarea.value = text;

    textarea.dispatchEvent(
        new Event("input")
    );

    textarea.focus();

}


/* ========================================================
   ANALYZE SENTIMENT
======================================================== */

async function analyzeSentiment() {

    const text = textarea.value.trim();

    if (!text) {

        textarea.focus();

        alert("Please enter some text first.");

        return;
    }


    loader.style.display = "inline-block";

    analyzeBtn.disabled = true;

    analyzeBtn.innerHTML =
        '<span class="loader" style="display:inline-block"></span> Analyzing...';


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })

        });


        const data = await response.json();


        result.classList.add("show");

        resultIcon.textContent = data.icon;

        sentiment.textContent =
            data.sentiment;

        description.textContent =
            data.description;

        const value =
            Number(data.confidence || 0);

        confidence.textContent =
            value.toFixed(2) + "%";


        setTimeout(() => {

            progressBar.style.width =
                Math.min(value, 100) + "%";

        }, 100);


        /* Sentiment styling */

        const s =
            data.sentiment.toLowerCase();

        if (s.includes("positive")) {

            resultIcon.style.background =
                "rgba(34,197,94,0.12)";

        }
        else if (s.includes("negative")) {

            resultIcon.style.background =
                "rgba(239,68,68,0.12)";

        }
        else {

            resultIcon.style.background =
                "rgba(245,158,11,0.12)";

        }


        result.scrollIntoView({

            behavior: "smooth",

            block: "nearest"

        });

    }

    catch(error) {

        console.error(error);

        alert(
            "Something went wrong while analyzing the text."
        );

    }

    finally {

        loader.style.display = "none";

        analyzeBtn.disabled = false;

        analyzeBtn.innerHTML =
            "Analyze Sentiment";

    }

}


/* ========================================================
   THEME PANEL
======================================================== */

function toggleThemePanel() {

    document
        .getElementById("themePanel")
        .classList.toggle("show");

}


function setTheme(theme) {

    document.body.classList.remove(
        "light",
        "blue",
        "green",
        "pink"
    );

    if (theme) {

        document.body.classList.add(theme);

    }

    localStorage.setItem(
        "sentimentTheme",
        theme
    );

}


/* ========================================================
   LOAD SAVED THEME
======================================================== */

const savedTheme =
    localStorage.getItem("sentimentTheme");

if (savedTheme) {

    setTheme(savedTheme);

}


/* ========================================================
   CLOSE THEME PANEL
======================================================== */

document.addEventListener("click", function(event) {

    const panel =
        document.getElementById("themePanel");

    const button =
        document.querySelector(".theme-btn");

    if (
        !panel.contains(event.target) &&
        !button.contains(event.target)
    ) {

        panel.classList.remove("show");

    }

});

</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        result = predict_sentiment(text)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("       SENTIMENT ANALYSIS")
    print("=" * 60)
    print("🚀 Starting Flask application...")
    print("🌐 Open: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
