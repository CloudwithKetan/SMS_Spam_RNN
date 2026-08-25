import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

# Initialize Flask App
app = Flask(__name__)

# Load Model
MODEL_PATH = "RNNmodel.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Spam Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc; min-height: 100vh; display: flex;
            align-items: center; justify-content: center; padding: 20px;
        }
        .container {
            background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px;
            padding: 40px; width: 100%; max-width: 550px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }
        h1 {
            font-size: 1.8rem; font-weight: 700; margin-bottom: 8px;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        p.subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 25px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-size: 0.875rem; font-weight: 600; color: #cbd5e1; }
        textarea {
            width: 100%; height: 120px; padding: 14px; background: #0f172a;
            border: 1px solid #334155; border-radius: 12px; color: #f8fafc;
            font-size: 0.95rem; resize: vertical; outline: none; transition: all 0.3s ease;
        }
        textarea:focus { border-color: #818cf8; box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25); }
        button {
            width: 100%; padding: 14px;
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            border: none; border-radius: 12px; color: #ffffff;
            font-size: 1rem; font-weight: 600; cursor: pointer; transition: transform 0.2s ease;
        }
        button:hover { opacity: 0.95; transform: translateY(-1px); }
        .result-card {
            margin-top: 25px; padding: 20px; border-radius: 12px;
            background: #0f172a; border: 1px solid #334155; text-align: center;
        }
        .result-title { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 5px; }
        .result-label { font-size: 1.6rem; font-weight: 700; }
        .spam { color: #f87171; }
        .ham { color: #4ade80; }
        .error-msg {
            margin-top: 15px; padding: 12px; background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444; color: #fca5a5; border-radius: 10px; font-size: 0.9rem; text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SMS Spam Classifier</h1>
        <p class="subtitle">Enter SMS text or space-separated token sequence below.</p>

        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="sequence_input">Input Message or Token Sequence</label>
                <textarea id="sequence_input" name="sequence_input" placeholder="e.g. Free entry in 2 a wkly comp to win FA Cup final tkts..." required>{{ user_input if user_input else '' }}</textarea>
            </div>
            <button type="submit">Analyze Message</button>
        </form>

        {% if prediction_score is not none %}
        <div class="result-card">
            <div class="result-title">Classification Result</div>
            <div class="result-label {{ 'spam' if prediction_score > 0.5 else 'ham' }}">
                {{ "SPAM" if prediction_score > 0.5 else "HAM (Clean)" }}
            </div>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 5px;">
                Probability Score: {{ "%.4f"|format(prediction_score) }}
            </p>
        </div>
        {% endif %}

        {% if error_message %}
        <div class="error-msg">{{ error_message }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

def preprocess_input(raw_input, target_length=50):
    """
    Parses comma/space-separated numbers or falls back to basic encoding 
    and pads/truncates sequence to match input shape (batch, 50).
    """
    cleaned = raw_input.replace(",", " ")
    tokens = [int(tok) for tok in cleaned.split() if tok.isdigit()]

    # Fallback character encoding if numeric sequence tokens aren't provided directly
    if not tokens:
        tokens = [ord(char) % 1000 for char in raw_input if char.isalnum()]

    if len(tokens) == 0:
        raise ValueError("Please enter a valid text message or numeric sequence.")

    # Pre-padding to target_length of 50
    if len(tokens) < target_length:
        padded = [0] * (target_length - len(tokens)) + tokens
    else:
        padded = tokens[-target_length:]

    return np.array([padded], dtype=np.float32)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, prediction_score=None, error_message=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_score=None,
            error_message="Model file 'RNNmodel.pkl' could not be loaded."
        )

    user_input = request.form.get("sequence_input", "")
    try:
        processed_data = preprocess_input(user_input, target_length=50)
        prediction = model.predict(processed_data)
        
        # Extract binary classification score
        score = float(prediction[0][0]) if hasattr(prediction[0], '__len__') else float(prediction)

        return render_template_string(
            HTML_TEMPLATE,
            prediction_score=score,
            error_message=None,
            user_input=user_input
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_score=None,
            error_message=str(e),
            user_input=user_input
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
