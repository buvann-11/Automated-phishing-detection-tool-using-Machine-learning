from flask import Flask, render_template, request
import os
#from phish_detector.models_src.predict import ensemble_predict
#from phish_detector.utils.email_sender import send_alert_email
from models_src.predict import ensemble_predict
from utils.email_sender import send_alert_email


app = Flask(__name__)

# Load phishing database (simple text file with known phishing URLs) if present
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_DIR, "data", "phishing_db.txt")
phishing_db = set()
if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "r", encoding="utf-8", errors="ignore") as f:
        phishing_db = set(line.strip() for line in f if line.strip())


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_email = None
    url = None

    if request.method == "POST":
        url = request.form.get("url")
        user_email = request.form.get("email")

        if url in phishing_db:
            result = "⚠️ This link is a known phishing site!"
            send_alert_email(user_email, url, is_phishing=True)
        else:
            try:
                prediction = ensemble_predict(url)
                # prediction is a dict: use fields
                final = prediction.get("final", 0)
                is_phish = final == 1
                conf = prediction.get("confidence", 0.0)
                votes = prediction.get("votes", 0)
                risk = prediction.get("risk", {})
                rscore = risk.get("score", 0.0)
                rreasons = ", ".join(risk.get("reasons", [])[:3])
                details = f"conf {conf:.2f}, votes {votes}/3, risk {rscore:.2f}{' - ' + rreasons if rreasons else ''}"
                if final == -1:
                    # The models could not classify this URL confidently; don't claim it is safe.
                    result = f"❓ Could not classify this URL confidently ({details}). Treat it with caution."
                elif is_phish:
                    result = f"⚠️ Classified as phishing (conf {conf:.2f}, votes {votes}/3, risk {rscore:.2f}{' - ' + rreasons if rreasons else ''})."
                    send_alert_email(user_email, url, is_phishing=True)
                else:
                    result = f"✅ Appears safe (conf {conf:.2f}, votes {votes}/3, risk {rscore:.2f}{' - ' + rreasons if rreasons else ''})."
                    # Notify the provided email even for safe results
                    send_alert_email(user_email, url, is_phishing=False)
            except FileNotFoundError as e:
                result = "Models not found. Please train first: python phish_detector/models_src/train_models.py"
            except Exception as e:
                result = f"Prediction failed: {e}"

    return render_template("index.html", result=result, url=url, email=user_email)


if __name__ == "__main__":
    # Debug mode is opt-in: set FLASK_DEBUG=1 while developing.
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
