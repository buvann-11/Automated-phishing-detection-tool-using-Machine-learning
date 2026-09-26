# Automated Phishing Detection Tool using Machine Learning

A Flask web app that checks whether a URL is likely to be **phishing**. It combines an ensemble of machine-learning models trained on the [PhiUSIIL Phishing URL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) with rule-based risk checks. It can also email the user the verdict.

## Where the code is

The project lives in:

```
Downloads/Phishing_Detection1 (2)/Phishing_Detection1/Phishing_Detection/phish_detector/
├── .env.example                  # SMTP settings template (copy to .env)
├── requirements.txt
├── phiusiil+phishing+url+dataset/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
└── phish_detector/
    ├── app.py                    # Flask web app
    ├── templates/index.html      # UI
    ├── data/phishing_db.txt      # Known phishing URLs (exact-match blocklist)
    ├── models/                   # Pre-trained models (.joblib)
    ├── models_src/
    │   ├── train_models.py       # Trains vectorizer + 3 models
    │   └── predict.py            # Ensemble prediction (also a CLI)
    └── utils/
        ├── feature_extraction.py # URL features, typosquatting & risk heuristics
        └── email_sender.py       # Optional SMTP email alerts
```

## How it works

1. **Blocklist:** if the URL is listed in `data/phishing_db.txt`, it is flagged right away.
2. **Features:** hand-crafted URL features (length, digits, special characters, IP-as-domain, HTTPS, and more) are combined with TF-IDF over URL tokens (1–2-grams).
3. **Ensemble:** three models vote:
   - calibrated LinearSVC
   - logistic regression (saved as `rf_model.joblib`)
   - XGBoost
4. **Risk heuristics:** typosquatting, suspicious keywords, suspicious domain patterns and missing HTTPS add to a risk score.
5. **Verdict:**
   - ⚠️ **phishing** when the models or the risk score say so
   - ✅ **safe** otherwise
   - ❓ **uncertain** when the URL's tokens are mostly unknown to the models and the risk checks are inconclusive

## Getting started

```bash
cd "Downloads/Phishing_Detection1 (2)/Phishing_Detection1/Phishing_Detection/phish_detector"
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cd phish_detector
python app.py                   # open http://127.0.0.1:5000
```

Pre-trained models are included. To retrain them on the dataset, run this from the inner `phish_detector/` folder:

```bash
python models_src/train_models.py
```

To test URLs from the terminal:

```bash
python models_src/predict.py
```

### Email alerts (optional)

Copy `.env.example` to `.env` and fill in your SMTP details. For Gmail, use an [App Password](https://myaccount.google.com/apppasswords). **Never commit `.env`.** It is listed in `.gitignore`. Without SMTP settings, the app still works and prints the alert to the console instead.

## Other folders in this repository

`Desktop/SpyDev/e-commerce` and `Desktop/donation-tracker` contain separate React/Vite front-end projects that were uploaded alongside this tool. They are unrelated to phishing detection.

## Tech stack

Python · Flask · scikit-learn · XGBoost · pandas · SciPy · tldextract · BeautifulSoup
