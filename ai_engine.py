import joblib
import numpy as np
from datetime import datetime, timedelta
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')

STOP = set(stopwords.words("english"))
PS = PorterStemmer()

def clean_text(text):
    if not text:
        return ""
    s = text.lower()
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'[^\x00-\x7F]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = nltk.word_tokenize(s)
    tokens = [PS.stem(t) for t in tokens if t not in STOP and len(t) > 1]
    return " ".join(tokens)

def load_models():
    tfidf = joblib.load("models/tfidf.joblib")
    task_model = joblib.load("models/task_svm.joblib")
    task_le = joblib.load("models/task_label_encoder.joblib")

    prio_model = joblib.load("models/priority_rf.joblib")
    prio_le = joblib.load("models/priority_label_encoder.joblib")

    return tfidf, task_model, task_le, prio_model, prio_le

def predict_all(description):
    tfidf, task_model, task_le, prio_model, prio_le = load_models()

    clean = clean_text(description)
    X = tfidf.transform([clean])

    # Task Type Prediction
    t_pred = task_model.predict(X)[0]
    task_type = task_le.inverse_transform([t_pred])[0]

    # Priority Prediction
    dummy = np.array([[7, 3, 0]])
    priority_pred = prio_model.predict(np.hstack([X.toarray(), dummy]))[0]
    priority = prio_le.inverse_transform([priority_pred])[0]

    # Suggested Effort
    wc = len(clean.split())
    effort = 1 if wc < 5 else 3 if wc < 15 else 5

    # Suggested Due Date
    if priority == "Urgent":
        due = datetime.now() + timedelta(days=1)
    elif priority == "High":
        due = datetime.now() + timedelta(days=3)
    else:
        due = datetime.now() + timedelta(days=7)

    return {
        "task_type": task_type,
        "priority": priority,
        "suggested_effort": effort,
        "suggested_due_date": due.strftime("%Y-%m-%d"),
    }
