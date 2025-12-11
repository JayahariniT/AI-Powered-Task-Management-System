import pandas as pd
import os
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk

# Download nltk data if needed
nltk.download('stopwords')
nltk.download('punkt')

STOP = set(stopwords.words("english"))
PS = PorterStemmer()

CSV_FILE = "merged_task_dataset.csv"

DEFAULT_COLUMNS = [
    "task_id", "description", "task_type", "priority",
    "assignee", "created_at", "due_date",
    "estimated_effort", "user_workload"
]


# ------------------------- TEXT CLEANING -------------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    s = str(text).lower()

    # remove URLs
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)

    # remove emojis
    s = re.sub(r'[^\x00-\x7F]+', ' ', s)

    # remove punctuation
    s = re.sub(r'[^a-z0-9\s]', ' ', s)

    # tokenize
    tokens = nltk.word_tokenize(s)

    # remove stopwords + stem
    tokens = [PS.stem(t) for t in tokens if t not in STOP and len(t) > 1]

    return " ".join(tokens)


# ------------------------- COLUMN CHECK -------------------------
def ensure_df_columns(df):
    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[DEFAULT_COLUMNS]


# ------------------------- LOAD TASKS -------------------------
def load_tasks():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)

        df = ensure_df_columns(df)

        # Normalize ID
        df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")

        # Normalize assignee
        df["assignee"] = df["assignee"].astype(str).fillna("Unassigned").replace("", "Unassigned")
        df["assignee_norm"] = df["assignee"].str.lower().str.strip()

        # Normalize priority (fix inconsistent labels)
        df["priority"] = df["priority"].astype(str).str.title().replace({
            "Urgent ": "Urgent",
            "High ": "High",
            "Medium ": "Medium",
            "Low ": "Low"
        })

        # Normalize task types
        df["task_type"] = df["task_type"].astype(str).str.title().replace({
            "Bug ": "Bug",
            "Task ": "Task",
            "Feature ": "Feature"
        })

        # Clean description
        df["description_clean"] = df["description"].fillna("").apply(clean_text)

        # Normalize dates
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")

        # Fill missing created_at
        df["created_at"] = df["created_at"].fillna(datetime.now())

        # Fill missing due_date → default 7 days later
        df["due_date"] = df["due_date"].fillna(df["created_at"] + pd.Timedelta(days=7))

        # Numeric cleanup
        df["estimated_effort"] = pd.to_numeric(df["estimated_effort"], errors="coerce").fillna(0).astype(int)
        df["user_workload"] = pd.to_numeric(df["user_workload"], errors="coerce").fillna(0).astype(int)

        # FEATURE: days until due
        df["days_until_due"] = (df["due_date"] - df["created_at"]).dt.days

        return df

    # Empty file → return correct structure
    return pd.DataFrame(columns=DEFAULT_COLUMNS + ["assignee_norm", "description_clean", "days_until_due"])


# ------------------------- SAVE -------------------------
def save_tasks(df):
    df_out = df.copy()
    # remove helper columns
    for col in ["assignee_norm", "description_clean", "days_until_due"]:
        if col in df_out:
            df_out = df_out.drop(columns=[col])
    df_out.to_csv(CSV_FILE, index=False)


# ------------------------- ADD TASK -------------------------
def add_task(description, task_type, priority, assignee, due_date,
             estimated_effort, user_workload):

    df = load_tasks()

    new_id = 1 if df.empty else int(df["task_id"].max()) + 1

    try:
        due_date_fmt = pd.to_datetime(due_date) if due_date else pd.NaT
    except:
        due_date_fmt = pd.NaT

    new_task = {
        "task_id": new_id,
        "description": description or "",
        "description_clean": clean_text(description),
        "task_type": (task_type or "Task").title(),
        "priority": (priority or "Medium").title(),
        "assignee": assignee or "Unassigned",
        "created_at": datetime.now(),
        "due_date": due_date_fmt,
        "estimated_effort": int(estimated_effort or 0),
        "user_workload": int(user_workload or 0),
        "assignee_norm": (assignee or "Unassigned").lower(),
    }

    # days until due
    if pd.notna(due_date_fmt):
        new_task["days_until_due"] = (due_date_fmt - datetime.now()).days
    else:
        new_task["days_until_due"] = 7

    df = pd.concat([df, pd.DataFrame([new_task])], ignore_index=True)
    save_tasks(df)

    out = new_task.copy()
    out.pop("assignee_norm", None)
    return out
