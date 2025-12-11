from flask import Flask, request, jsonify
from flask_cors import CORS
from task_manager import load_tasks, add_task, save_tasks
import pandas as pd
from datetime import date

app = Flask(__name__)
CORS(app)

# ---------------- GET TASKS ----------------
@app.route("/tasks", methods=["GET"])
def get_tasks():
    assignee = request.args.get("assignee", "").lower().strip()
    df = load_tasks()

    if assignee:
        df = df[df["assignee_norm"] == assignee]

    df["created_sort"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.sort_values("created_sort", ascending=False)

    df = df.drop(columns=["created_sort", "assignee_norm"], errors="ignore")

    return jsonify(df.to_dict(orient="records"))

# ---------------- ADD TASK ----------------
@app.route("/tasks", methods=["POST"])
def create_task():
    data = request.json or {}

    new_task = add_task(
        description=data.get("description"),
        task_type=data.get("task_type"),
        priority=data.get("priority"),
        assignee=data.get("assignee"),
        due_date=data.get("due_date"),
        estimated_effort=data.get("estimated_effort"),
        user_workload=data.get("user_workload")
    )
    return jsonify(new_task), 201

# ---------------- DELETE TASK ----------------
@app.route("/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    df = load_tasks()
    df = df[df["task_id"] != task_id]
    save_tasks(df)
    return jsonify({"status": "deleted", "id": task_id})

# ---------------- EDIT TASK ----------------
@app.route("/tasks/<int:task_id>", methods=["PUT"])
def edit_task(task_id):
    df = load_tasks()
    data = request.json or {}

    if task_id not in df["task_id"].values:
        return jsonify({"error": "Task not found"}), 404

    for col in ["description", "priority", "due_date"]:
        if col in data:
            df.loc[df["task_id"] == task_id, col] = data[col]

    save_tasks(df)

    updated = df[df["task_id"] == task_id].to_dict(orient="records")[0]
    return jsonify(updated)

# ---------------- PRIORITY CHART ----------------
@app.route("/metrics/priority")
def priority_metrics():
    df = load_tasks()
    counts = df["priority"].value_counts()
    return jsonify({"labels": counts.index.tolist(),
                    "values": counts.values.tolist()})

# ---------------- ASSIGNEE CHART ----------------
@app.route("/metrics/assignee")
def assign_metrics():
    df = load_tasks()
    grp = df.groupby("assignee")["task_id"].count()
    return jsonify({"labels": grp.index.tolist(),
                    "values": grp.values.tolist()})

# ---------------- WORKLOAD CHART ----------------
@app.route("/metrics/workload")
def workload_metrics():
    df = load_tasks()
    grp = df.groupby("assignee")["user_workload"].sum()
    return jsonify({"labels": grp.index.tolist(),
                    "values": grp.values.tolist()})

# ---------------- OVERDUE ----------------
@app.route("/metrics/overdue")
def overdue_metrics():
    df = load_tasks()
    df["due_dt"] = pd.to_datetime(df["due_date"], errors="coerce")
    today = pd.to_datetime(date.today())

    od = df[df["due_dt"] < today]

    od = od.drop(columns=["assignee_norm", "due_dt"], errors="ignore")
    return jsonify({"tasks": od.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(debug=True)
