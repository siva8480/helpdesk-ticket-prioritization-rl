from flask import Flask, render_template, request, redirect
import numpy as np
import pickle

app = Flask(__name__)

q_table = pickle.load(open("q_table.pkl", "rb"))

PRIORITY_MAP = {0:"Critical",1:"High",2:"Medium",3:"Low"}

DEPARTMENTS = ["IT", "HR", "Finance", "Sales", "Operations"]
DEPT_MAP = {d:i for i,d in enumerate(DEPARTMENTS)}

ticket_queue = []

def extract_features(text, dept):
    text = text.lower()

    features = [
        int("down" in text or "crash" in text or "failed" in text),
        int("all users" in text or "entire" in text or "everyone" in text),
        int("breach" in text or "hack" in text or "unauthorized" in text),
        int("data" in text or "database" in text or "loss" in text),
        int("payment" in text or "revenue" in text or "transaction" in text),
        int("slow" in text or "delay" in text or "latency" in text),
        int("vpn" in text or "network" in text or "connectivity" in text),
        int("urgent" in text or "asap" in text or "immediately" in text),
        int("error" in text or "failure" in text),
        int("login" in text or "authentication" in text),
        int("blocked" in text or "cannot" in text),
        int("impact" in text or "business" in text),
        int("issue" in text or "problem" in text),
        int("request" in text or "please" in text or "can you" in text),
    ]

    dept_vector = [0]*len(DEPARTMENTS)
    dept_vector[DEPT_MAP.get(dept, 0)] = 1

    return np.array(features + dept_vector)

@app.route("/", methods=["GET", "POST"])
def home():
    global ticket_queue

    if request.method == "POST":
        text = request.form["ticket"]
        dept = request.form["department"]

        fv = extract_features(text, dept)
        state = tuple(fv.astype(int))

        q_values = q_table.get(state, np.zeros(4))
        action = int(np.argmax(q_values))
        result = PRIORITY_MAP[action]

        ticket_queue.append({
            "text": text,
            "department": dept,
            "priority": result
        })

        priority_order = {"Critical":0, "High":1, "Medium":2, "Low":3}
        ticket_queue = sorted(ticket_queue, key=lambda x: priority_order[x["priority"]])

    return render_template("index.html", queue=ticket_queue)

@app.route("/clear", methods=["POST"])
def clear():
    global ticket_queue
    ticket_queue = []
    return redirect("/")

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))