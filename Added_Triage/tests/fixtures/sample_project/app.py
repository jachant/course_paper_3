"""Sample vulnerable Flask application for testing."""

import sqlite3
from flask import Flask, request

app = Flask(__name__)
DB_PATH = "app.db"


def get_db():
    return sqlite3.connect(DB_PATH)


@app.route("/user")
def get_user():
    # SQL Injection — TRUE POSITIVE
    user_id = request.args.get("id")
    db = get_db()
    cursor = db.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")  # vulnerable!
    return str(cursor.fetchall())


@app.route("/user_safe")
def get_user_safe():
    # Parameterized query — FALSE POSITIVE if flagged
    user_id = request.args.get("id")
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return str(cursor.fetchall())


@app.route("/greet")
def greet():
    # XSS — TRUE POSITIVE (no escaping)
    name = request.args.get("name", "world")
    return f"<h1>Hello, {name}!</h1>"


@app.route("/eval")
def run_eval():
    # Code injection — TRUE POSITIVE
    expr = request.args.get("expr")
    result = eval(expr)  # dangerous!
    return str(result)


@app.route("/safe_eval")
def safe_eval_endpoint():
    # Not actually using eval — FALSE POSITIVE if flagged as eval
    expr = request.args.get("expr", "0")
    allowed = set("0123456789+-*/(). ")
    if all(c in allowed for c in expr):
        result = eval(expr)
        return str(result)
    return "Invalid expression", 400
