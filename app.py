import os
import joblib
import pandas as pd
import json

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.analysis import analyze_data
from src.model_training import train_model
from src.model_prediction import predict
from src.pdf_report import generate_report

from src.chart_data import get_chart_data
from src.database import (
    create_database,
    register_user,
    login_user,
    save_dataset,
    get_latest_dataset
)
# ===============================
# NEW DATABASE IMPORTS
# ===============================

from src.database import (
    create_database,
    register_user,
    login_user
)

# ===============================
# Flask
# ===============================

app = Flask(__name__)

CORS(app)

# ===============================
# Create SQLite Database
# ===============================

create_database()

# ===============================
# Configuration
# ===============================

UPLOAD_FOLDER = "uploads"

MODEL_FOLDER = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CURRENT_FILE = None

MODEL_PATH = os.path.join(
    app.root_path,
    "models",
    "final_model.pkl"
)

# ===============================
# Helper Functions
# ===============================

# ===============================
# Helper Function
# ===============================

def get_user_dataset():

    user_header = request.headers.get("X-User")

    if not user_header:

        return None

    user = json.loads(user_header)

    dataset = get_latest_dataset(

        user["id"]

    )

    if dataset is None:

        return None

    return dataset["dataset_path"]


def get_model():

    if not os.path.exists(MODEL_PATH):

        return None

    return joblib.load(MODEL_PATH)

# ===============================
# Home
# ===============================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "project": "AI Business Intelligence Platform",

        "version": "3.0",

        "status": "Running"

    })

# ===============================
# Register
# ===============================

@app.route("/register", methods=["POST"])
def register():

    data = request.get_json()

    if not data:

        return jsonify({

            "success": False,

            "message": "Invalid request."

        }), 400

    name = data.get("name", "").strip()

    email = data.get("email", "").strip().lower()

    password = data.get("password", "")

    if not name or not email or not password:

        return jsonify({

            "success": False,

            "message": "All fields are required."

        }), 400

    success, message = register_user(

        name,

        email,

        password

    )

    status = 200 if success else 400

    return jsonify({

        "success": success,

        "message": message

    }), status

# ===============================
# Login
# ===============================

@app.route("/login", methods=["POST"])
def login():

    data = request.get_json()

    if not data:

        return jsonify({

            "success": False,

            "message": "Invalid request."

        }), 400

    email = data.get(

        "email",

        ""

    ).strip().lower()

    password = data.get(

        "password",

        ""

    )

    user = login_user(

        email,

        password

    )

    if user is None:

        return jsonify({

            "success": False,

            "message": "Invalid email or password."

        }), 401

    return jsonify({

        "success": True,

        "message": "Login Successful",

        "user": user

    })
# ===============================
# Upload Dataset
# ===============================

# ===============================
# Upload Dataset
# ===============================

@app.route("/upload", methods=["POST"])
def upload_file():

    global CURRENT_FILE

    user_header = request.headers.get("X-User")

    if not user_header:

        return jsonify({
            "success": False,
            "message": "User not authenticated."
        }), 401

    user = json.loads(user_header)

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "message": "No file uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "No file selected."
        }), 400

    # ==========================
    # Create User Folder
    # ==========================

    user_folder = os.path.join(

        app.config["UPLOAD_FOLDER"],

        str(user["id"])

    )

    os.makedirs(user_folder, exist_ok=True)

    filename = file.filename

    file_path = os.path.join(

        user_folder,

        filename

    )

    file.save(file_path)

    

    CURRENT_FILE = file_path

    save_dataset(

        user["id"],

        filename,

        file_path

    )

    return jsonify({

        "success": True,

        "message": "Dataset uploaded successfully.",

        "dataset_name": filename

    })

# ===============================
# Dataset Information
# ===============================

@app.route("/dataset-info", methods=["GET"])
def dataset_info():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({

            "success": False,

            "message": "No dataset uploaded."

        }), 400

    analysis = analyze_data(dataset)

    model = get_model()

    target = None
    algorithm = None
    status = "Not Trained"

    if model:

        target = model["target"]
        algorithm = model["algorithm"]
        status = "Trained"

    return jsonify({

        "success": True,

        "dataset_name": os.path.basename(dataset),

        "rows": analysis["rows"],

        "columns": analysis["columns"],

        "target": target,

        "best_model": algorithm,

        "training_status": status

    })


# ===============================
# Analysis
# ===============================

@app.route("/analysis", methods=["GET"])
def analysis():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({

            "success": False,

            "message": "No dataset uploaded."

        }), 400

    return jsonify(
        analyze_data(dataset)
    )


# ===============================
# Preview
# ===============================

@app.route("/preview", methods=["GET"])
def preview():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({

            "success": False,

            "message": "No dataset uploaded."

        }), 400

    df = load_data(dataset)

    return jsonify({

        "success": True,

        "dataset_name": os.path.basename(dataset),

        "total_rows": len(df),

        "total_columns": len(df.columns),

        "columns": df.columns.tolist(),

        "preview": df.head(10).to_dict(
            orient="records"
        )

    })
# ===============================
# Train Model
# ===============================

@app.route("/train", methods=["POST"])
def train():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({
            "success": False,
            "message": "No dataset uploaded."
        }), 400

    data = request.get_json()

    target = data.get("target")

    if not target:

        return jsonify({
            "success": False,
            "message": "Target column is required."
        }), 400

    try:

        result = train_model(
            dataset,
            target
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ===============================
# Model Information
# ===============================

@app.route("/model-info", methods=["GET"])
def model_info():

    model = get_model()

    if model is None:

        return jsonify({

            "success": False,

            "message": "No trained model found."

        }), 400

    return jsonify({

        "success": True,

        "algorithm": model["algorithm"],

        "target": model["target"],

        "features": model["features"]

    })


# ===============================
# Prediction
# ===============================

@app.route("/predict", methods=["POST"])
def prediction():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({

            "success": False,

            "message": "No dataset uploaded."

        }), 400

    model = get_model()

    if model is None:

        return jsonify({

            "success": False,

            "message": "Please train a model first."

        }), 400

    data = request.get_json()

    if not data:

        return jsonify({

            "success": False,

            "message": "Invalid request."

        }), 400

    try:

        result = predict(data)

        return jsonify({

            "success": True,

            "prediction": result

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500
    # ===============================
# Charts
# ===============================

@app.route("/charts", methods=["GET"])
def charts():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({
            "success": False,
            "message": "No dataset uploaded."
        }), 400

    try:

        return jsonify({

            "success": True,

            "charts": get_chart_data(dataset)

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ===============================
# Download CSV
# ===============================

@app.route("/download-csv", methods=["GET"])
def download_csv():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({

            "success": False,

            "message": "No dataset uploaded."

        }), 404

    # ==========================
    # Load Original Dataset
    # ==========================

    df = load_data(dataset)

    # ==========================
    # Clean Dataset
    # ==========================

    df = clean_data(df)

    # ==========================
    # Create Reports Folder
    # ==========================

    os.makedirs("reports", exist_ok=True)

    clean_path = os.path.join(

        "reports",

        "Preprocessed_Dataset.csv"

    )

    # ==========================
    # Save Clean Dataset
    # ==========================

    df.to_csv(

        clean_path,

        index=False

    )

    # ==========================
    # Download Clean Dataset
    # ==========================

    return send_file(

        clean_path,

        as_attachment=True,

        download_name="Preprocessed_Dataset.csv"

    )

# ===============================
# Download PDF Report
# ===============================

@app.route("/download-report", methods=["GET"])
def download_report():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({

            "success": False,

            "message": "No dataset uploaded."

        }), 404

    analysis = analyze_data(dataset)

    model = get_model()

    target = "Not Trained"

    algorithm = "Not Trained"

    if model:

        target = model["target"]

        algorithm = model["algorithm"]

    pdf_path = generate_report(

        dataset_name=os.path.basename(dataset),

        rows=analysis["rows"],

        columns=analysis["columns"],

        target=target,

        algorithm=algorithm,

        missing_values=analysis["missing_values"],

        duplicate_rows=analysis["duplicate_rows"],

        numeric_columns=analysis["numeric_columns"],

        categorical_columns=analysis["categorical_columns"],

        total_sales=analysis["total_sales"],

        average_sales=analysis["average_sales"],

        max_sales=analysis["max_sales"],

        min_sales=analysis["min_sales"],

        total_profit=analysis["total_profit"],

        average_profit=analysis["average_profit"],

        total_quantity=analysis["total_quantity"],

        average_discount=analysis["average_discount"]

    )

    return send_file(

        pdf_path,

        as_attachment=True,

        download_name="AI_Business_Intelligence_Report.pdf"

    )

@app.route("/ai-insights", methods=["GET"])
def ai_insights():

    dataset = get_user_dataset()

    if dataset is None:

        return jsonify({

            "success": False,

            "message": "No dataset uploaded."

        }), 404

    result = analyze_data(dataset)

    return jsonify({

        "success": True,

        "dataset_quality": result["dataset_quality"],

        "ai_score": result["ai_score"],

        "highest_sales_region": result["highest_sales_region"],

        "highest_sales_value": result["highest_sales_value"],

        "highest_profit_category": result["highest_profit_category"],

        "highest_profit_value": result["highest_profit_value"],

        "lowest_profit_category": result["lowest_profit_category"],

        "lowest_profit_value": result["lowest_profit_value"],

        "best_selling_category": result["best_selling_category"],

        "insights": result["insights"],

        "recommendations": result["recommendations"]

    })

# ===============================
# Run Flask
# ===============================

if __name__ == "__main__":

    port = int(

        os.environ.get(

            "PORT",

            5000

        )

    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )
    