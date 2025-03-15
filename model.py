import pandas as pd
import pickle
import os

# Get absolute paths for model files
base_dir = os.path.dirname(__file__)  # Get the directory where this script is located

# Function to load model safely
def load_model(filename):
    try:
        filepath = os.path.join(base_dir, filename)
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found in {base_dir}")
        return None
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

# Load trained models
lr_model = load_model("linear_regression.pkl")
log_model = load_model("logistic_regression.pkl")
dt_model = load_model("decision_tree.pkl")
svm_model = load_model("svm.pkl")
scaler = load_model("scaler.pkl")

# Ensure models are loaded before proceeding
if not all([lr_model, log_model, dt_model, svm_model, scaler]):
    raise Exception("❌ One or more models failed to load. Please check your .pkl files.")

# Prediction function
def predict_experience(user_id, destination_id, visit_year, visit_month, visit_day):
    # Prepare input data
    user_input = pd.DataFrame([[user_id, destination_id, visit_year, visit_month, visit_day]],
                              columns=['UserID', 'DestinationID', 'VisitYear', 'VisitMonth', 'VisitDay'])

    # Scale input data
    user_input_scaled = scaler.transform(user_input)  # Apply the same scaler used in training

    # Predict using all models
    try:
        lr_prediction = round(lr_model.predict(user_input_scaled)[0], 2)
        log_prediction = log_model.predict(user_input_scaled)[0]
        log_result = "Good (4-5)" if log_prediction == 1 else "Bad (1-3)"
        dt_prediction = dt_model.predict(user_input_scaled)[0]
        svm_prediction = svm_model.predict(user_input_scaled)[0]
    except Exception as e:
        return {"error": f"❌ Prediction failed: {e}"}

    return {
        "Linear Regression": lr_prediction,
        "Logistic Regression": log_result,
        "Decision Tree": dt_prediction,
        "SVM": svm_prediction
    }
