import pandas as pd
import pickle

# Load trained models
lr_model = pickle.load(open("linear_regression.pkl", "rb"))
log_model = pickle.load(open("logistic_regression.pkl", "rb"))
dt_model = pickle.load(open("decision_tree.pkl", "rb"))
svm_model = pickle.load(open("svm.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_experience(user_id, destination_id, visit_year, visit_month, visit_day):
    # Prepare input data
    user_input = pd.DataFrame([[user_id, destination_id, visit_year, visit_month, visit_day]],
                              columns=['UserID', 'DestinationID', 'VisitYear', 'VisitMonth', 'VisitDay'])
    user_input_scaled = scaler.transform(user_input)

    # Predict using all models
    lr_prediction = lr_model.predict(user_input)[0]
    log_prediction = log_model.predict(user_input)[0]
    log_result = "Good (4-5)" if log_prediction == 1 else "Bad (1-3)"
    dt_prediction = dt_model.predict(user_input)[0]
    svm_prediction = svm_model.predict(user_input_scaled)[0]

    return {
        "Linear Regression": round(lr_prediction, 2),
        "Logistic Regression": log_result,
        "Decision Tree": dt_prediction,
        "SVM": svm_prediction
    }
