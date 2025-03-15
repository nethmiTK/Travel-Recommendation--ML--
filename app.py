import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(file_name):
    file_path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_name} not found in {BASE_DIR}")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load models (example)
linear_regression_model = load_model('linear_regression.pkl')
# Do the same for logistic_regression.pkl, decision_tree.pkl, svm.pkl, scaler.pkl

def predict_experience(user_id, destination_id, visit_year, visit_month, visit_day):
    if not all([linear_regression_model]):  # Add other models here
        raise Exception("❌ One or more models failed to load. Please check your .pkl files.")
    
    # Example prediction (adjust for your use case)
    input_features = [[user_id, destination_id, visit_year, visit_month, visit_day]]
    
    # You might need to scale the input or transform it
    # prediction = linear_regression_model.predict(input_features)
    
    return {
        "linear_regression": "Sample result",
        "logistic_regression": "Sample result",
        "decision_tree": "Sample result",
        "svm": "Sample result"
    }
