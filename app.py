from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(file_name):
    file_path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_name} not found in {BASE_DIR}")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load models
linear_regression_model = load_model('linear_regression.pkl')
logistic_regression_model = load_model('logistic_regression.pkl')
decision_tree_model = load_model('decision_tree.pkl')
svm_model = load_model('svm.pkl')
scaler = load_model('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_id = int(request.form['user_id'])
        destination_id = int(request.form['destination_id'])
        visit_year = int(request.form['visit_year'])
        visit_month = int(request.form['visit_month'])
        visit_day = int(request.form['visit_day'])
    except ValueError:
        return "❌ Error: Invalid input. Please enter valid numerical values."

    if not all([linear_regression_model, logistic_regression_model, decision_tree_model, svm_model, scaler]):
        return "❌ One or more models failed to load."

    input_features = [[user_id, destination_id, visit_year, visit_month, visit_day]]
    
    # Example scaling (optional, depends on your model)
    # scaled_input = scaler.transform(input_features)

    # Example predictions
    linear_pred = linear_regression_model.predict(input_features)[0]
    logistic_pred = logistic_regression_model.predict(input_features)[0]
    decision_tree_pred = decision_tree_model.predict(input_features)[0]
    svm_pred = svm_model.predict(input_features)[0]

    prediction_result = {
        'Linear Regression': linear_pred,
        'Logistic Regression': logistic_pred,
        'Decision Tree': decision_tree_pred,
        'SVM': svm_pred
    }

    print("✅ Predictions:", prediction_result)

    return render_template('index.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
