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
# Add other models like logistic_regression.pkl, decision_tree.pkl, svm.pkl, scaler.pkl

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

    if not all([linear_regression_model]):  # Add other models here
        return "❌ One or more models failed to load. Please check your .pkl files."

    # Example prediction (adjust for your use case)
    input_features = [[user_id, destination_id, visit_year, visit_month, visit_day]]

    # You might need to scale the input or transform it
    prediction = linear_regression_model.predict(input_features)

    return render_template('index.html', prediction_result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
