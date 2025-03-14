from flask import Flask, render_template, request
from model import predict_experience

app = Flask(__name__)

# Route to render the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    user_id = int(request.form['user_id'])
    destination_id = int(request.form['destination_id'])
    visit_year = int(request.form['visit_year'])
    visit_month = int(request.form['visit_month'])
    visit_day = int(request.form['visit_day'])

    predictions = predict_experience(user_id, destination_id, visit_year, visit_month, visit_day)

    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
