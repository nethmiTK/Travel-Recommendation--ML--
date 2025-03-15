import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example dataset
data = {
    'user_id': [1, 2, 3, 4, 5],
    'destination_id': [10, 20, 30, 40, 50],
    'visit_year': [2020, 2021, 2022, 2023, 2024],
    'visit_month': [1, 2, 3, 4, 5],
    'visit_day': [15, 10, 5, 20, 25],
    'experience_score': [3.5, 4.0, 2.5, 4.5, 3.0]  # Target/label
}

df = pd.DataFrame(data)

# Features and target
X = df[['user_id', 'destination_id', 'visit_year', 'visit_month', 'visit_day']]
y = df['experience_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('linear_regression.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ linear_regression.pkl has been created successfully!")
