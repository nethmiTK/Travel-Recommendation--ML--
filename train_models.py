import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load dataset (update the filename if necessary)
df = pd.read_csv("your_dataset.csv")

# Select features (X) and target (y)
X = df[['UserID', 'DestinationID', 'VisitYear', 'VisitMonth', 'VisitDay']]
y = df['ExperienceRating']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Save models
pickle.dump(lr_model, open("linear_regression.pkl", "wb"))
pickle.dump(log_model, open("logistic_regression.pkl", "wb"))
pickle.dump(dt_model, open("decision_tree.pkl", "wb"))
pickle.dump(svm_model, open("svm.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Models saved successfully!")
