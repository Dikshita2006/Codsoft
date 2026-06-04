import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Select columns
X = data[["Pclass", "Age", "Fare"]]

# Output column
y = data["Survived"]

# Fill missing Age values
X["Age"] = X["Age"].fillna(X["Age"].mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

# Predict passenger survival
prediction = model.predict([[3, 22, 7.25]])

print("Prediction:", prediction)
