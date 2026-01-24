import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("heart_database.csv")

print(data.head())

print("\nMissing Values:\n", data.isnull().sum())

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

patient_data = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])

sample = scaler.transform(patient_data)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nHeart Disease Detected")
else:
    print("\nNo Heart Disease Detected")
