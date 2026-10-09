
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
                    6, 7, 3, 5, 8, 9, 1, 3, 6, 9],
    "exam_score": [20, 25, 35, 40, 50, 60, 65, 80, 30, 45,
                   62, 72, 32, 55, 70, 88, 18, 38, 58, 90]
}

df = pd.DataFrame(data)

print("First five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()


X = df[["study_hours"]]
y = df["exam_score"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


hours = float(input("\nEnter study hours: "))
prediction = model.predict(pd.DataFrame({"study_hours": [hours]}))
print("Predicted exam score:", round(prediction[0], 2))

plt.scatter(X["study_hours"], y, label="Actual scores")
plt.plot(X["study_hours"], model.predict(X), label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Student Performance Prediction")
plt.legend()
plt.show()