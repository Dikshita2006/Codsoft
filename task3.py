import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("advertising.csv")
print(data.head())

plt.scatter(data["TV"], data["Sales"])
plt.show()

X = data[["TV"]]
y = data["Sales"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
