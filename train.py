import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv("reliance_clean.csv")

# Clean data
data = data.iloc[2:]
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data = data.dropna()

# Create Day column
data["Day"] = range(len(data))

X = data[["Day"]]
y = data["Close"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("R2 Score:", score)

# Graph
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

plt.show()