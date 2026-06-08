import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("reliance_clean.csv")

data = data.iloc[2:]
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data = data.dropna()

data["Day"] = range(len(data))

X = data[["Day"]]
y = data["Close"]

model = LinearRegression()
model.fit(X, y)

future_day = [[len(data) + 7]]

prediction = model.predict(future_day)

print("Predicted Price After 7 Days:")
print(prediction[0])