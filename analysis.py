import pandas as pd
import matplotlib.pyplot as plt

# Read file
data = pd.read_csv("reliance_clean.csv")

# Remove first 2 rows (Price and Ticker)
data = data.iloc[2:]

# Convert Close column to numbers
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

# Remove empty values
data = data.dropna()

print(data.head())

# Plot graph
plt.figure(figsize=(10,5))
plt.plot(data["Close"])

plt.title("Reliance Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")

plt.show()