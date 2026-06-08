import yfinance as yf

data = yf.download("RELIANCE.NS",
                   start="2020-01-01",
                   end="2025-01-01",
                   auto_adjust=True)

# Keep only Close price
data = data[["Close"]]

data.to_csv("reliance_clean.csv")

print(data.head())
print("Clean data saved!")