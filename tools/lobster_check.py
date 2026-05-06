import pandas as pd
df = pd.read_csv('lobbook.csv')
price_min = df.min().min()  # across all price columns
price_max = df.max().max()
print(f"Range: {price_min} to {price_max}")
print(f"Levels at $0.01 tick: {int((price_max - price_min) / 0.01) + 1}")
