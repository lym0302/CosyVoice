import pandas as pd

df = pd.read_parquet("data/test/parquet/parquet_000000000.tar")
print(df.head())
print(df.columns)
print(df.iloc[0])

