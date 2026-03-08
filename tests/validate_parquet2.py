import pandas as pd
df = pd.read_parquet('data/results/bourrin_v7.parquet')
print("COLONNES:", list(df.columns))
print("\nDTYPES:\n", df.dtypes)
print("\n5 premières lignes:")
print(df.head(2).to_string())