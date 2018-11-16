import pandas as pd

df = pd.read_csv("database", sep = '\t', index_col = 0)
df = df.round(3)
