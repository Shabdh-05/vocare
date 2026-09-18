import sys
print("Script started", flush=True)
import pandas as pd
print("Pandas imported", flush=True)
df = pd.read_csv(r"c:\Users\msdha\OneDrive\Desktop\health project\Vocal_Disorder_Detection\vocal_features.csv", nrows=10)
print(f"Columns: {len(df.columns)}, rows read: {len(df)}", flush=True)
