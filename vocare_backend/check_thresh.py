import os
import sys
import joblib
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
thresh_path = os.path.join(base_dir, "avhfm_threshold.pkl")
if os.path.exists(thresh_path):
    thresh = joblib.load(thresh_path)
    print(f"Saved threshold: {thresh}")
else:
    print("No threshold file found")
