import sys
for pkg in ["xgboost", "lightgbm", "catboost", "sklearn", "scipy"]:
    try:
        __import__(pkg)
        print(f"{pkg}: available", flush=True)
    except ImportError:
        print(f"{pkg}: NOT available", flush=True)
