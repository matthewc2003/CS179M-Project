import pandas as pd

df = pd.read_csv("Data/prepro_data/nhanes_diet_risk.csv")

features = [
    "Sugar_per_1000kcal",
    "Fiber_per_1000kcal",
    "Sodium_per_1000kcal",
    "SatFat_per_1000kcal",
    "Protein_per_1000kcal"
]

percentile_data = {}

for f in features:
    percentile_data[f] = df[f].dropna().values

# Save
import joblib
joblib.dump(percentile_data, "Data/prepro_data/percentile_data.pkl")