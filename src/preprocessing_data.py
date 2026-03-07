import pandas as pd
import numpy as np
# Load NHANES Data
diet = pd.read_sas('Data/src_data/DR1TOT_L.xpt', format='xport')
demo = pd.read_sas('Data/src_data/DEMO_L.xpt', format='xport')

# keep only first day dietary recall (DR1DRSTZ == 1)
diet = diet[diet['DR1DRSTZ'] == 1]

diet_cols = [
    'SEQN',
    'WTDRD1',
    'DR1TKCAL',
    'DR1TPROT',
    'DR1TSUGR',
    'DR1TFIBE',
    'DR1TSFAT',
    'DR1TSODI'
]

# drop rows with missing calories or weights
diet = diet[diet_cols]
diet = diet.dropna(subset=['DR1TKCAL'])

# keep only relevant demographic columns
demo = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].dropna()
demo.rename(columns={'RIDAGEYR': 'Age', 'RIAGENDR': 'Sex'}, inplace=True)

#merge datasets on SEQN
df = diet.merge(demo, on='SEQN', how='inner')

#remove implausible calorie intakes
df = df[(df['DR1TKCAL'] > 500) & (df['DR1TKCAL'] < 6000)]

nutrient_cols = [
    'DR1TPROT',
    'DR1TSUGR',
    'DR1TFIBE',
    'DR1TSFAT',
    'DR1TSODI'
]

df = df.dropna(subset=nutrient_cols)

# convert to nutrient densities per 1000 kcal
# essential for clustering and regression models to be calorie-adjusted (oterhwise would just cluster by calorie intake)
df['Sugar_per_1000kcal'] = df['DR1TSUGR'] / df['DR1TKCAL'] * 1000
df['Fiber_per_1000kcal'] = df['DR1TFIBE'] / df['DR1TKCAL'] * 1000
df['Sodium_per_1000kcal'] = df['DR1TSODI'] / df['DR1TKCAL'] * 1000
df['SatFat_per_1000kcal'] = df['DR1TSFAT'] / df['DR1TKCAL'] * 1000
df['Protein_per_1000kcal'] = df['DR1TPROT'] / df['DR1TKCAL'] * 1000

final_cols = [
    'SEQN',
    'WTDRD1',
    'Age',
    'Sex',
    'DR1TKCAL',
    'Sugar_per_1000kcal',
    'Fiber_per_1000kcal',
    'Sodium_per_1000kcal',
    'SatFat_per_1000kcal',
    'Protein_per_1000kcal'
]

df_final = df[final_cols].dropna()

# Apply survey weights normalization
weights = df_final['WTDRD1']
weights = weights / weights.mean()

population_means = {
    col: float(np.average(df_final[col], weights=weights))
    for col in [
        'Sugar_per_1000kcal',
        'Fiber_per_1000kcal',
        'Sodium_per_1000kcal',
        'SatFat_per_1000kcal',
        'Protein_per_1000kcal'
    ]
}

# Compute weighted percentiles for population distribution
def weighted_percentile(values, percentiles, sample_weight):
    values = np.asarray(values)
    sample_weight = np.asarray(sample_weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cumulative_weight = np.cumsum(sample_weight) # cumulative sum of weights
    total_weight = cumulative_weight[-1]

    percentile_values = []
    for p in percentiles:
        cutoff = p / 100 * total_weight
        percentile_values.append(values[cumulative_weight >= cutoff][0]) # find the value where cumulative weight exceeds cutoff

    return percentile_values

population_percentiles = {}


# Compute percentiles for each nutrient density
for col in [
    'Sugar_per_1000kcal',
    'Fiber_per_1000kcal',
    'Sodium_per_1000kcal',
    'SatFat_per_1000kcal',
    'Protein_per_1000kcal'
]:
    population_percentiles[col] = weighted_percentile(
        df_final[col],
        [10, 25, 50, 75, 90],
        weights
    )

df_final.to_pickle("Data/prepro_data/diet_modeling_dataset.pkl")




# Save population stats artifact for use in app
import joblib
model_features = [
    'Sugar_per_1000kcal',
    'Fiber_per_1000kcal',
    'Sodium_per_1000kcal',
    'SatFat_per_1000kcal',
    'Protein_per_1000kcal'
]

artifact = {
    "data_path": "Data/prepro_data/diet_modeling_dataset.pkl",
    "population_means": population_means,
    "population_percentiles": population_percentiles,
    "model_features": model_features
}

joblib.dump(artifact, "Data/prepro_data/population_stats.pkl")


print("Saved to Data/prepro_data/diet_modeling_dataset.pkl")
print("Final dataset shape:", df_final.shape)