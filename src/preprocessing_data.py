import pandas as pd
import numpy as np


diet = pd.read_sas('Data/src_data/DR1TOT_L.xpt', format='xport')
body = pd.read_sas('Data/src_data/BMX_L.xpt', format='xport')
demo = pd.read_sas('Data/src_data/DEMO_L.xpt', format='xport')


# Only keep first day dietary recall
diet = diet[diet['DR1DRSTZ'] == 1]

diet_cols = [
    'SEQN',
    'WTDRD1',    # Dietary sample weight
    'DR1TKCAL',   # Calories
    'DR1TPROT',   # Protein (g)
    'DR1TSUGR',   # Total sugar (g)
    'DR1TFIBE',   # Fiber (g)
    'DR1TSFAT',   # Saturated fat (g)
    'DR1TSODI'    # Sodium (mg)
]

diet = diet[diet_cols]

# Drop missing calories and body measurements
diet = diet.dropna(subset=['DR1TKCAL'])
body = body[['SEQN', 'BMXWT', 'BMXHT']].dropna()

# Compute BMI
body['BMI'] = body['BMXWT'] / ((body['BMXHT'] / 100) ** 2)

# Create obesity label
body['Obese'] = (body['BMI'] >= 30).astype(int)

# Keep only necessary demographic columns and drop missing values
demo = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].dropna()


# Rename for clarity
demo.rename(columns={'RIDAGEYR': 'Age','RIAGENDR': 'Sex'}, inplace=True)

df = diet.merge(body, on='SEQN', how='inner')
df = df.merge(demo, on='SEQN', how='inner')

# Nutrients per 1000 kcal
# We compute these to normalize for total calorie intake, which can vary widely between individuals and may mess with the relationship between specific nutreitns and obesity.
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
    'Obese',
    'DR1TKCAL',
    'Sugar_per_1000kcal',
    'Fiber_per_1000kcal',
    'Sodium_per_1000kcal',
    'SatFat_per_1000kcal',
    'Protein_per_1000kcal'
]

df_final = df[final_cols]

# Remove extreme calorie outliers 
df_final = df_final[(df_final['DR1TKCAL'] > 500) & (df_final['DR1TKCAL'] < 6000)]

# Drop remaining missing values
df_final = df_final.dropna()

df_final.to_csv("Data/prepro_data/nhanes_diet_risk.csv", index=False)

# For debugging
# print("Final dataset shape:", df_final.shape)
print("Saved to Data/prepro_data/nhanes_diet_risk.csv")
