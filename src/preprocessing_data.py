# Load Library
import pandas as pd

# Load Data
diet_food_day_1 = pd.read_sas('Data/src_data/DR1IFF_L.xpt', format='xport')
diet_nutrient_day_1 = pd.read_sas('Data/src_data/DR1TOT_L.xpt', format='xport')
body_measures = pd.read_sas('Data/src_data/BMX_L.xpt', format='xport')

# Preprocessing diet_nutrient_day_1

# 1. Only keep data that is reliable and meets the minimum criteria (DR1DRSTZ = 1)
diet_nutrient_day_1 = diet_nutrient_day_1[diet_nutrient_day_1['DR1DRSTZ'] == 1]

# 2. Drop useless columns
cols_to_drop = ['DR1EXMER', 'DR1DAY', 'DR1LANG', 'DR1MRESP', 'DR1HELP']
diet_nutrient_day_1.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 3. Drop columns with high missing values
missing_rate = 0.8
missing_fractions = diet_nutrient_day_1.isnull().mean()
cols_to_drop = missing_fractions[missing_fractions > missing_rate].index.tolist()
diet_nutrient_day_1.drop(columns=cols_to_drop, inplace=True)

# 4. Append BMXWT from BMX_L (Only keep the rows with BMXWT)
body_measures_weight = body_measures[['SEQN', 'BMXWT']].copy()
body_measures_weight.dropna(subset=['BMXWT'], inplace=True)
diet_nutrient_day_1 = pd.merge(diet_nutrient_day_1, body_measures_weight, on='SEQN', how='inner')

# 5. There are a few columns we want to use for our minimal viable product (MVP) model. We will keep those columns and drop the rest.
cols_to_keep = ['SEQN', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT', 
                'DR1TSFAT', 'DR1TMFAT', 'DR1TPFAT', 'DR1TSODI', 'DR1TCALC',
                'DR1TPOTA', 'DR1TIRON','DR1DRSTZ', 'BMXWT']
diet_nutrient_day_1 = diet_nutrient_day_1[cols_to_keep]
diet_nutrient_day_1.to_csv("Data/prepro_data/DR1TOT_L.csv", index=False)

# apply PCA to the DR1TOT_L dataset and keep the top 5 principal components
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler
# Standardize the data
features = diet_nutrient_day_1.drop(columns=['SEQN', 'DR1DRSTZ'])
features_scaled = StandardScaler().fit_transform(features)
# Apply PCA
pca = PCA(n_components=5)
principal_components = pca.fit_transform(features_scaled)
# Create a DataFrame with the principal components
pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)
# Add SEQN and DR1DRSTZ back to the PCA DataFrame
pca_df['SEQN'] = diet_nutrient_day_1['SEQN'].values
pca_df['DR1DRSTZ'] = diet_nutrient_day_1['DR1DRSTZ'].values
# Save the PCA DataFrame to a CSV file
pca_df.to_csv("Data/prepro_data/DR1TOT_L_PCA.csv", index=False)
