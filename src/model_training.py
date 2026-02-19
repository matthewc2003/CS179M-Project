# For visualization, let's use seaborn and matplotlib to create some plots of DR1TOT_L_PCA.csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the PCA data
pca_df = pd.read_csv("Data/prepro_data/DR1TOT_L_PCA.csv")
# Create a pairplot of the principal components colored by DR1DRSTZ 
sns.pairplot(pca_df, vars=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], hue='DR1DRSTZ')
plt.suptitle("Pairplot of Principal Components Colored by DR1DRSTZ", y=1.02)
plt.show()

# Now we need to figure out a good K for K-means clustering. We can use the elbow method to do this.
from sklearn.cluster import KMeans
# Elbow method to find the optimal number of clusters
wcss = []