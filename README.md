# Dietary Pattern–Informed Recommendation System (CS179 Design Project)

Team **MRTY** (Aidan Ciesla, Timothy Tse, Yuzhe Lin, Matthew Chung) is building a dietary pattern–informed recommendation system that **groups users into real-world eating patterns** and then provides **actionable lifestyle recommendations** based on common strengths and gaps observed in the most similar NHANES participants.

This project uses the **CDC NHANES 2021–2023** cycle and focuses on clustering individuals based on reported dietary intake (e.g., total nutrients and/or individual foods). 

---

## What this project does

1. **Ingests NHANES dietary data** (2021–2023 cycle) and prepares it for analysis.  
2. **Builds a nutritionally meaningful representation** (e.g., nutrient totals/densities) suitable for distance-based clustering.
3. **Reduces dimensionality** (PCA) to avoid clustering in 100+ dimensions.
4. **Clusters individuals into dietary patterns** using **K-means in PCA-reduced nutrient space**, selecting K using elbow + silhouette. 
5. **Interprets clusters** by analyzing centroid nutrient profiles and comparing against public dietary guidance to produce cluster-level “strengths/gaps” and recommendations.
6. **Assigns a user to a subgroup** by comparing their inputs to the learned clusters and returns tailored suggestions. 

---

## Why dietary patterns (instead of generic advice)?

Common diet guidance is often population-level (“eat more vegetables”) and doesn’t reflect how most individuals actually eat day-to-day. This project aims to discover **population-wide, real eating patterns** and provide **more realistic, pattern-aligned actions**. 

---

## Data sources

### Primary dataset
- **NHANES (August 2021 – August 2023)** dietary intake data.

### Files used (current + planned)
- Dietary totals: **DR1TOT_L** (and maybe DR2TOT_L) 
- Individual foods: **DR1IFF_L** (and maybe DR2IFF_L)
- Optional validation sources (future scope): body measures and lab data

### Weights (population-representative analysis)
- **TBD**

---

## Core components

### 1) Data ingestion
- Loads NHANES XPT files (dietary totals and optionally individual foods).
- Joins records by participant ID (`SEQN`).  

### 2) Preprocessing + feature selection
- Filters to reliable intakes (e.g., Day 1 reliable records).
- Selects a compact set of dietary variables that best capture nutrition patterns (e.g., energy, macro totals, sugar/fiber/sodium, key micronutrients).   
- Handles missingness + removes low-response variables to avoid instability.

### 3) Dimensionality reduction (PCA)
- Applies PCA to reduce the nutrient feature space before clustering.  

### 4) Clustering
- **K-means** in PCA-reduced nutrient space.  
- Chooses K using:
  - within-cluster sum of squares (elbow)
  - silhouette score for cohesion  

### 5) Cluster analysis + recommendation layer
- Summarizes clusters using centroid nutrient profiles.
- Translates imbalances into clear, realistic actions aligned with public dietary guidance. 

### 6) User classification (demo app / CLI)
- Collects basic user inputs and assigns the user to the nearest pattern cluster.
- Returns cluster-informed insights and recommendations. 

---

Data Source : https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&Cycle=2021-2023 ; DR1IFF_L Doc
