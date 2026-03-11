# Dietary Pattern–Informed Recommendation System (CS179 Design Project)

Team **MRTY** (Aidan Ciesla, Timothy Tse, Yuzhe Lin, Matthew Chung) is building a dietary pattern–informed recommendation system that **groups users into real-world eating patterns** and then provides **actionable lifestyle recommendations** based on common strengths and gaps observed in the most similar NHANES participants.

This project uses the **CDC NHANES 2021–2023** cycle and focuses on clustering individuals based on reported dietary intake (e.g., total nutrients and/or individual foods). 

---

## What this project does

1. **Ingests NHANES dietary data** (2021–2023 cycle) and prepares it for analysis.  
2. **Builds a nutritionally meaningful representation** (e.g., nutrient totals/densities) suitable for distance-based clustering.
3. **Normalizes data** for accurate representation and comparison.
4. **Clusters individuals into dietary patterns** using **K-means in standardized nutrient space**, selecting K using silhouette score. 
5. **Interprets clusters** by analyzing centroid nutrient profiles and comparing against public dietary guidance to produce cluster-level “strengths/gaps” and recommendations.
6. **Assigns a user to a subgroup** by comparing their inputs to the learned clusters and returns data-driven suggestions. 

---

## Why dietary patterns (instead of generic advice)?

Most Americans don't know how their diet compares to others and assume their diet is healthier than the average person. This project aims to discover **population-wide, real eating patterns** and provide **more realistic, pattern-aligned actions** alongside **comparisons to the average U.S. person**. 

---

## Data sources

### Primary dataset
- **NHANES (August 2021 – August 2023)** dietary intake data.

### Files used (current + planned)
- Dietary totals: **DR1TOT_L**
- Individual foods: **DR1IFF_L**
- Planned: Optional validation sources (future scope) - body measures and lab data

---

## Core components

### 1) Data ingestion
- Loads NHANES XPT files (dietary totals and optionally individual foods).
- Joins records by participant ID (`SEQN`).  

### 2) Preprocessing + feature selection
- Filters to reliable intakes (e.g., Day 1 reliable records).
- Selects a compact set of dietary variables that best capture nutrition patterns (e.g., energy, macro totals, sugar/fiber/sodium, key micronutrients).   
- Handles missingness + removes low-response variables to avoid instability.

### 3) Normalization
- Converst nutrients to densities and use z-scores to normalize feature lenghts.  

### 4) Clustering
- **K-means** in PCA-reduced nutrient space.  
- Chooses K based on sillhouette score.

### 5) Cluster analysis + recommendation layer
- Summarizes clusters using centroid nutrient profiles.
- Translates imbalances into clear, realistic actions aligned with public dietary guidance. 

### 6) User classification (demo app / CLI)
- Collects basic user inputs and assigns the user to the nearest pattern cluster.
- Returns cluster-informed insights and recommendations. 

---

Data Source : https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&Cycle=2021-2023
