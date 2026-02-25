import numpy as np
import joblib
import pandas as pd

clustering_artifact = joblib.load(
    "Data/prepro_data/diet_clustering_model.pkl"
)

population_artifact = joblib.load(
    "Data/prepro_data/population_stats.pkl"
)

pipeline = clustering_artifact["pipeline"]
cluster_features = clustering_artifact["cluster_features"]
cluster_profiles = clustering_artifact["cluster_profiles"]
cluster_distribution = clustering_artifact["cluster_distribution"]

population_means = population_artifact["population_means"]
population_percentiles = population_artifact["population_percentiles"]

def get_percentile_position(value, percentile_list):
    """
    Given a value and a list of percentiles, return a string indicating the percentile band.
    For example, if percentile_list = [10, 25, 50, 75, 90], then:
    - value < 10th percentile: "<10th percentile"
    """
    p10, p25, p50, p75, p90 = percentile_list

    if value < p10:
        return "<10th percentile"
    elif value < p25:
        return "10th-25th percentile"
    elif value < p50:
        return "25th-50th percentile"
    elif value < p75:
        return "50th-75th percentile"
    elif value < p90:
        return "75th-90th percentile"
    else:
        return ">90th percentile"

def generate_recommendation(
    calories,
    sugar_g,
    fiber_g,
    sodium_mg,
    satfat_g,
    protein_g
):

    """
    Takes raw daily intake values and returns:
    - Cluster assignment
    - Percentile comparisons
    - Population-level advice
    """
    sugar_density = sugar_g / calories * 1000
    fiber_density = fiber_g / calories * 1000
    sodium_density = sodium_mg / calories * 1000
    satfat_density = satfat_g / calories * 1000
    protein_density = protein_g / calories * 1000

    user_vector = pd.DataFrame([{
        "Sugar_per_1000kcal": sugar_density,
        "Fiber_per_1000kcal": fiber_density,
        "Sodium_per_1000kcal": sodium_density,
        "SatFat_per_1000kcal": satfat_density,
        "Protein_per_1000kcal": protein_density
    }])
    
    cluster = int(pipeline.predict(user_vector)[0])
    cluster_pct = cluster_distribution[cluster] * 100

    nutrient_values = user_vector.iloc[0].to_dict()
    percentile_results = {}

    for nutrient, value in nutrient_values.items():
        percentile_band = get_percentile_position(
            value,
            population_percentiles[nutrient]
        )
        percentile_results[nutrient] = percentile_band

    advice_lines = []

    advice_lines.append(
        f"\nYour dietary pattern most closely resembles Cluster {cluster}, "
        f"which represents approximately {cluster_pct:.1f}% of U.S. adults."
    )

    advice_lines.append("\nCompared to the U.S. adult population:")

    for nutrient, band in percentile_results.items():
        clean_name = nutrient.replace("_per_1000kcal", "")
        advice_lines.append(f"- {clean_name}: {band}")

    advice_lines.append(
        "\nThese comparisons reflect population-level dietary patterns "
        "and are not medical diagnoses."
    )

    return {
        "cluster": cluster,
        "cluster_prevalence_percent": cluster_pct,
        "percentile_comparison": percentile_results,
        "advice_text": "\n".join(advice_lines),
    }


# debug
if __name__ == "__main__":

    result = generate_recommendation(
        calories=2200,
        sugar_g=90,
        fiber_g=18,
        sodium_mg=3200,
        satfat_g=25,
        protein_g=80
    )

    print(result["advice_text"])