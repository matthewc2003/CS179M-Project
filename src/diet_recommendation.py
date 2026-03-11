from altair import value
import numpy as np
import joblib
import pandas as pd

# Healthy reference diet values
# Note there is a LOT of nuance here based on the user. For example, a very active person may have a higher calorie intake and thus higher absolute nutrient intakes, but their nutrient densities could still be healthy.
# In this case, we are using general population-level guidelines for a 2000 kcal diet, which is a common reference point. Future improvement could be to personalize this based on user characteristics.
healthy_reference = {
    "Sugar_per_1000kcal": (0, 25), # grams per 1000 https://www.fda.gov/food/nutrition-facts-label/added-sugars-nutrition-facts-label?utm_source=chatgpt.com
    "Fiber_per_1000kcal": (14, 40), # grams per 1000 https://www.health.harvard.edu/blog/should-i-be-eating-more-fiber-2019022115927#:~:text=Fiber:%20how%20much%20is%20enough,and%2030%20daily%20grams%2C%20respectively.
    "Sodium_per_1000kcal": (0, 1500), # mg per 1000 https://www.heart.org/en/healthy-living/healthy-eating/eat-smart/sodium/how-much-sodium-should-i-eat-per-day
    "SatFat_per_1000kcal": (0,10), # grams per 1000 https://odphp.health.gov/sites/default/files/2019-10/DGA_Cut-Down-On-Saturated-Fats.pdf
    "Protein_per_1000kcal": (30, 45) # grams per 1000 https://www.unitypoint.org/news-and-articles/how-much-protein-do-you-need-daily-ideal-protein-intake-for-muscle-growth-weight-loss-and-managing-chronic-conditions
}

healthy_direction = {
    "Sugar_per_1000kcal": "lower",
    "Fiber_per_1000kcal": "higher",
    "Sodium_per_1000kcal": "lower",
    "SatFat_per_1000kcal": "lower",
    "Protein_per_1000kcal": "higher"
}

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
    
    percentile_results = {
        nutrient: get_percentile_position(value, population_percentiles[nutrient])
        for nutrient, value in nutrient_values.items()
    }
    
    guideline_results = {}
    for nutrient, value in nutrient_values.items():
        direction = healthy_direction[nutrient]
        guideline_value = healthy_reference[nutrient]

        for nutrient, value in nutrient_values.items():

            lower, upper = healthy_reference[nutrient]

            if value < lower:
                status = "below recommended level"
            elif value > upper:
                status = "above recommended level"
            else:
                status = "within recommended range"

            guideline_results[nutrient] = status

    advice_lines = []

    advice_lines.append(
        "\nThese comparisons reflect population-level dietary patterns "
        "and are not medical diagnoses."
    )
    
    advice_lines.append("\nCompared to general dietary guidelines:")
    for nutrient, status in guideline_results.items():
        clean_name = nutrient.replace("_per_1000kcal", "")
        advice_lines.append(f"- {clean_name}: {status}")

    return {
        "cluster": cluster,
        "cluster_prevalence_percent": cluster_pct,
        "percentile_comparison": percentile_results,
        "advice_text": "\n".join(advice_lines),
        "guideline_comparison": guideline_results
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