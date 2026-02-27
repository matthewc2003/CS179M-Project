import streamlit as st
from diet_recommendation import generate_recommendation
import random

st.set_page_config(page_title="Diet Pattern Demo", layout="centered")

st.title("Population-Based Dietary Pattern Intelligence")
st.markdown(
    """
This system compares a user's dietary intake
to nationally representative U.S. dietary patterns
derived from NHANES data.

It provides **population-based insights**, not medical diagnoses.
"""
)



# TODO: Make these questions actually influence the intake estimation and cluster assignment in the final version. For now, they are static for demo purposes.

st.divider()

st.header("Dietary Screening Questionnaire")

st.markdown("These questions help contextualize intake patterns.")

fruit_freq = st.selectbox(
    "How often do you consume fruits?",
    ["Rarely/Never", "1–2 times per week", "3–6 times per week", "Daily"]
)

veg_freq = st.selectbox(
    "How often do you consume vegetables?",
    ["Rarely/Never", "1–2 times per week", "3–6 times per week", "Daily"]
)

processed_food = st.selectbox(
    "How often do you consume highly processed foods?",
    ["Rarely/Never", "1–2 times per week", "3–6 times per week", "Daily"]
)

sugary_drinks = st.selectbox(
    "How often do you consume sugar-sweetened beverages?",
    ["Rarely/Never", "1–2 times per week", "3–6 times per week", "Daily"]
)

st.caption(
    "Note: In the final version, questionnaire responses will assist "
    "in estimating intake values and improving user input accuracy."
)

st.divider()

st.header("Daily Intake Estimation")

calories = st.number_input("Calories (kcal)", min_value=500, max_value=6000, value=2000)
sugar = st.number_input("Total Sugar (grams)", min_value=0.0, value=80.0)
fiber = st.number_input("Fiber (grams)", min_value=0.0, value=20.0)
sodium = st.number_input("Sodium (mg)", min_value=0.0, value=3000.0)
satfat = st.number_input("Saturated Fat (grams)", min_value=0.0, value=25.0)
protein = st.number_input("Protein (grams)", min_value=0.0, value=75.0)

st.divider()

if st.button("Generate Personalized Insight"):

    if calories <= 0:
        st.error("Calories must be greater than zero.")
    else:

        with st.spinner("Analyzing dietary pattern..."):

            result = generate_recommendation(
                calories=calories,
                sugar_g=sugar,
                fiber_g=fiber,
                sodium_mg=sodium,
                satfat_g=satfat,
                protein_g=protein
            )


        st.subheader("Personalized Dietary Pattern Report")

# TODO: Dynamically generate cluster assignment and prevalence based on user's input and the clustering model.
# IMPORTANT: WE CURRENTLY HARD CODE THE CLUSTER ASSIGNMENT AND PREVALENCE FOR DEMO PURPOSES. IN THE FINAL VERSION, THESE WILL BE DYNAMICALLY GENERATED BASED ON THE USER'S INPUT AND THE CLUSTERING MODEL.

        st.markdown(f"""

### Dietary Pattern Classification
You most closely resemble **Cluster {result['cluster']} - High Saturated Fat**

This pattern represents approximately  
**{result['cluster_prevalence_percent']:.1f}% of U.S. adults**
        """)

        # st.divider()

        # NDQI Placeholder
        # st.markdown("### Nutrient Density Quality Index (NDQI)")
        # mock_ndqi_score = random.uniform(45, 75)

        # st.metric(
        #     label="Overall Diet Quality Score",
        #     value=f"{mock_ndqi_score:.1f} / 100"
        # )

        # st.progress(mock_ndqi_score / 100)

        st.divider()

        # Percentile Positioning
        st.markdown("### Nutrient Percentile Positioning")

        percentile_map = {
            "<10th percentile": 5,
            "10th-25th percentile": 17,
            "25th-50th percentile": 37,
            "50th-75th percentile": 62,
            "75th-90th percentile": 82,
            ">90th percentile": 95
        }

        for nutrient, band in result["percentile_comparison"].items():
            clean_name = nutrient.replace("_per_1000kcal", "")
            approx_value = percentile_map.get(band, 50)

            st.write(f"**{clean_name}** — {band}")
            st.progress(approx_value / 100)

        st.divider()

    st.markdown("### Dietary Recommendations")

    recommendations = []
    # TODO: Implement more sophisticated recommendation logic based on cluster characteristics and nutrient positioning. For now, we use simple rules to generate demo recommendations.
    # Simple rule-based enhancement for demo visuals
    for nutrient, band in result["percentile_comparison"].items():

        if nutrient == "fiber_per_1000kcal" and band in ["<10th percentile", "10th-25th percentile"]:
            recommendations.append("Increase intake of fiber-rich foods such as vegetables, legumes, and whole grains.")

        if nutrient == "sodium_per_1000kcal" and band in ["75th-90th percentile", ">90th percentile"]:
            recommendations.append("Consider reducing processed and packaged foods to lower sodium intake.")

        if nutrient == "sugar_per_1000kcal" and band in ["75th-90th percentile", ">90th percentile"]:
            recommendations.append("Limit sugar-sweetened beverages and added sugars to improve nutrient balance.")

        if nutrient == "satfat_per_1000kcal" and band in ["75th-90th percentile", ">90th percentile"]:
            recommendations.append("Substitute saturated fats with unsaturated sources such as nuts, seeds, and olive oil.")

        if nutrient == "protein_per_1000kcal" and band in ["<10th percentile", "10th-25th percentile"]:
            recommendations.append("Increase lean protein sources such as legumes, fish, poultry, or tofu.")

    # If no specific flags triggered
    if len(recommendations) == 0:
        recommendations.append("Increase intake of fiber-rich foods such as vegetables, legumes, and whole grains.")
        recommendations.append("Consider reducing processed and packaged foods to lower sodium intake.")

    for rec in recommendations:
        st.write(f"- {rec}")

    st.divider()

    st.markdown("### Interpretation Summary")
    st.write(result["advice_text"])

    st.caption(
        "This tool provides population-level comparisons only and "
        "is not a substitute for professional medical advice."
    )