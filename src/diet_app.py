import streamlit as st
import pandas as pd
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



# Source for questions: https://academic.oup.com/crohnscolitis360/article/7/4/otaf052/8315020?login=false&utm_source=chatgpt.com

st.divider()

st.header("Dietary Screening Questionnaire")

st.markdown("These questions help contextualize intake patterns.")

#------------
# Insert questions here
#------------
meal_freq = st.selectbox(
    "How many meals or snacks do you typically eat per day?",
    ["1-2", "3", "4-5", "6+"]
)

sugary_drinks = st.selectbox(
    "How often do you drink sugar-sweetened beverages (e.g., soda, sweetened tea)?",
    ["Rarely/Never", "1-3 per week", "4-6 per week", "Daily"]
)

fruit_veg = st.selectbox(
    "How many servings of fruits or vegetables do you eat per day?",
    ["0-1", "2-3", "4-5", "6+"]
)

processed_food = st.selectbox(
    "How often do you eat highly processed foods (fast food, frozen meals, packaged snacks)?",
    ["Rarely", "1-2 times per week", "3-6 times per week", "Daily"]
)

protein_freq = st.selectbox(
    "How often do you eat protein-rich foods (e.g., legumes, fish, poultry, tofu)?",
    ["Less than once per day", "Once per day", "Twice per day", "Three or more times per day"]
)

whole_grains = st.selectbox(
    "How often do you eat whole grains (oats, brown rice, whole wheat bread)?",
    ["Rarely", "Sometimes", "Often", "Almost always"]
)

calorie_map = {
    "1-2": 1800,
    "3": 2000,
    "4-5": 2200,
    "6+": 2400
}
sugar_map = {
    "Rarely/Never": 5,
    "1-3 per week": 20,
    "4-6 per week": 40,
    "Daily": 70
}

fiber_map = {
    "0-1": 8,
    "2-3": 16,
    "4-5": 25,
    "6+": 32
}

wholegrain_map = {
    "Rarely": 0,
    "Sometimes": 3,
    "Often": 6,
    "Almost always": 10
}
processed_map = {
    "Rarely": {"sodium": 1800, "satfat": 15},
    "1-2 times per week": {"sodium": 2400, "satfat": 20},
    "3-6 times per week": {"sodium": 3200, "satfat": 28},
    "Daily": {"sodium": 4200, "satfat": 35}
}
protein_map = {
    "Less than once per day": 40,
    "Once per day": 55,
    "Twice per day": 75,
    "Three or more times per day": 95
}

estimated_calories = calorie_map[meal_freq]
estimated_sugar = sugar_map[sugary_drinks]
estimated_fiber = fiber_map[fruit_veg] + wholegrain_map[whole_grains]
estimated_sodium = processed_map[processed_food]["sodium"]
estimated_satfat = processed_map[processed_food]["satfat"]
estimated_protein = protein_map[protein_freq]

st.divider()

if st.button("Estimate Intake from Questionnaire"):
    st.session_state.estimated = {
        "calories": estimated_calories,
        "sugar_g": estimated_sugar,
        "fiber_g": estimated_fiber,
        "sodium_mg": estimated_sodium,
        "satfat_g": estimated_satfat,
        "protein_g": estimated_protein
    }
    st.success("Estimated intake values have been populated in the input fields below. You can adjust them if needed before analysis.")


st.header("Daily Intake Estimation")

if "estimated" in st.session_state and st.session_state.estimated:
    st.divider()
    st.subheader("Adjust Your Estimated Intake")
    calories = st.number_input("Estimated Daily Calories", value=st.session_state.estimated["calories"], min_value=0)
    sugar = st.number_input("Estimated Daily Sugar (g)", value=float(st.session_state.estimated["sugar_g"]), min_value=0.0)
    fiber = st.number_input("Estimated Daily Fiber (g)", value=float(st.session_state.estimated["fiber_g"]), min_value=0.0)
    sodium = st.number_input("Estimated Daily Sodium (mg)", value=float(st.session_state.estimated["sodium_mg"]), min_value=0.0)
    satfat = st.number_input("Estimated Daily Saturated Fat (g)", value=float(st.session_state.estimated["satfat_g"]), min_value=0.0)
    protein = st.number_input("Estimated Daily Protein (g)", value=float(st.session_state.estimated["protein_g"]), min_value=0.0)

    st.divider()

    if st.button("Analyze my diet"):

        if calories <= 0:
            st.error("Calories must be greater than zero.")
        else:
            with st.spinner("Analyzing dietary pattern..."):
                try:
                    st.session_state.result = generate_recommendation(
                        calories=calories,
                        sugar_g=sugar,
                        fiber_g=fiber,
                        sodium_mg=sodium,
                        satfat_g=satfat,
                        protein_g=protein
                    )
                except Exception as e:
                    st.error("An error occurred during analysis. Please check your input values and try again.")
                    st.stop()
                
                cluster_type = "Mixed Pattern"
                if st.session_state.result['cluster'] == 0:
                    cluster_type = "High Sugar, Low Sodium, Low Protein"
                elif st.session_state.result['cluster'] == 1:
                    cluster_type = "High Saturated Fat"
                elif st.session_state.result['cluster'] == 2:
                    cluster_type = "High Sodium, High Protein, Low Sugar"
                elif st.session_state.result['cluster'] == 3:
                    cluster_type = "High Fiber"

        # We no longer hard code the cluster interpretations here.
        # It is still "hard coded" since the data shows that these are the main patterns, 
        # but the actual interpretation is now based on the data and not just a static string.
        # This allows for more flexibility in the future if we want to change the cluster definitions or add more clusters.
        st.markdown(f"""
        ### Dietary Pattern Classification
        You most closely resemble **Diet {st.session_state.result['cluster']} - {cluster_type}**
        This pattern represents approximately  
        **{st.session_state.result['cluster_prevalence_percent']:.1f}% of U.S. adults**
        """)
        
        st.divider()

        st.markdown("### Your Intake vs Recommended Intake (per 1000 kcal)")

        user_density = {
            "Sugar": sugar / calories * 1000,
            "Fiber": fiber / calories * 1000,
            "Sodium": (sodium / calories * 1000) / 1000, # convert mg to g for better visualization
            "SatFat": satfat / calories * 1000,
            "Protein": protein / calories * 1000
        }

        healthy_density = {
            "Sugar": 25,     # g per 1000 kcal
            "Fiber": 14,     # g per 1000 kcal
            "Sodium": 1.5,   # g per 1000 kcal
            "SatFat": 10,    # g per 1000 kcal
            "Protein": 50    # g per 1000 kcal
        }

        df = pd.DataFrame({
            "Your Intake (g/1000 kcal)": user_density,
            "Recommended (g/1000 kcal)": healthy_density
        })

        st.bar_chart(df)
        st.caption("Note: Sodium is shown in grams per 1000 kcal for better visualization.")

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
        percentiles = st.session_state.result.get("percentile_comparison", {})
        for nutrient, band in percentiles.items():
            clean_name = nutrient.replace("_per_1000kcal", "")
            approx_value = percentile_map.get(band, 50)

            st.write(f"**{clean_name} — {approx_value}th percentile**")
            st.progress(approx_value / 100)
            
            if approx_value < 50: #50 is just the median, so anything below that is "lower than average"
                st.caption(f"Your intake of {clean_name.lower()} is lower than approximately {100 - approx_value}% of U.S. adults.")
            else:
                st.caption(f"Your intake of {clean_name.lower()} is higher than approximately {approx_value}% of U.S. adults.")

    st.divider()
if "result" in st.session_state and st.session_state.result:
    result = st.session_state.result
    st.markdown("### Interpretation Summary")
    st.write(result["advice_text"])
    st.markdown("### Dietary Recommendations")

    recommendations = []

    percentiles = result.get("percentile_comparison", {})
    # st.write(percentiles)
    fiber_band = percentiles.get("Fiber_per_1000kcal")
    sodium_band = percentiles.get("Sodium_per_1000kcal")
    sugar_band = percentiles.get("Sugar_per_1000kcal")
    satfat_band = percentiles.get("SatFat_per_1000kcal")
    protein_band = percentiles.get("Protein_per_1000kcal")


    if fiber_band in ["<10th percentile", "10th-25th percentile"]:
        recommendations.append(
            "Increase intake of fiber-rich foods such as vegetables, legumes, and whole grains."
        )

    if sodium_band in ["75th-90th percentile", ">90th percentile"]:
        recommendations.append(
            "Consider reducing processed and packaged foods to lower sodium intake."
        )

    if sugar_band in ["75th-90th percentile", ">90th percentile"]:
        recommendations.append(
            "Limit sugar-sweetened beverages and added sugars to improve nutrient balance."
        )

    if satfat_band in ["75th-90th percentile", ">90th percentile"]:
        recommendations.append(
            "Substitute saturated fats with unsaturated sources such as nuts, seeds, and olive oil."
        )

    if protein_band in ["<10th percentile", "10th-25th percentile"]:
        recommendations.append(
            "Increase lean protein sources such as legumes, fish, poultry, or tofu."
        )

    if fruit_veg in ["0-1", "2-3"]:
        recommendations.append(
            "Aim to include more servings of fruits and vegetables in your daily diet."
        )
    if processed_food in ["3-6 times per week", "Daily"]:
        recommendations.append(
            "Try to limit highly processed foods and opt for whole, minimally processed options when possible."
        )
        
    if whole_grains in ["Rarely", "Sometimes"]:
        recommendations.append(
            "Incorporate more whole grains like oats, brown rice, and whole wheat bread into your meals."
        )
        
    if sugary_drinks in ["1-3 per week", "4-6 per week", "Daily"]:
        recommendations.append(
            "Consider reducing consumption of sugar-sweetened beverages and replacing them with water or unsweetened alternatives."
        )
    if meal_freq in ["1-2", "3"]:
        recommendations.append(
            "Eating balanced meals thoughout the day may help maintain consistent energy and nutrient intake."
        )
    if meal_freq in ["6+"]:
        recommendations.append(
            "Frequent snacking can increase calorie intake. Focus on nutrient-dense snacks and balanced meals to support overall diet quality."
        )

    # fallback if nothing triggered
    if not recommendations:
        recommendations.append(
            "Your nutrient intake appears broadly aligned with population patterns. Maintain a balanced diet rich in whole foods."
        )
    for rec in recommendations:
        st.write(f"- {rec}")

    st.caption(
        "This tool provides population-level comparisons only and "
        "is not a substitute for professional medical advice."
    )