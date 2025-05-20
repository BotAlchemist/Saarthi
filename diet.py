import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import matplotlib.pyplot as plt
import ast

def run(i_passcode):
    # Set OpenAI API key in environment
    # Combine base key with the user-provided passcode
    i_key = 'sk-proj-gUo7UuBh5llI5FHenFKjT3BlbkFJ01MwxYNzCtIQD9t426H'
    i_key = i_key + i_passcode

    # Set OpenAI API key in environment
    os.environ["OPENAI_API_KEY"] = i_key
    client = OpenAI()



    # Page selector
    page = st.sidebar.radio("Select Page", ["Planner", "Tracker"])

    if page == "Planner":
        st.title("🍱 Saarthi: Weekly Weight Gain Diet Planner")

        st.sidebar.header("🔍 Preferences")
        weight = st.sidebar.number_input("Your weight (kg)", min_value=30, max_value=150, value=45)
        protein_target = round(weight * 1.5)
        nonveg_percent = st.sidebar.slider("% of Non-Veg Meals", 0, 100, 30)

        available_items = st.text_area("Enter the list of food items available (comma separated):",
                                        "Banana, Apple, Milk, Curd, Eggs, Bread, Rice, Dal, Peanuts, Paneer, Roti, Poha, Suji, Oats, Ghee, Potato, Beans, Spinach, Chicken")

        submit = st.button("Generate Weekly Plan")

        if submit:
            with st.spinner("Creating your personalized diet plan..."):
                prompt = f"""
                I want you to act as a nutritionist and meal planner for a healthy weight gain diet. Here are my preferences:

                - Weight: {weight} kg, aiming for at least 1.5g/kg protein per day (~{protein_target}g/day)
                - Indian household-based weekly diet plan
                - 3 meals + 1 snack per day
                - {nonveg_percent}% of meals can be non-vegetarian (eggs, chicken, fish)
                - Avoid repeating same meals each weekday
                - Easy-to-make recipes using common kitchen ingredients

                Available ingredients this week:
                {available_items}

                Please generate a 7-day plan with:
                - Breakfast, Snack, Lunch, Dinner for each day
                - Return the result in a Markdown-style table format with the following columns:
                | Day | Meal | Description | Calories | Protein (g) | Carbs (g) | Fat (g) |

                Example:
                | Day | Meal      | Description                     | Calories | Protein (g) | Carbs (g) | Fat (g) |
                |-----|-----------|----------------------------------|----------|-------------|-----------|---------|
                | Day 1 | Breakfast | Paneer Paratha with Curd        | 450      | 20          | 40        | 18      |

                Follow this format strictly for all days. Only return the table, no extra commentary.
                """

                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful nutrition expert."},
                        {"role": "user", "content": prompt}
                    ]
                )

                output = response.choices[0].message.content

                try:
                    df = pd.read_csv(pd.compat.StringIO(output))
                except:
                    import re
                    from io import StringIO

                    table_text = re.search(r'(\| ?Day ?\|.*?\n\|[-| ]+\|\n(?:\|.*?\n)+)', output, re.DOTALL)
                    if table_text:
                        df = pd.read_csv(StringIO(table_text.group(1)), sep="|", engine='python')
                        df = df.dropna(axis=1, how='all')
                        df.columns = [c.strip() for c in df.columns]
                        df = df.iloc[1:].reset_index(drop=True)

                        for col in ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]:
                            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
                    else:
                        st.error("Couldn't extract table from the response. Try refining your food list.")
                        st.text_area("Raw Output", output)
                        st.stop()

                st.subheader("📋 Your Weekly Diet Plan")
                st.dataframe(df, use_container_width=True)
                st.download_button("Download Plan as CSV", df.to_csv(index=False), file_name="weekly_diet_plan.csv")

                st.subheader("📊 Daily Nutrient Summary")
                df_totals = df.groupby("Day")[["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]].sum().reset_index()
                df_totals = df_totals.sort_values(by="Day")

                fig, ax = plt.subplots(figsize=(12, 6))
                df_totals.set_index("Day")[["Protein (g)", "Carbs (g)", "Fat (g)"]].plot(
                    kind="bar",
                    stacked=True,
                    ax=ax,
                    colormap="tab20"
                )

                for idx, row in df_totals.iterrows():
                    y_offset = 0
                    for nutrient in ["Protein (g)", "Carbs (g)", "Fat (g)"]:
                        value = row[nutrient]
                        ax.text(idx, y_offset + value / 2, f'{int(value)}g', ha='center', va='center', fontsize=8, color='white')
                        y_offset += value

                ax.set_title("Macronutrient Stack per Day")
                ax.set_ylabel("Grams")
                ax.set_xlabel("Day")
                ax.legend(title="Macronutrients")
                st.pyplot(fig)

    elif page == "Tracker":
        st.title("📈 Meal Intake Tracker")
        st.markdown("Log your meal and let AI estimate its nutrition:")
        meal_input = st.text_area("What did you eat for your recent meal?", "e.g., 2 eggs, 1 bowl rice, 1 cup dal")

        if st.button("Analyze Meal"):
            with st.spinner("Estimating nutritional values..."):
                track_prompt = f"""
                You are a nutrition assistant. Given a food description, return a JSON object with the following keys:
                "Meal", "Description", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"

                Example:
                {{
                  "Meal": "Lunch",
                  "Description": "2 eggs, 1 bowl rice, 1 cup dal",
                  "Calories": 550,
                  "Protein (g)": 22,
                  "Carbs (g)": 60,
                  "Fat (g)": 18
                }}

                Now analyze this:
                {meal_input}
                """

                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful nutrition assistant."},
                        {"role": "user", "content": track_prompt}
                    ]
                )

                if response.choices and response.choices[0].message.content:
                    try:
                        import json
                        parsed_dict = json.loads(response.choices[0].message.content.strip())
                        if isinstance(parsed_dict, dict):
                            df_meal = pd.DataFrame([parsed_dict])
                            for col in ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]:
                                df_meal[col] = pd.to_numeric(df_meal[col], errors='coerce')

                            st.subheader("🍽️ Meal Nutrient Breakdown")
                            st.dataframe(df_meal, use_container_width=True)
                        else:
                            st.warning("The response was not in the expected dictionary format.")
                    except Exception as e:
                        st.error(f"Failed to parse response as dictionary: {e}")
                else:
                    st.warning("No valid response received. Try rephrasing the meal description.")
