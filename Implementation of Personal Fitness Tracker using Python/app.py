import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')

# Set up Streamlit page with navigation icons
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🏋️", layout="wide")

# Apply a new darker fitness-themed background
def add_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(0, 0, 0, 0.6); /* Dark overlay */
            background-blend-mode: darken;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()  # Apply the background

# App Title
st.write("## 🏃 Personal Fitness Tracker")
st.write("In this WebApp, you can observe your predicted calories burned. Enter your parameters such as `Age`, `Gender`, `BMI`, etc., and get the predicted kilocalories burned.")

# Sidebar with sections and new icons
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Predictions", "🔍 Similar Results", "ℹ️ General Info"], 
                        index=0, 
                        format_func=lambda x: f"🔹 {x}")

st.sidebar.title("🛠 User Input")

def user_input_features():
    age = st.sidebar.slider("📅 Age:", 10, 100, 30)
    bmi = st.sidebar.slider("⚖️ BMI:", 15, 40, 20)
    duration = st.sidebar.slider("⏳ Duration (min):", 0, 35, 15)
    heart_rate = st.sidebar.slider("💓 Heart Rate:", 60, 130, 80)
    body_temp = st.sidebar.slider("🌡 Temperature (C):", 36, 42, 38)
    gender_button = st.sidebar.radio("👤 Gender:", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

# Load data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

# Ensure data consistency
X_train = X_train.select_dtypes(include=[np.number])
y_train = y_train.astype(float)

# Train the model
model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
model.fit(X_train, y_train)

# Ensure df columns align with training data
df = df.reindex(columns=X_train.columns, fill_value=0)
df = df.select_dtypes(include=[np.number])  # Ensure df is numeric

# Make the prediction
prediction = model.predict(df)[0]

# Handle different pages in the sidebar
if page == "🏠 Home":
    st.write("### Welcome to the Personal Fitness Tracker! Use the sidebar to navigate.")

elif page == "📊 Predictions":
    st.write("---")
    st.header("📌 Your Parameters")
    st.write(df)

    st.write("---")
    st.header("📈 Prediction")

    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

    st.write(f"🔥 Estimated Burn: {round(prediction, 2)} **kilocalories**")

elif page == "🔍 Similar Results":
    st.write("---")
    st.header("🔍 Similar Results")

    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

    # Find similar results based on predicted calories
    calorie_range = [prediction - 10, prediction + 10]
    similar_data = exercise_df[
        (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
    ]

    if not similar_data.empty:
        st.write(similar_data.sample(min(5, len(similar_data))))  # Show up to 5 similar results
    else:
        st.write("No similar results found.")

elif page == "ℹ️ General Info":
    st.write("---")
    st.header("ℹ️ General Information")

    # Boolean logic for age, duration, etc., compared to the user's input
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

    st.write(f"📅 You are older than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of other people.")
    st.write(f"⏱ Your exercise duration is higher than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of other people.")
    st.write(f"❤️ You have a higher heart rate than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of other people during exercise.")
    st.write(f"🌡 You have a higher body temperature than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of other people during exercise.")
