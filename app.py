import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="IPL Win Predictor 🏏", layout="wide")

# --- Load Model with Caching ---
@st.cache_resource
def load_model():
    return pickle.load(open('ipl_win_predictor.pkl', 'rb'))

pipe = load_model()

# Team and City Lists
teams = [
    'Royal Challengers Bengaluru', 'Punjab Kings', 'Delhi Capitals',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Mumbai Indians',
    'Chennai Super Kings', 'Sunrisers Hyderabad', 'Gujarat Titans',
    'Lucknow Super Giants'
]

cities = [
    'Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
    'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
    'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Bengaluru', 'Indore', 'Sharjah', 'Dubai', 'Navi Mumbai',
    'Lucknow', 'Guwahati', 'Mohali'
]

# --- UI ---


st.markdown("""
    <h1 style="text-align:center; color:#E50914;">🏆 IPL Win Predictor</h1>
    <p style="text-align:center;">Predict the match outcome based on current match stats!</p>
    <hr/>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)
    runs_left = st.number_input("🏃 Runs Left", min_value=0, max_value=300, step=1)
    city = st.selectbox("📍 Match City", sorted(cities))
    custom_city = st.text_input("✍️ Or type a city (optional)", "")

with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", teams)
    balls_left = st.number_input("🏐 Balls Left", min_value=0, max_value=120, step=1)
    wickets = st.number_input("❌ Wickets Down", min_value=0, max_value=10, step=1)
    target = st.number_input("🎯 Target Runs", min_value=1, max_value=300, step=1)

# Use custom typed city if available
final_city = custom_city.strip() if custom_city else city

# --- Predict on Button ---
if st.button("🔍 Predict Winning Chances"):
    crr = (target - runs_left) / ((120 - balls_left) / 6) if balls_left else 0
    rrr = (runs_left / (balls_left / 6)) if balls_left else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [final_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_down': [wickets],
        'target_runs': [target],
        'crr': [round(crr, 2)],
        'rrr': [round(rrr, 2)]
    })

    # Predict
    prediction = pipe.predict_proba(input_df)
    win_prob = prediction[0][0] * 100
    lose_prob = prediction[0][1] * 100

    st.markdown("### 📊 Prediction Result")
    st.success(f"🔵 **{batting_team} Win Probability:** {win_prob:.2f}%")
    st.error(f"🔴 **{bowling_team} Win Probability:** {lose_prob:.2f}%")

    st.progress(int(win_prob))
