import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="ExtraaLearn Lead Conversion Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load Trained Model Pipeline Artifact
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# App Header
st.title("🎓 ExtraaLearn Lead Conversion Predictor")
st.markdown("""
This intelligent web application predicts whether a prospective lead is likely to convert into a paid customer,
helping the sales team allocate resources effectively and target high-intent leads.
""")

if model is None:
    st.error("Error: Trained model artifact ('model.pkl') not found in directory. Please run the training pipeline first.")
else:
    st.sidebar.header("📋 Enter Lead Attributes")

    def user_input_features():
        age = st.sidebar.slider("Age of Lead", 18, 65, 30)
        current_occupation = st.sidebar.selectbox("Current Occupation", ['Professional', 'Unemployed', 'Student'])
        first_interaction = st.sidebar.selectbox("First Interaction Channel", ['Website', 'Mobile App'])
        profile_completed = st.sidebar.selectbox("Profile Completion %", ['Low', 'Medium', 'High'])
        website_visits = st.sidebar.slider("Website Visits", 0, 30, 3)
        time_spent_on_website = st.sidebar.slider("Time Spent on Website (seconds)", 0, 2500, 500)
        page_views_per_visit = st.sidebar.slider("Avg Page Views per Visit", 0.0, 18.0, 3.0)
        last_activity = st.sidebar.selectbox("Last Activity", ['Email Activity', 'Phone Activity', 'Website Activity'])

        print_media_type1 = st.sidebar.selectbox("Seen Ad in Newspaper?", ['No', 'Yes'])
        print_media_type2 = st.sidebar.selectbox("Seen Ad in Magazine?", ['No', 'Yes'])
        digital_media = st.sidebar.selectbox("Seen Ad on Digital Platforms?", ['No', 'Yes'])
        educational_channels = st.sidebar.selectbox("Heard via Educational Channels?", ['No', 'Yes'])
        referral = st.sidebar.selectbox("Heard via Referral?", ['No', 'Yes'])

        data = {
            'age': age,
            'current_occupation': current_occupation,
            'first_interaction': first_interaction,
            'profile_completed': profile_completed,
            'website_visits': website_visits,
            'time_spent_on_website': time_spent_on_website,
            'page_views_per_visit': page_views_per_visit,
            'last_activity': last_activity,
            'print_media_type1': print_media_type1,
            'print_media_type2': print_media_type2,
            'digital_media': digital_media,
            'educational_channels': educational_channels,
            'referral': referral
        }
        return pd.DataFrame([data])

    input_df = user_input_features()

    # Display User Input Summary
    st.subheader("📊 Lead Profile Summary")
    st.dataframe(input_df)

    # Prediction Button
    if st.button("🔮 Predict Lead Conversion"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("🎯 Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success("### **Converted Lead (High Intent)**")
                st.markdown("✅ *Recommendation: Prioritize immediate sales outreach and schedule a call.*")
            else:
                st.warning("### **Unconverted Lead (Low Intent)**")
                st.markdown("⚠️ *Recommendation: Nurture via automated email campaigns.*")

        with col2:
            st.metric(label="Conversion Probability", value=f"{probability * 100:.2f}%")

        # Probability progress bar
        st.progress(float(probability))
