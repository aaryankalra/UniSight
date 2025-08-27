import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import joblib

st.set_page_config(
    page_title="UniSight",
    page_icon="🎓",
)

model = joblib.load("lr_model.joblib")
scaler = joblib.load("scaler.joblib")

df = pd.read_csv("cleaned_dataset.csv")
df.columns = df.columns.str.strip()
importance_df = pd.read_csv("importance_df.csv")

def predict(model, scaler, gre, toefl, rank, sop, lor, cgpa, research):
    research_val = 1 if str(research).lower() == "yes" else 0
    features = np.array([[gre, toefl, rank, sop, lor, cgpa, research_val]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    return max(0, min(1, prediction))

if "page" not in st.session_state:
    st.session_state.page = "Home"
    
pages = ["Home", "Explore", "Predict", "Matches", "About"]
for p in pages:
    if st.sidebar.button(p, use_container_width=True):
        st.session_state.page = p
        
if st.session_state.page == "Home":
    st.title("UniSight 🎓")
    
    st.write("")
    
    st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExMTRrdjI3eHR1Y2R2eW1sZWY4c2FsZW9udWw3bWV2OGU3NHZtZGc1OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tsSjkX8fshUsr2byS3/giphy.gif", width=300)
    
    st.write("")

    st.text("Applying to your dream university can feel uncertain. UniSight helps you cut through the guesswork by using real student data and advanced machine learning to provide data-driven insights into your chances of admission.")

    st.markdown(
        """
        ##### **What Can You Do Here?**
        - **Personalized Predictions:** Enter your GRE, TOEFL, CGPA, LOR, SOP, and research background to get an estimated probability of admission.
        - **Admission Trends:** Explore how different factors (like GRE scores, GPA, and research experience) affect acceptance chances across universities.
        - **Peer Comparisons:** See how students with similar profiles performed in past admissions.
        """
        
    )
    
elif st.session_state.page == "Explore":
    st.title("Explore Admission Insights")
    st.info("In-depth analysis of important features affecting student's admission chances")
    
    st.text("")
    st.text("")
    st.text("")
    
    st.write("### **Correlation Heatmap of Features**")
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### **GRE Score vs. Admission Chances**")
    fig = px.scatter(
        df,
        x="GRE",
        y="Chance",
        size="CGPA",
        color="GRE",
        hover_data=["TOEFL", "SOP", "LOR", "CGPA"],
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Students with strong GRE scores consistently show a much higher probability of admission. This is one of the most visible factors.")
    
    st.write("### Research Experience vs. Admission Chance")
    fig = px.box(
        df,
        x="Research",
        y="Chance",
        points="all",
        color="Research",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Having research experience does improve chances, especially at top universities. However, many applicants without research experience still managed admission chances above 70%, proving it’s not a strict requirement.")
    
    st.write("### TOEFL Score vs. Admission Chances")
    fig = px.scatter(
        df,
        x="TOEFL",
        y="Chance",
        size="CGPA",
        color="TOEFL",
        hover_data=["GRE", "SOP", "LOR", "CGPA"]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Just like GRE, TOEFL scores play a role in shaping outcomes, particularly for international applicants where strong language skills are essential.")
    
    st.write("### CGPA vs. Admission Chances")
    df_cgpa = df.copy()
    bins = [0,6,7,8,9,10]
    labels = ["0-6", "6-7", "7-8", "8-9", "9-10"]
    df_cgpa["CGPA Range"] = pd.cut(df["CGPA"], bins=bins, labels=labels, include_lowest=True)
    grouped = df_cgpa.groupby("CGPA Range", as_index=False)["Chance"].mean()
    fig = px.line(
        grouped,
        x="CGPA Range",
        y="Chance",
        markers=True,
        line_shape="linear"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Academic performance (CGPA) stands out as one of the most critical factors. Applicants with higher CGPA values consistently show higher chances of admission across all university tiers.")
    
    st.write("### SOP & LOR vs. Admission Chances")
    heatmap_data = df.groupby(["SOP", "LOR"], as_index=False)["Chance"].mean()
    fig = px.density_heatmap(
        heatmap_data,
        x="SOP",
        y="LOR",
        z="Chance",
        histfunc="avg",
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Well-written Statements of Purpose and strong Letters of Recommendation significantly improve admission chances. They can often give applicants with average scores a meaningful edge.")
            
elif st.session_state.page == "Predict":
    st.title("Predicting Your Admission Chances")
    st.info("Fill your details to get your estimated chance of admission")
    
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    
    st.subheader("Key Features Affecting Admission Chances")
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.text("")
    st.text("")
    st.text("")
    
    gre = st.number_input("GRE Score (260-340)", min_value=260, max_value=340, step=1)
    toefl = st.number_input("TOEFL SCore (0-120)", min_value=0, max_value=120, step=1)
    rank = st.slider("University Rank (1-5)", 1, 5, 3)
    sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, 3.0, 0.5)
    lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0, 0.5)
    cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.1)
    research = st.radio("Research Experience", ["Yes", "No"])
    
    st.write("")
    
    if st.button("Predict Admission Chance", use_container_width=True, type="primary"):
        chance = predict(model, scaler, gre, toefl, rank, sop, lor, cgpa, research)
        st.success(f"Estimated Chance of Admission: **{chance*100:.2f}%**")
        
elif st.session_state.page == "Matches":
    st.title("Find Similar Profiles")
    st.info("Find out how many applicants with a same profile as yours got selected in universities")
    
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    
    st.write("### Enter Your Profile Details")
    
    st.text("")
    
    cgpa_range = st.slider("CGPA Range", float(df["CGPA"].min()), float(df["CGPA"].max()), (float(df["CGPA"].min()), float(df["CGPA"].max())))
    lor_range = st.slider("LOR Range", float(df["LOR"].min()), float(df["LOR"].max()), (float(df["LOR"].min()), float(df["LOR"].max())))
    sop_range = st.slider("SOP Range", float(df["SOP"].min()), float(df["SOP"].max()), (float(df["SOP"].min()), float(df["SOP"].max())))
    toefl_range = st.slider("TOEFL Range", int(df["TOEFL"].min()), int(df["TOEFL"].max()), (int(df["TOEFL"].min()), int(df["TOEFL"].max())))
    gre_range = st.slider("GRE Range", int(df["GRE"].min()), int(df["GRE"].max()), (int(df["GRE"].min()), int(df["GRE"].max())))
    research_choice = st.radio("Research Experience", ["Yes", "No"])
    
    filtered_df = df[
        (df["CGPA"].between(cgpa_range[0], cgpa_range[1])) &
        (df["LOR"].between(lor_range[0], lor_range[1])) &
        (df["SOP"].between(sop_range[0], sop_range[1])) &
        (df["TOEFL"].between(toefl_range[0], toefl_range[1])) &
        (df["GRE"].between(gre_range[0], gre_range[1]))
    ]

    if research_choice == "Yes":
        filtered_df = filtered_df[filtered_df["Research"] == 1]
    elif research_choice == "No":
        filtered_df = filtered_df[filtered_df["Research"] == 0]
        
    st.write("")
    st.write("")
    
    st.subheader("Results")
    if filtered_df.empty:
        st.warning("No students match the selected criteria.")
    else:
        st.bar_chart(filtered_df["Rank"].value_counts())
        st.write(filtered_df["Rank"].value_counts())
        
elif st.session_state.page == "About":
    st.title("About")
    
    st.write("")
    st.write("")
    
    st.write("The insights you see here are generated from a **publicly available dataset**.")
    st.write("It contains details such as GRE, TOEFL, CGPA, SOP, LOR, Research Experience and University Rank.")
    st.write("Filters you apply will help you explore how students with similar profiles performed in university admissions based on the dataset.")
    
    st.write("")
    st.write("")
    
    st.write('##### Dataset Used')
    st.write("**Graduate Admission 2**")
    st.write("Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019")
    st.link_button("Visit", "https://www.kaggle.com/datasets/mohansacharya/graduate-admissions")