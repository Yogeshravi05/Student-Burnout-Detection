import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Behavioural Analytics Dashboard", layout="wide")

st.title("🎓 Behavioural Analytics Dashboard")
st.subheader("Early Detection of Student Burnout & Dropout Risk")

# -------------------------------------
# Generate Synthetic Dataset
# -------------------------------------

np.random.seed(42)
n = 1000

data = pd.DataFrame({
"LMS_Login_Frequency": np.random.randint(0,30,n),
"Assignment_Delay": np.random.randint(0,10,n),
"Attendance": np.random.randint(40,100,n),
"Missed_Submissions": np.random.randint(0,5,n),
"Sentiment_Score": np.random.uniform(-1,1,n),
"GPA": np.random.uniform(4,10,n)
})

data["Engagement_Score"] = (
0.4*data["LMS_Login_Frequency"] +
0.6*data["Attendance"]
)

data["Stress_Index"] = (
data["Assignment_Delay"] +
data["Missed_Submissions"]
)

data["Dropout"] = np.where(
(data["Attendance"] < 60) |
(data["Assignment_Delay"] > 6) |
(data["Sentiment_Score"] < -0.5),1,0
)

data["Burnout_Level"] = np.where(
data["Attendance"] > 80,"Low",
np.where(data["Attendance"] > 60,"Medium","High")
)

# -------------------------------------
# Train Models
# -------------------------------------

features = [
"LMS_Login_Frequency",
"Assignment_Delay",
"Attendance",
"Missed_Submissions",
"Sentiment_Score",
"GPA",
"Engagement_Score",
"Stress_Index"
]

X = data[features]
y_burnout = data["Burnout_Level"]
y_dropout = data["Dropout"]

X_train,X_test,y_train,y_test = train_test_split(
X,y_burnout,test_size=0.2,random_state=42
)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

burnout_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test,burnout_pred)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression()
lr.fit(X_scaled,y_dropout)

# -------------------------------------
# Tabs Layout
# -------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
"Overview",
"Behaviour Trends",
"Visual Analytics",
"Model Performance",
"Risk Prediction"
])

# -------------------------------------
# TAB 1 - Overview
# -------------------------------------

with tab1:

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Students",len(data))
    col2.metric("High Burnout",
        len(data[data["Burnout_Level"]=="High"]))
    col3.metric("Avg Engagement",
        round(data["Engagement_Score"].mean(),2))
    col4.metric("Model Accuracy",
        round(accuracy*100,2))

    burnout_counts = data["Burnout_Level"].value_counts().reset_index()
    burnout_counts.columns = ["Burnout_Level","Count"]

    fig = px.bar(
        burnout_counts,
        x="Burnout_Level",
        y="Count",
        color="Burnout_Level",
    title="Burnout Level Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------
# TAB 2 - Behaviour Trends
# -------------------------------------

with tab2:

    st.subheader("Attendance Trend")

    attendance_line = data.groupby("Attendance").size()

    fig = px.line(
        x=attendance_line.index,
        y=attendance_line.values,
        labels={"x":"Attendance","y":"Students"}
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Engagement Score Distribution")

    fig2 = px.histogram(
        data,
        x="Engagement_Score",
        nbins=40
    )

    st.plotly_chart(fig2,use_container_width=True)

# -------------------------------------
# TAB 3 - Visual Analytics
# -------------------------------------

with tab3:

    st.subheader("Assignment Delay vs Burnout")

    fig = px.box(
        data,
        x="Burnout_Level",
        y="Assignment_Delay",
        color="Burnout_Level"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Sentiment vs Stress Index")

    fig2 = px.scatter(
        data,
        x="Sentiment_Score",
        y="Stress_Index",
        color="Burnout_Level"
    )

    st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Feature Correlation")

    corr = data[features].corr()

    fig3 = px.imshow(corr,text_auto=True)

    st.plotly_chart(fig3,use_container_width=True)

# -------------------------------------
# TAB 4 - Model Performance
# -------------------------------------

with tab4:

    st.subheader("Model Accuracy")

    st.write("Random Forest Accuracy:",round(accuracy*100,2),"%")

    report = classification_report(
        y_test,
        burnout_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    st.subheader("Feature Importance")

    importance = rf.feature_importances_

    imp_df = pd.DataFrame({
        "Feature":features,
        "Importance":importance
    })

    fig = px.bar(
        imp_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig,use_container_width=True)

# -------------------------------------
# TAB 5 - Risk Prediction
# -------------------------------------

with tab5:

    st.subheader("Predict Student Burnout Risk")

    col1,col2 = st.columns(2)

    with col1:
        lms = st.slider("LMS Login Frequency",0,30,10)
        delay = st.slider("Assignment Delay",0,10,3)
        attendance = st.slider("Attendance %",40,100,75)

    with col2:
        missed = st.slider("Missed Submissions",0,5,1)
        sentiment = st.slider("Sentiment Score",-1.0,1.0,0.1)
        gpa = st.slider("GPA",4.0,10.0,7.0)

    engagement = 0.4*lms + 0.6*attendance
    stress = delay + missed

    input_data = pd.DataFrame(
        [[lms,delay,attendance,missed,sentiment,gpa,engagement,stress]],
        columns=features
    )

    burnout = rf.predict(input_data)[0]

    dropout_prob = lr.predict_proba(
        scaler.transform(input_data)
    )[0][1]

    risk_score = round(dropout_prob*100,2)

    st.success(f"Burnout Level: {burnout}")
    st.warning(f"Dropout Probability: {round(dropout_prob,3)}")
    st.error(f"Risk Score: {risk_score}/100")
