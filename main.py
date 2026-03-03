# ============================================
# Student Burnout & Dropout Risk Detection
# Behavioural Analytics Project
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Generate Synthetic Dataset
# -----------------------------

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "LMS_Login_Frequency": np.random.randint(0, 30, n),
    "Assignment_Delay": np.random.randint(0, 10, n),
    "Attendance_Percentage": np.random.randint(40, 100, n),
    "Missed_Submissions": np.random.randint(0, 5, n),
    "Sentiment_Score": np.random.uniform(-1, 1, n),
    "GPA": np.random.uniform(4, 10, n)
})

# -----------------------------
# 2. Feature Engineering
# -----------------------------

data["Engagement_Score"] = (
    0.4 * data["LMS_Login_Frequency"] +
    0.6 * data["Attendance_Percentage"]
)

data["Academic_Stress_Index"] = (
    data["Assignment_Delay"] +
    data["Missed_Submissions"]
)

# -----------------------------
# 3. Create Target Variables
# -----------------------------

# Burnout Level (Classification)
conditions = [
    (data["Attendance_Percentage"] > 80) &
    (data["Assignment_Delay"] < 3),

    (data["Attendance_Percentage"] > 60),

    (data["Attendance_Percentage"] <= 60)
]

choices = ["Low", "Medium", "High"]
data["Burnout_Level"] = np.select(conditions, choices, default="Medium")
# Dropout Probability (Binary Target)
data["Dropout"] = np.where(
    (data["Attendance_Percentage"] < 60) |
    (data["Assignment_Delay"] > 6) |
    (data["Sentiment_Score"] < -0.5),
    1, 0
)

# -----------------------------
# 4. Model Training
# -----------------------------

features = [
    "LMS_Login_Frequency",
    "Assignment_Delay",
    "Attendance_Percentage",
    "Missed_Submissions",
    "Sentiment_Score",
    "GPA",
    "Engagement_Score",
    "Academic_Stress_Index"
]

X = data[features]
y_burnout = data["Burnout_Level"]
y_dropout = data["Dropout"]

X_train, X_test, y_train_b, y_test_b = train_test_split(
    X, y_burnout, test_size=0.2, random_state=42
)

_, _, y_train_d, y_test_d = train_test_split(
    X, y_dropout, test_size=0.2, random_state=42
)

# Random Forest for Burnout
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train_b)

# Logistic Regression for Dropout
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train_d)

# -----------------------------
# 5. Evaluation
# -----------------------------

print("\n=== Burnout Classification Report ===")
print(classification_report(y_test_b, rf_model.predict(X_test)))

dropout_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
print("\nDropout ROC-AUC Score:",
      roc_auc_score(y_test_d, dropout_probs))

# -----------------------------
# 6. Example Prediction
# -----------------------------

sample_student = X_test.iloc[0:1]

burnout_prediction = rf_model.predict(sample_student)[0]
dropout_prediction = lr_model.predict_proba(
    scaler.transform(sample_student)
)[0][1]

risk_score = round(dropout_prediction * 100, 2)

print("\n=== Sample Student Prediction ===")
print("Burnout Level:", burnout_prediction)
print("Dropout Probability:", round(dropout_prediction, 3))
print("Risk Score (0-100):", risk_score)

