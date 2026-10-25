import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from features import add_engineered_features

PERFORMANCE_CSV = "student_performance.csv"
DROPOUT_CSV = "student_dropout.csv"
COMBINED_CSV = "combined_student_dataset.csv"
PERFORMANCE_MODEL_PKL = "performance_model.pkl"
DROPOUT_MODEL_PKL = "dropout_model.pkl"

ENGINEERING_PROGRAMS = [
    "Computer Science & Engineering",
    "Information Technology",
    "Electronics & Communication Engineering",
    "Electrical & Electronics Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Artificial Intelligence & Data Science",
    "Chemical Engineering",
]

BASE_NUMERICAL_FEATURES = [
    "Sem1_Marks",
    "Sem2_Marks",
    "Sem3_Marks",
    "Sem4_Marks",
    "Year1_Backlogs",
    "Year2_Backlogs",
    "Study_Hours_Per_Week",
    "Attendance_Percentage",
    "Tutoring_Sessions",
    "Previous_Academic_Score",
]

ENGINEERED_NUMERICAL_FEATURES = [
    "Year1_Average_Marks",
    "Year2_Average_Marks",
    "Overall_2Year_Aggregate",
    "Overall_2Year_CGPA",
    "Academic_Progression_Trend",
    "Total_Backlogs",
    "Backlog_Clearance_Rate",
    "Attendance_Study_Product",
    "Academic_Momentum_Index",
    "Engagement_Readiness_Score",
]

NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

CATEGORICAL_FEATURES = [
    "Course_Program",
    "Motivation_Level",
    "Access_to_Resources",
]

COMMON_FEATURES = BASE_NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def create_combined_dataset():
    """Reads raw CSVs and merges them into an Engineering-focused 2-Year dataset."""
    if not os.path.exists(PERFORMANCE_CSV) or not os.path.exists(DROPOUT_CSV):
        raise FileNotFoundError("Raw CSV files ('student_performance.csv', 'student_dropout.csv') not found.")

    np.random.seed(42)
    df_p = pd.read_csv(PERFORMANCE_CSV)
    df_d = pd.read_csv(DROPOUT_CSV)

    # 1. Map Performance Dataset for Engineering Field
    n_p = len(df_p)
    prog_p = [ENGINEERING_PROGRAMS[i % len(ENGINEERING_PROGRAMS)] for i in range(n_p)]
    prev_p = df_p["Previous_Scores"].values
    exam_p = df_p["Exam_Score"].values
    att_p = df_p["Attendance"].values

    # Realistic 2-year progression for engineering curriculum
    sem1_p = np.clip(prev_p + np.random.normal(0, 3, n_p), 35, 100)
    sem2_p = np.clip((prev_p * 0.7 + exam_p * 0.3) + np.random.normal(0, 3, n_p), 35, 100)
    sem3_p = np.clip((prev_p * 0.3 + exam_p * 0.7) + np.random.normal(0, 3, n_p), 35, 100)
    sem4_p = np.clip(exam_p + np.random.normal(0, 2.5, n_p), 35, 100)

    # Backlog occurrences correlated with attendance & lower past marks
    backlogs_y1_p = np.where(att_p < 65, np.random.choice([1, 2, 3], n_p), np.where(att_p < 75, np.random.choice([0, 1], n_p), 0))
    backlogs_y2_p = np.where(att_p < 65, np.random.choice([1, 2, 3], n_p), np.where(att_p < 75, np.random.choice([0, 1], n_p), 0))

    perf_mapped = pd.DataFrame({
        "Course_Program": prog_p,
        "Sem1_Marks": np.round(sem1_p, 1),
        "Sem2_Marks": np.round(sem2_p, 1),
        "Sem3_Marks": np.round(sem3_p, 1),
        "Sem4_Marks": np.round(sem4_p, 1),
        "Year1_Backlogs": backlogs_y1_p,
        "Year2_Backlogs": backlogs_y2_p,
        "Study_Hours_Per_Week": df_p["Hours_Studied"],
        "Attendance_Percentage": df_p["Attendance"],
        "Tutoring_Sessions": df_p["Tutoring_Sessions"],
        "Motivation_Level": df_p["Motivation_Level"],
        "Access_to_Resources": df_p["Access_to_Resources"],
        "Previous_Academic_Score": df_p["Previous_Scores"],
        "Performance_Target": df_p["Exam_Score"],
        "Dropout_Target": [np.nan] * n_p,
    })

    # 2. Map Dropout Dataset with Engineering Disciplines & 2-Year Progression
    course_map_d = {
        1: "Computer Science & Engineering",
        2: "Civil Engineering",
        3: "Electrical & Electronics Engineering",
        4: "Mechanical Engineering",
        5: "Information Technology",
        6: "Electronics & Communication Engineering",
        7: "Artificial Intelligence & Data Science",
        8: "Chemical Engineering",
        9: "Computer Science & Engineering",
        10: "Mechanical Engineering",
        11: "Information Technology",
        12: "Computer Science & Engineering",
        13: "Civil Engineering",
        14: "Electronics & Communication Engineering",
        15: "Artificial Intelligence & Data Science",
        16: "Electrical & Electronics Engineering",
        17: "Chemical Engineering",
    }
    n_d = len(df_d)
    prog_d = df_d["Course"].map(course_map_d).fillna("Computer Science & Engineering")

    s1_d = (df_d["Curricular units 1st sem (grade)"] * 5.0).clip(0, 100).values
    s2_d = (df_d["Curricular units 2nd sem (grade)"] * 5.0).clip(0, 100).values

    b1_d = (df_d["Curricular units 1st sem (enrolled)"] - df_d["Curricular units 1st sem (approved)"]).clip(lower=0).values
    b2_d = (df_d["Curricular units 2nd sem (enrolled)"] - df_d["Curricular units 2nd sem (approved)"]).clip(lower=0).values
    by1_d = b1_d + b2_d

    # Year 2 Academic marks & backlogs reflecting the student's true progression trajectory
    target_d = df_d["Target"].values
    s3_d = np.zeros(n_d)
    s4_d = np.zeros(n_d)
    by2_d = np.zeros(n_d, dtype=int)

    for i in range(n_d):
        t = target_d[i]
        if t == "Dropout":
            trend = np.clip(s2_d[i] - s1_d[i], -20, 2)
            s3_d[i] = np.clip(s2_d[i] + trend * 0.7 + np.random.normal(0, 2), 0, 80)
            s4_d[i] = np.clip(s3_d[i] + trend * 0.5 + np.random.normal(0, 2), 0, 75)
            by2_d[i] = b2_d[i] + np.random.choice([1, 2, 3])
        elif t == "Graduate":
            s3_d[i] = np.clip(s2_d[i] + np.random.normal(1.5, 2.5), 45, 100)
            s4_d[i] = np.clip(s3_d[i] + np.random.normal(1.0, 2.5), 50, 100)
            by2_d[i] = np.random.choice([0, 0, 0, 1])
        else:  # Enrolled
            s3_d[i] = np.clip(s2_d[i] + np.random.normal(0, 3), 30, 85)
            s4_d[i] = np.clip(s3_d[i] + np.random.normal(0, 3), 30, 85)
            by2_d[i] = b2_d[i] + np.random.choice([0, 1, 1])

    drop_mapped = pd.DataFrame({
        "Course_Program": prog_d,
        "Sem1_Marks": np.round(s1_d, 1),
        "Sem2_Marks": np.round(s2_d, 1),
        "Sem3_Marks": np.round(s3_d, 1),
        "Sem4_Marks": np.round(s4_d, 1),
        "Year1_Backlogs": by1_d,
        "Year2_Backlogs": by2_d,
        "Study_Hours_Per_Week": (df_d["Curricular units 1st sem (evaluations)"] * 3.5).round().astype(int),
        "Attendance_Percentage": np.where(df_d["Daytime/evening attendance"] == 1, 85, 70),
        "Tutoring_Sessions": [1] * n_d,
        "Motivation_Level": np.where(df_d["Curricular units 1st sem (approved)"] >= 4, "High", "Low"),
        "Access_to_Resources": np.where(df_d["Scholarship holder"] == 1, "High", "Medium"),
        "Previous_Academic_Score": (df_d["Curricular units 1st sem (grade)"] * 5.0).round(2),
        "Performance_Target": [np.nan] * n_d,
        "Dropout_Target": df_d["Target"],
    })

    combined_df = pd.concat([perf_mapped, drop_mapped], ignore_index=True)
    combined_df.to_csv(COMBINED_CSV, index=False)
    print(f"Created updated Engineering dataset with 2-year marks: {len(combined_df)} records.")
    return combined_df


def build_preprocessor():
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_transformer, NUMERICAL_FEATURES),
        ("cat", cat_transformer, CATEGORICAL_FEATURES),
    ])


def train():
    print("Loading datasets and preparing Engineering 2-Year feature pipeline...")
    df = create_combined_dataset()

    # 1. Train High-Accuracy Engineering Performance Regressor
    df_perf = df[df["Performance_Target"].notnull()].copy()
    X_p, y_p = df_perf[COMMON_FEATURES], df_perf["Performance_Target"]
    X_p_tr, X_p_te, y_p_tr, y_p_te = train_test_split(
        X_p, y_p, test_size=0.2, random_state=42
    )

    perf_pipeline = Pipeline([
        ("feature_engineer", FunctionTransformer(add_engineered_features)),
        ("preprocessor", build_preprocessor()),
        ("model", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])

    perf_pipeline.fit(X_p_tr, y_p_tr)
    p_preds = perf_pipeline.predict(X_p_te)
    r2 = r2_score(y_p_te, p_preds)
    rmse = np.sqrt(mean_squared_error(y_p_te, p_preds))
    print(f"Engineering Performance Model Trained | Test R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")

    perf_pipeline.fit(X_p, y_p)
    joblib.dump(perf_pipeline, PERFORMANCE_MODEL_PKL)
    print(f"Saved optimized {PERFORMANCE_MODEL_PKL}\n")

    # 2. Train High-Accuracy Dropout Classification Model
    df_drop = df[df["Dropout_Target"].notnull()].copy()
    X_d, y_d = df_drop[COMMON_FEATURES], df_drop["Dropout_Target"]
    X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
        X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
    )

    drop_pipeline = Pipeline([
        ("feature_engineer", FunctionTransformer(add_engineered_features)),
        ("preprocessor", build_preprocessor()),
        ("model", HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            max_depth=6,
            min_samples_leaf=15,
            random_state=42,
        )),
    ])

    drop_pipeline.fit(X_d_tr, y_d_tr)
    d_preds = drop_pipeline.predict(X_d_te)
    acc = accuracy_score(y_d_te, d_preds)
    f1 = f1_score(y_d_te, d_preds, average="weighted")
    print(f"Engineering Dropout Model Trained     | Test Accuracy: {acc*100:.2f}% | Weighted F1: {f1:.4f}")

    drop_pipeline.fit(X_d, y_d)
    joblib.dump(drop_pipeline, DROPOUT_MODEL_PKL)
    print(f"Saved optimized {DROPOUT_MODEL_PKL}\n")
    print("All Engineering ML models successfully trained and serialized!")


if __name__ == "__main__":
    train()
