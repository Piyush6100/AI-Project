import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from features import add_engineered_features

PERFORMANCE_CSV = "student_performance.csv"
DROPOUT_CSV = "student_dropout.csv"
COMBINED_CSV = "combined_student_dataset.csv"
PERFORMANCE_MODEL_PKL = "performance_model.pkl"
DROPOUT_MODEL_PKL = "dropout_model.pkl"

BASE_NUMERICAL_FEATURES = [
    "Sem1_Courses_Enrolled",
    "Sem1_Courses_Approved",
    "Study_Hours_Per_Week",
    "Attendance_Percentage",
    "Tutoring_Sessions",
    "Previous_Academic_Score",
    "Sem1_Average_Grade",
]

ENGINEERED_NUMERICAL_FEATURES = [
    "Course_Approval_Rate",
    "Pending_Backlogs",
    "Study_Effort_Per_Course",
    "Academic_Index",
    "Attendance_Study_Product",
    "Support_Score",
]

NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

CATEGORICAL_FEATURES = [
    "Course_Program",
    "Motivation_Level",
    "Access_to_Resources",
    "Tuition_Fees_Up_To_Date",
    "Scholarship_Holder",
]

COMMON_FEATURES = BASE_NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def create_combined_dataset():
    """Reads raw CSVs and merges them into standard format."""
    if not os.path.exists(PERFORMANCE_CSV) or not os.path.exists(DROPOUT_CSV):
        raise FileNotFoundError("Raw CSV files ('student_performance.csv', 'student_dropout.csv') not found.")

    df_p = pd.read_csv(PERFORMANCE_CSV)
    df_d = pd.read_csv(DROPOUT_CSV)

    perf_mapped = pd.DataFrame({
        "Course_Program": ["Computer Science"] * len(df_p),
        "Sem1_Courses_Enrolled": [6] * len(df_p),
        "Sem1_Courses_Approved": (df_p["Attendance"] / 100.0 * 6).round().astype(int),
        "Study_Hours_Per_Week": df_p["Hours_Studied"],
        "Attendance_Percentage": df_p["Attendance"],
        "Tutoring_Sessions": df_p["Tutoring_Sessions"],
        "Motivation_Level": df_p["Motivation_Level"],
        "Access_to_Resources": df_p["Access_to_Resources"],
        "Previous_Academic_Score": df_p["Previous_Scores"],
        "Sem1_Average_Grade": (df_p["Previous_Scores"] / 5.0).round(2),
        "Tuition_Fees_Up_To_Date": ["Yes"] * len(df_p),
        "Scholarship_Holder": ["No"] * len(df_p),
        "Performance_Target": df_p["Exam_Score"],
        "Dropout_Target": [np.nan] * len(df_p),
    })

    course_map = {
        33: "Biofuel Production", 171: "Animation & Multimedia", 8014: "Social Service",
        9003: "Agronomy", 9070: "Communication Design", 9085: "Veterinary Nursing",
        9119: "Computer Science", 9130: "Equiniculture", 9147: "Business & Management",
        9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 9556: "Oral Hygiene",
        9670: "Advertising & Marketing", 9773: "Journalism & Communication",
        9853: "Basic Education", 9991: "Management (Evening)"
    }

    drop_mapped = pd.DataFrame({
        "Course_Program": df_d["Course"].map(course_map).fillna("Computer Science"),
        "Sem1_Courses_Enrolled": df_d["Curricular units 1st sem (enrolled)"],
        "Sem1_Courses_Approved": df_d["Curricular units 1st sem (approved)"],
        "Study_Hours_Per_Week": (df_d["Curricular units 1st sem (evaluations)"] * 3.5).round().astype(int),
        "Attendance_Percentage": np.where(df_d["Daytime/evening attendance"] == 1, 85, 70),
        "Tutoring_Sessions": [1] * len(df_d),
        "Motivation_Level": np.where(df_d["Curricular units 1st sem (approved)"] >= 4, "High", "Low"),
        "Access_to_Resources": np.where(df_d["Scholarship holder"] == 1, "High", "Medium"),
        "Previous_Academic_Score": (df_d["Curricular units 1st sem (grade)"] * 5.0).round(2),
        "Sem1_Average_Grade": df_d["Curricular units 1st sem (grade)"],
        "Tuition_Fees_Up_To_Date": np.where(df_d["Tuition fees up to date"] == 1, "Yes", "No"),
        "Scholarship_Holder": np.where(df_d["Scholarship holder"] == 1, "Yes", "No"),
        "Performance_Target": [np.nan] * len(df_d),
        "Dropout_Target": df_d["Target"],
    })

    combined_df = pd.concat([perf_mapped, drop_mapped], ignore_index=True)
    combined_df.to_csv(COMBINED_CSV, index=False)
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
    print("Loading and preparing datasets...")
    df = create_combined_dataset()

    # 1. Train High-Accuracy Performance Regression Model
    df_perf = df[df["Performance_Target"].notnull()].copy()
    X_p, y_p = df_perf[COMMON_FEATURES], df_perf["Performance_Target"]
    X_p_tr, X_p_te, y_p_tr, y_p_te = train_test_split(
        X_p, y_p, test_size=0.2, random_state=42
    )

    perf_pipeline = Pipeline([
        ("feature_engineer", FunctionTransformer(add_engineered_features)),
        ("preprocessor", build_preprocessor()),
        ("model", Ridge(alpha=1.0)),
    ])

    perf_pipeline.fit(X_p_tr, y_p_tr)
    p_preds = perf_pipeline.predict(X_p_te)
    r2 = r2_score(y_p_te, p_preds)
    rmse = np.sqrt(mean_squared_error(y_p_te, p_preds))
    print(f"Performance Model Trained | Test R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")

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
        ("model", GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            random_state=42
        )),
    ])

    drop_pipeline.fit(X_d_tr, y_d_tr)
    d_preds = drop_pipeline.predict(X_d_te)
    acc = accuracy_score(y_d_te, d_preds)
    f1 = f1_score(y_d_te, d_preds, average="weighted")
    print(f"Dropout Model Trained     | Test Accuracy: {acc*100:.2f}% | Weighted F1: {f1:.4f}")

    drop_pipeline.fit(X_d, y_d)
    joblib.dump(drop_pipeline, DROPOUT_MODEL_PKL)
    print(f"Saved optimized {DROPOUT_MODEL_PKL}\n")
    print("All models successfully trained and updated!")


if __name__ == "__main__":
    train()
