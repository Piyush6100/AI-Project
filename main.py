import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model File Constants
PERFORMANCE_MODEL_PKL = "performance_model.pkl"
DROPOUT_MODEL_PKL = "dropout_model.pkl"

COMMON_FEATURES = [
    "Course_Program",
    "Sem1_Courses_Enrolled",
    "Sem1_Courses_Approved",
    "Study_Hours_Per_Week",
    "Attendance_Percentage",
    "Tutoring_Sessions",
    "Motivation_Level",
    "Access_to_Resources",
    "Previous_Academic_Score",
    "Sem1_Average_Grade",
    "Tuition_Fees_Up_To_Date",
    "Scholarship_Holder",
]

perf_pipeline = None
drop_pipeline = None


def load_models():
    """Loads pre-trained machine learning pipelines."""
    global perf_pipeline, drop_pipeline
    if not os.path.exists(PERFORMANCE_MODEL_PKL) or not os.path.exists(DROPOUT_MODEL_PKL):
        raise FileNotFoundError(
            "Model files (.pkl) not found. Run 'python train.py' first to train models."
        )
    perf_pipeline = joblib.load(PERFORMANCE_MODEL_PKL)
    drop_pipeline = joblib.load(DROPOUT_MODEL_PKL)


def explain_risk_factors(data):
    """Analyzes student input data and identifies key academic risk factors."""
    factors = []

    attendance = float(data.get("Attendance_Percentage", 100))
    if attendance < 60:
        factors.append(f"Attendance is very low ({attendance:.0f}%), which is associated with increased academic difficulty.")
    elif attendance < 75:
        factors.append(f"Attendance ({attendance:.0f}%) is below the recommended 75% threshold.")

    prev_score = float(data.get("Previous_Academic_Score", 100))
    if prev_score < 50:
        factors.append(f"Previous academic score is low ({prev_score:.1f} / 100).")
    elif prev_score < 60:
        factors.append(f"Previous academic score ({prev_score:.1f} / 100) is below average.")

    enrolled = int(data.get("Sem1_Courses_Enrolled", 6))
    approved = int(data.get("Sem1_Courses_Approved", 6))
    backlogs = max(0, enrolled - approved)
    if backlogs == 1:
        factors.append("Student has 1 pending backlog subject.")
    elif backlogs > 1:
        factors.append(f"Student has {backlogs} pending backlog subjects.")

    study_hours = float(data.get("Study_Hours_Per_Week", 20))
    if study_hours < 10:
        factors.append(f"Weekly self-study time is very low ({study_hours:.1f} hours/week).")
    elif study_hours < 15:
        factors.append(f"Weekly self-study time ({study_hours:.1f} hours/week) is below recommended level.")

    sem1_grade = float(data.get("Sem1_Average_Grade", 14))
    if sem1_grade < 10.0:
        factors.append(f"1st semester average grade is low ({sem1_grade:.1f} / 20).")

    if str(data.get("Tuition_Fees_Up_To_Date", "Yes")) == "No":
        factors.append("Tuition fees are currently overdue.")

    if str(data.get("Motivation_Level", "Medium")) == "Low":
        factors.append("Study motivation level is low.")

    if str(data.get("Access_to_Resources", "Medium")) == "Low":
        factors.append("Access to study resources and learning materials is low.")

    if not factors:
        factors.append("No major academic risk factors identified based on current input values.")

    return factors


def generate_suggestions(data):
    """Generates personalized, actionable recommendations based on student inputs."""
    suggestions = []

    attendance = float(data.get("Attendance_Percentage", 100))
    if attendance < 75:
        suggestions.append("Focus on improving lecture and lab attendance to maintain regular class participation.")

    enrolled = int(data.get("Sem1_Courses_Enrolled", 6))
    approved = int(data.get("Sem1_Courses_Approved", 6))
    if enrolled > approved:
        suggestions.append("Create a study schedule to clear pending backlog subjects, starting with core concepts.")

    study_hours = float(data.get("Study_Hours_Per_Week", 20))
    if study_hours < 15:
        suggestions.append("Gradually increase consistent self-study time (aim for at least 15–20 hours per week).")

    prev_score = float(data.get("Previous_Academic_Score", 100))
    sem1_grade = float(data.get("Sem1_Average_Grade", 14))
    if prev_score < 60 or sem1_grade < 11.0:
        suggestions.append("Discuss challenging subjects with faculty members or join academic tutoring sessions.")

    if str(data.get("Tuition_Fees_Up_To_Date", "Yes")) == "No":
        suggestions.append("Contact the college financial aid office or administrative desk regarding tuition payment plans.")

    if str(data.get("Motivation_Level", "Medium")) == "Low":
        suggestions.append("Join peer study groups or consult a student counselor to rebuild academic momentum.")

    if str(data.get("Access_to_Resources", "Medium")) == "Low":
        suggestions.append("Utilize campus library facilities, digital courseware, and open-access study materials.")

    if not suggestions:
        suggestions.append("Maintain your current excellent study habits, attendance, and course engagement.")

    return suggestions


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([{feat: data.get(feat) for feat in COMMON_FEATURES}])

        # Generate model predictions
        predicted_score = float(perf_pipeline.predict(df_input)[0])
        predicted_dropout = str(drop_pipeline.predict(df_input)[0])

        dropout_prob = None
        if hasattr(drop_pipeline, "predict_proba"):
            probs = drop_pipeline.predict_proba(df_input)[0]
            classes = list(drop_pipeline.named_steps["model"].classes_)
            prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            if "Dropout" in prob_dict:
                dropout_prob = round(prob_dict["Dropout"] * 100, 1)

        res_payload = {
            "performance_prediction": f"{predicted_score:.2f} / 100",
            "dropout_prediction": predicted_dropout,
            "risk_factors": explain_risk_factors(data),
            "suggestions": generate_suggestions(data),
        }
        if dropout_prob is not None:
            res_payload["dropout_probability"] = dropout_prob

        return jsonify(res_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    load_models()
    print("\nServer running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
