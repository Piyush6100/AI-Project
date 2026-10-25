import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model File Constants
PERFORMANCE_MODEL_PKL = "performance_model.pkl"
DROPOUT_MODEL_PKL = "dropout_model.pkl"

COMMON_FEATURES = [
    "Course_Program",
    "Sem1_Marks",
    "Sem2_Marks",
    "Sem3_Marks",
    "Sem4_Marks",
    "Year1_Backlogs",
    "Year2_Backlogs",
    "Study_Hours_Per_Week",
    "Attendance_Percentage",
    "Tutoring_Sessions",
    "Motivation_Level",
    "Access_to_Resources",
    "Previous_Academic_Score",
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
    """Identifies practical engineering-specific risk factors."""
    factors = []

    att = float(data.get("Attendance_Percentage", 85))
    if att < 60:
        factors.append(
            f"Critical Attendance Shortage ({att:.0f}%): Severe risk of university exam debarment (<75% mandate)."
        )
    elif att < 75:
        factors.append(
            f"Attendance Warning ({att:.0f}%): Below the 75% institutional eligibility requirement for semester exams."
        )

    y1_bl = int(data.get("Year1_Backlogs", 0))
    y2_bl = int(data.get("Year2_Backlogs", 0))
    total_bl = y1_bl + y2_bl

    if total_bl > 3:
        factors.append(
            f"Critical Backlog Accumulation: {total_bl} pending subjects ({y1_bl} in Year 1, {y2_bl} in Year 2) risk semester detention."
        )
    elif total_bl > 0:
        factors.append(
            f"Pending Arrears: {total_bl} active backlog(s) ({y1_bl} Year 1, {y2_bl} Year 2) requiring supplementary exams."
        )

    s1 = float(data.get("Sem1_Marks", 70))
    s2 = float(data.get("Sem2_Marks", 70))
    s3 = float(data.get("Sem3_Marks", 70))
    s4 = float(data.get("Sem4_Marks", 70))
    y1_avg = (s1 + s2) / 2.0
    y2_avg = (s3 + s4) / 2.0
    trend = y2_avg - y1_avg

    if trend < -10.0:
        factors.append(
            f"Steep 2nd-Year Decline: Marks dropped by {abs(trend):.1f}% in Year 2 core engineering courses compared to Year 1."
        )
    elif trend < -4.0:
        factors.append(
            f"Negative Trajectory: 2nd-year core subject scores ({y2_avg:.1f}%) are trailing 1st-year scores ({y1_avg:.1f}%)."
        )

    if y2_avg < 50.0:
        factors.append(
            f"Low 2nd-Year Core Score ({y2_avg:.1f}%): Demonstrates difficulty with core departmental engineering subjects."
        )

    cum_cgpa = ((y1_avg + y2_avg) / 2.0) / 10.0
    if cum_cgpa < 6.0:
        factors.append(
            f"Low Cumulative CGPA ({cum_cgpa:.2f} / 10.0): Falls below First-Class campus placement cut-off (6.0+ CGPA)."
        )

    study_hrs = float(data.get("Study_Hours_Per_Week", 20))
    if study_hrs < 12:
        factors.append(
            f"Insufficient Self-Study ({study_hrs:.0f} hrs/week): Below the 15–20 hrs needed for engineering problem solving and coding."
        )

    tutor = int(data.get("Tutoring_Sessions", 1))
    if tutor == 0 and total_bl > 0:
        factors.append(
            "Lack of Mentoring: Student has active backlogs but is not attending any faculty doubt-clearing sessions."
        )

    if str(data.get("Motivation_Level", "Medium")) == "Low":
        factors.append("Low Academic Motivation: At risk of disengagement during intensive engineering project semesters.")

    if str(data.get("Access_to_Resources", "Medium")) == "Low":
        factors.append("Limited Access to Resources: Constrained access to computing labs, simulation tools, or textbooks.")

    if not factors:
        factors.append("No major academic risk factors identified. The student demonstrates steady engineering progress.")

    return factors


def generate_faculty_solutions(data, risk_level, total_backlogs):
    """Generates actionable guidance tailored specifically for engineering faculty mentors & HODs."""
    actions = []

    att = float(data.get("Attendance_Percentage", 85))
    if att < 75:
        actions.append(
            "Issue Official Attendance Advisory: Notify the student and parents regarding exam debarment risk and mandate minimum 80% attendance in upcoming lab and theory sessions."
        )

    if total_backlogs > 0:
        actions.append(
            f"Assign Subject-Specific Remedial Faculty: Enroll the student in departmental tutorial batches to clear {total_backlogs} active backlogs before next university examinations."
        )

    s1 = float(data.get("Sem1_Marks", 70))
    s2 = float(data.get("Sem2_Marks", 70))
    s3 = float(data.get("Sem3_Marks", 70))
    s4 = float(data.get("Sem4_Marks", 70))
    y1_avg = (s1 + s2) / 2.0
    y2_avg = (s3 + s4) / 2.0

    if y2_avg < y1_avg - 5.0 or y2_avg < 55.0:
        actions.append(
            "Core Engineering Diagnostic Review: Conduct a 1-on-1 review with course instructors for 2nd-year core subjects (Mechanics/Circuits/Data Structures) to diagnose conceptual bottlenecks."
        )

    if int(data.get("Tutoring_Sessions", 1)) < 2:
        actions.append(
            "Mandate Bi-Weekly Faculty Mentorship: Schedule bi-weekly mentor check-ins to track coursework submissions, practical lab files, and mid-term test performance."
        )

    if risk_level == "High Risk":
        actions.append(
            "Departmental Academic Counseling: Refer student to the Academic Progress Committee for structured intervention to prevent semester dropout."
        )

    if not actions:
        actions.append(
            "Nominate for Advanced Engineering Projects: Encourage participation in technical hackathons, industry internships, and undergraduate research."
        )

    return actions


def generate_student_solutions(data, risk_level, total_backlogs):
    """Generates personalized, practical roadmap for the engineering student."""
    suggestions = []

    att = float(data.get("Attendance_Percentage", 85))
    if att < 75:
        suggestions.append(
            f"Prioritize Daily Lecture & Lab Attendance: Attend all scheduled classes to lift your attendance from {att:.0f}% above the 75% exam threshold."
        )

    if total_backlogs > 0:
        suggestions.append(
            f"Structured Backlog Clearance Plan: Allocate 5–7 dedicated weekly hours to previous question papers and core concepts for your {total_backlogs} pending backlog(s)."
        )

    study_hrs = float(data.get("Study_Hours_Per_Week", 20))
    if study_hrs < 18:
        suggestions.append(
            "Increase Self-Study & Hands-On Practice: Target 18–22 hours per week covering problem solving, numerical derivation, and coding practice."
        )

    s1 = float(data.get("Sem1_Marks", 70))
    s2 = float(data.get("Sem2_Marks", 70))
    s3 = float(data.get("Sem3_Marks", 70))
    s4 = float(data.get("Sem4_Marks", 70))
    y2_avg = (s3 + s4) / 2.0
    if y2_avg < 60.0:
        suggestions.append(
            "Master 2nd-Year Foundational Courses: Form peer study circles with high-performing batchmates to review complex engineering concepts before mid-terms."
        )

    if str(data.get("Access_to_Resources", "Medium")) == "Low":
        suggestions.append(
            "Leverage Campus Infrastructure: Utilize college central library book-bank schemes, digital NPTEL/SWAYAM lectures, and university open lab hours."
        )

    if not suggestions:
        suggestions.append(
            "Maintain High Academic Momentum: Build industry-ready engineering portfolio projects and prepare for competitive exams (GATE/GRE) or placement drives."
        )

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
        predicted_score = max(0.0, min(100.0, predicted_score))
        predicted_dropout = str(drop_pipeline.predict(df_input)[0])

        dropout_prob = None
        if hasattr(drop_pipeline, "predict_proba"):
            probs = drop_pipeline.predict_proba(df_input)[0]
            classes = list(drop_pipeline.named_steps["model"].classes_)
            prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            if "Dropout" in prob_dict:
                dropout_prob = round(prob_dict["Dropout"] * 100, 1)

        # 2-Year Derived Analytics
        s1 = float(data.get("Sem1_Marks", 0))
        s2 = float(data.get("Sem2_Marks", 0))
        s3 = float(data.get("Sem3_Marks", 0))
        s4 = float(data.get("Sem4_Marks", 0))
        y1_avg = round((s1 + s2) / 2.0, 1)
        y2_avg = round((s3 + s4) / 2.0, 1)
        trend = round(y2_avg - y1_avg, 1)
        cum_2yr_marks = round((y1_avg + y2_avg) / 2.0, 1)
        cum_cgpa = round(cum_2yr_marks / 10.0, 2)

        y1_bl = int(data.get("Year1_Backlogs", 0))
        y2_bl = int(data.get("Year2_Backlogs", 0))
        total_backlogs = y1_bl + y2_bl

        # Academic Division / Honours
        if predicted_score >= 75.0:
            academic_division = "First Class with Distinction"
        elif predicted_score >= 60.0:
            academic_division = "First Class"
        elif predicted_score >= 50.0:
            academic_division = "Second Class"
        elif predicted_score >= 40.0:
            academic_division = "Pass Class"
        else:
            academic_division = "Academic Probation / Critical"

        # Overall Risk Level classification
        prob_val = dropout_prob if dropout_prob is not None else 0.0
        att_val = float(data.get("Attendance_Percentage", 100))

        if prob_val >= 45.0 or total_backlogs >= 4 or att_val < 60:
            risk_level = "High Risk"
            risk_color = "#dc2626"
        elif prob_val >= 20.0 or total_backlogs >= 2 or att_val < 75 or trend < -8.0:
            risk_level = "Moderate Risk"
            risk_color = "#d97706"
        else:
            risk_level = "Low Risk (On Track)"
            risk_color = "#16a34a"

        risk_factors = explain_risk_factors(data)
        faculty_solutions = generate_faculty_solutions(data, risk_level, total_backlogs)
        student_solutions = generate_student_solutions(data, risk_level, total_backlogs)

        res_payload = {
            "performance_prediction": f"{predicted_score:.1f}%",
            "predicted_cgpa": f"{predicted_score / 10.0:.2f} / 10.0",
            "academic_division": academic_division,
            "dropout_prediction": predicted_dropout,
            "dropout_probability": dropout_prob,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "analytics_summary": {
                "year1_average": f"{y1_avg:.1f}%",
                "year2_average": f"{y2_avg:.1f}%",
                "overall_2year_marks": f"{cum_2yr_marks:.1f}%",
                "overall_2year_cgpa": f"{cum_cgpa:.2f} / 10.0",
                "progression_trend": f"{'+' if trend >= 0 else ''}{trend:.1f}%",
                "total_backlogs": total_backlogs,
                "year1_backlogs": y1_bl,
                "year2_backlogs": y2_bl,
            },
            "risk_factors": risk_factors,
            "faculty_solutions": faculty_solutions,
            "student_solutions": student_solutions,
        }

        return jsonify(res_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    load_models()
    print("\nEngineering Student Success AI running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
