# College Student Performance & Dropout Risk Prediction System

## Project Overview

This project provides a practical, unified Machine Learning system that predicts **BOTH**:
1. **Academic Performance** (Continuous Exam Score 0–100)
2. **Dropout Risk Status** (`Graduate`, `Dropout`, `Enrolled` + Dropout Probability %)

It also includes an **Early Academic Support Layer**:
- **Why This Risk? (Risk Factors)**: Human-readable explanations derived from the student's actual input values and trained feature boundaries.
- **Personalized Suggestions**: Actionable academic advice customized to the student's situation.

Both predictions and explanations are evaluated simultaneously from a **SINGLE common set of 12 student study features**.

---

## Combined Dataset (`combined_student_dataset.csv`)

The two original datasets (`student_performance.csv` and `student_dropout.csv`) were cleaned, standardized, and aligned into `combined_student_dataset.csv`:

### Common Features (12 Study & College Factors)
- `Course_Program` (Degree / Major Field)
- `Sem1_Courses_Enrolled` (Subjects Enrolled)
- `Sem1_Courses_Approved` (Subjects Passed)
- `Study_Hours_Per_Week` (Weekly Self-Study Hours)
- `Attendance_Percentage` (Lecture Attendance Rate 0–100%)
- `Tutoring_Sessions` (Monthly Mentoring / Tutoring Sessions)
- `Motivation_Level` (Low / Medium / High)
- `Access_to_Resources` (Low / Medium / High)
- `Previous_Academic_Score` (Prior Marks / GPA 0–100)
- `Sem1_Average_Grade` (1st Semester Marks 0–20)
- `Tuition_Fees_Up_To_Date` (Payment Status: Yes / No)
- `Scholarship_Holder` (Recipient Status: Yes / No)

---

## Machine Learning Models & Evaluation

### 1. Performance Models (Regression Target: `Performance_Target`)
- **Linear Regression** (Selected Best Model: $R^2 = 0.6844$, MAE $= 1.0810$, RMSE $= 2.1121$)
- **Decision Tree Regressor** ($R^2 = 0.1881$, MAE $= 1.6826$, RMSE $= 3.3876$)
- **Random Forest Regressor** ($R^2 = 0.6119$, MAE $= 1.3060$, RMSE $= 2.3421$)

### 2. Dropout Models (Classification Target: `Dropout_Target`)
- **Random Forest Classifier** (Selected Best Model: Weighted F1 $= 0.7094$, Accuracy $= 0.7254$)
- **Logistic Regression** (Weighted F1 $= 0.6931$, Accuracy $= 0.7266$)
- **Decision Tree Classifier** (Weighted F1 $= 0.6852$, Accuracy $= 0.7062$)

---

## Installation & Setup

1. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python main.py
   ```

3. **Open Web Browser**:
   Navigate to `http://127.0.0.1:5000`
