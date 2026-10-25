# Engineering College Student Performance & Retention Prediction System

## 1. Project Overview

This project provides an AI-powered academic performance forecasting and retention risk diagnosis system designed specifically for **Engineering Colleges and Universities**.

It evaluates engineering students across **2-Year Semester Marks (Semesters 1, 2, 3, and 4)**, **active backlogs**, and **practical laboratory/attendance factors** to deliver:
1. **Academic Performance Forecasting**: Continuous Exam Score (0–100%), Projected Final CGPA (out of 10.0), and Academic Honours Division (First Class with Distinction, First Class, Second Class, etc.).
2. **Retention & Dropout Risk Status**: Multi-class Status (`Graduate` / On Track, `Enrolled` / Needs Advisory, `Dropout` Risk) + Calibrated Dropout Risk %.
3. **2-Year Progression & Trajectory Analytics**: Real-time evaluation of Year 1 vs Year 2 trends to detect students struggling with departmental core engineering subjects.
4. **Actionable Dual Solutions**:
   - **For Engineering Faculties & Mentors**: Institutional early warnings, departmental counseling, remedial tutorial batches in core subjects (Engineering Mathematics, Circuits, Mechanics, Coding), and exam eligibility notices.
   - **For Students**: Step-by-step roadmap for clearing pending arrears/backlogs, dedicated weekly study/lab practice targets, and campus placement CGPA goals.

---

## 2. Engineering Curriculum & 2-Year Feature Architecture

All administrative noise (such as *Tuition Fees Up to Date* and *Scholarship Holder*) has been removed to maintain 100% focus on practical academic metrics.

### Engineering Programs / Branches
- Computer Science & Engineering (CSE)
- Information Technology (IT)
- Electronics & Communication Engineering (ECE)
- Electrical & Electronics Engineering (EEE)
- Mechanical Engineering (ME)
- Civil Engineering (CE)
- Artificial Intelligence & Data Science (AI & DS)
- Chemical Engineering (ChemE)

### 2-Year Academic Marks & Curricular Features
- **1st Year (Foundational Sciences & Engineering Basics)**:
  - `Sem1_Marks` (0–100%)
  - `Sem2_Marks` (0–100%)
  - `Year1_Backlogs` (Active uncleared arrears)
- **2nd Year (Core Departmental Engineering Subjects)**:
  - `Sem3_Marks` (0–100%)
  - `Sem4_Marks` (0–100%)
  - `Year2_Backlogs` (Active uncleared arrears)
- **Engineered Academic Progression Metrics**:
  - `Year1_Average_Marks` & `Year2_Average_Marks`
  - `Academic_Progression_Trend` ($Year2 - Year1$)
  - `Overall_2Year_Aggregate` & `Cumulative_CGPA`
  - `Total_Backlogs` ($Year1 + Year2$)
- **Practical Engagement & Study Factors**:
  - `Attendance_Percentage` (Lecture & Laboratory Attendance, mandatory $\ge 75\%$ institutional threshold)
  - `Study_Hours_Per_Week` (Weekly problem solving & programming practice)
  - `Tutoring_Sessions` (Monthly faculty doubt-clearing and lab remedial sessions)
  - `Previous_Academic_Score` (Prior 12th / Diploma / Entrance score 0–100)
  - `Motivation_Level` (Low / Medium / High)
  - `Access_to_Resources` (Low / Medium / High access to computing labs, software, and library)

---

## 3. Machine Learning Model Performance

The system uses advanced ensemble pipelines trained on harmonized student datasets (`combined_student_dataset.csv`, 11,031 records):

| Model Task | Algorithm | Metric 1 | Metric 2 |
| :--- | :--- | :--- | :--- |
| **Academic Performance Regressor** | Histogram Gradient Boosting | **$R^2 = 0.9075$** | **RMSE = $1.14$** |
| **Dropout & Retention Classifier** | Histogram Gradient Boosting | **Accuracy = $89.60\%$** | **Weighted $F_1 = 0.8917$** |

---

## 4. Installation & Setup

1. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train & update ML models** (optional; pre-trained `.pkl` files are included):
   ```bash
   python train.py
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

4. **Access the Web Portal**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
