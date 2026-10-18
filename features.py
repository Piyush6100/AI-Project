import pandas as pd


def add_engineered_features(df_in):
    """Engineers domain specific numerical indicators to increase model accuracy."""
    d = df_in.copy()
    enrolled = d["Sem1_Courses_Enrolled"].replace(0, 1)

    d["Course_Approval_Rate"] = (d["Sem1_Courses_Approved"] / enrolled).clip(0, 1)
    d["Pending_Backlogs"] = (d["Sem1_Courses_Enrolled"] - d["Sem1_Courses_Approved"]).clip(lower=0)
    d["Study_Effort_Per_Course"] = d["Study_Hours_Per_Week"] / enrolled
    d["Academic_Index"] = (d["Previous_Academic_Score"] * 0.4) + (d["Sem1_Average_Grade"] * 3.0)
    d["Attendance_Study_Product"] = (d["Attendance_Percentage"] / 100.0) * d["Study_Hours_Per_Week"]

    mot_map = {"Low": 0, "Medium": 1, "High": 2}
    res_map = {"Low": 0, "Medium": 1, "High": 2}
    fee_map = {"No": 0, "Yes": 1}
    sch_map = {"No": 0, "Yes": 1}

    mot_num = d["Motivation_Level"].map(mot_map).fillna(1)
    res_num = d["Access_to_Resources"].map(res_map).fillna(1)
    fee_num = d["Tuition_Fees_Up_To_Date"].map(fee_map).fillna(1)
    sch_num = d["Scholarship_Holder"].map(sch_map).fillna(0)
    d["Support_Score"] = mot_num + res_num + (fee_num * 2) + sch_num

    return d
