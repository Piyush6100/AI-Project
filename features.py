import pandas as pd
import numpy as np


def add_engineered_features(df_in):
    """Engineers practical domain-specific indicators for Engineering students.
    
    Calculates 2-year marks metrics, progression trajectory, backlog load,
    and engagement indicators without administrative fee/scholarship dependencies.
    """
    d = df_in.copy()

    # 1. 2-Year Academic Marks Aggregates
    sem1 = d["Sem1_Marks"].clip(0, 100)
    sem2 = d["Sem2_Marks"].clip(0, 100)
    sem3 = d["Sem3_Marks"].clip(0, 100)
    sem4 = d["Sem4_Marks"].clip(0, 100)

    d["Year1_Average_Marks"] = (sem1 + sem2) / 2.0
    d["Year2_Average_Marks"] = (sem3 + sem4) / 2.0
    d["Overall_2Year_Aggregate"] = (d["Year1_Average_Marks"] + d["Year2_Average_Marks"]) / 2.0
    d["Overall_2Year_CGPA"] = d["Overall_2Year_Aggregate"] / 10.0

    # 2. Academic Progression & Trajectory (Year 2 vs Year 1 Trend)
    # Positive indicates improvement; negative indicates decline in harder core engineering subjects
    d["Academic_Progression_Trend"] = d["Year2_Average_Marks"] - d["Year1_Average_Marks"]

    # 3. Backlog Analytics
    y1_bl = d["Year1_Backlogs"].clip(lower=0)
    y2_bl = d["Year2_Backlogs"].clip(lower=0)
    d["Total_Backlogs"] = y1_bl + y2_bl
    d["Backlog_Clearance_Rate"] = (1.0 - (d["Total_Backlogs"] / 24.0)).clip(0.0, 1.0)

    # 4. Study & Engagement Products
    d["Attendance_Study_Product"] = (d["Attendance_Percentage"] / 100.0) * d["Study_Hours_Per_Week"]
    d["Academic_Momentum_Index"] = (
        (d["Previous_Academic_Score"] * 0.20)
        + (d["Year1_Average_Marks"] * 0.35)
        + (d["Year2_Average_Marks"] * 0.45)
    )

    # 5. Motivation and Resource numerical maps
    mot_map = {"Low": 0, "Medium": 1, "High": 2}
    res_map = {"Low": 0, "Medium": 1, "High": 2}
    mot_num = d["Motivation_Level"].map(mot_map).fillna(1)
    res_num = d["Access_to_Resources"].map(res_map).fillna(1)
    d["Engagement_Readiness_Score"] = (mot_num * 1.5) + (res_num * 1.5) + (d["Tutoring_Sessions"] * 0.5)

    return d
