import numpy as np
import pandas as pd

def calculate_asdas(
        back_pain, 
        peripheral_pain, 
        patient_global, 
        stiffness_duration, 
        crp):
    return (0.12 * back_pain 
            + 0.07 * peripheral_pain 
            + 0.11 * patient_global
            + 0.06 * stiffness_duration 
            + 0.58 * np.log(crp + 1) 
            )


def calculate_asdas_esr(
        back_pain, 
        peripheral_pain, 
        patient_global, 
        stiffness_duration, 
        esr):
    return (0.08 * back_pain 
            + 0.09 * peripheral_pain 
            + 0.11 * patient_global
            + 0.07 * stiffness_duration 
            + 0.29 * np.sqrt(esr)
            )

def calculate_basdai(
        fatigue, 
        back_pain,
        peripheral_pain, 
        tenderness, 
        stiffness_severity, 
        stiffness_duration):
    return (fatigue 
            + back_pain 
            + peripheral_pain 
            + tenderness 
            + (stiffness_severity + stiffness_duration) / 2) / 5

def asdas_activity_state(score):
    if pd.isna(score):
        return pd.NA
    if score < 1.3:
        return "inactive"
    if score < 2.1:
        return "low"
    if score <= 3.5:
        return "high"
    return "very high"

def basdai_activity_state(score):
    if pd.isna(score):
        return pd.NA
    if score < 4:
        return "low"
    return "high"