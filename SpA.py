import numpy as np
import pandas as pd

# ASDAS based on CRP (ASAS-preferred version): combines four patient-reported
# measures scored from 0 to 10 with CRP (mg/L). Per the ASAS calculator, CRP
# values below 2 mg/L are entered as 2 mg/L.
# References: ASAS ASDAS Calculator, https://www.asas-group.org/instruments/asdas-calculator/;
# Lukas C, et al. Ann Rheum Dis. 2009;68:18–24. doi:10.1136/ard.2008.094870.
def calculate_asdas(
        back_pain, 
        peripheral_pain, 
        patient_global, 
        stiffness_duration, 
        crp):
    crp = np.maximum(crp, 2)
    return (0.12 * back_pain 
            + 0.07 * peripheral_pain 
            + 0.11 * patient_global
            + 0.06 * stiffness_duration 
            + 0.58 * np.log(crp + 1) 
            )


# ASDAS-ESR (alternative version): uses ESR (mm/h) instead of CRP; the four
# patient-reported measures are scored from 0 to 10.
# References: ASAS ASDAS Calculator, https://www.asas-group.org/instruments/asdas-calculator/;
# Lukas C, et al. Ann Rheum Dis. 2009;68:18–24. doi:10.1136/ard.2008.094870.
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

# BASDAI: six self-reported items scored from 0 to 10 (fatigue, back pain,
# peripheral joint pain/swelling, tenderness, and morning stiffness severity
# and duration). Average the last two items, then average that result with
# the first four items.
# Reference: Garrett S, et al. J Rheumatol. 1994;21(12):2286–2291. PMID:7699630.
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

# ASDAS disease activity categories: <1.3 inactive, 1.3–<2.1 low,
# 2.1–3.5 high, and >3.5 very high.
# Reference: ASAS ASDAS Calculator, https://www.asas-group.org/instruments/asdas-calculator/.
def asdas_activity_state(score):
    if isinstance(score, pd.Series):
        result = pd.Series(pd.NA, index=score.index, dtype="string", name=score.name)
        valid = score.notna()
        result.loc[valid & score.lt(1.3)] = "inactive"
        result.loc[valid & score.ge(1.3) & score.lt(2.1)] = "low"
        result.loc[valid & score.ge(2.1) & score.le(3.5)] = "high"
        result.loc[valid & score.gt(3.5)] = "very high"
        return result
    if pd.isna(score):
        return pd.NA
    if score < 1.3:
        return "inactive"
    if score < 2.1:
        return "low"
    if score <= 3.5:
        return "high"
    return "very high"

# BASDAI activity categories in this function use a threshold of 4:
# <4 is "low" and >=4 is "high". The ASAS-EULAR guideline uses BASDAI >=4
# as an alternative high disease activity criterion when ASDAS is unavailable.
# Reference: Ramiro S, et al. Ann Rheum Dis. 2023;82:19–34.
# doi:10.1136/ard-2022-223296.
def basdai_activity_state(score):
    if isinstance(score, pd.Series):
        result = pd.Series(pd.NA, index=score.index, dtype="string", name=score.name)
        valid = score.notna()
        result.loc[valid & score.lt(4)] = "low"
        result.loc[valid & ~score.lt(4)] = "high"
        return result
    if pd.isna(score):
        return pd.NA
    if score < 4:
        return "low"
    return "high"
