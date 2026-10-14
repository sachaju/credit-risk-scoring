from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap

app = FastAPI(title="Credit Risk Scoring API")

# Loaded once when the server starts, not on every request — this is important for speed
model = joblib.load('model.pkl')
categorical_features = joblib.load('categorical_features.pkl')
feature_columns = joblib.load('feature_columns.pkl')
explainer = shap.TreeExplainer(model)


class ApplicantData(BaseModel):
    NAME_CONTRACT_TYPE: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_TYPE_SUITE: str
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: float
    FLAG_MOBIL: int
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: str
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: str
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    APARTMENTS_AVG: float
    BASEMENTAREA_AVG: float
    YEARS_BEGINEXPLUATATION_AVG: float
    YEARS_BUILD_AVG: float
    COMMONAREA_AVG: float
    ELEVATORS_AVG: float
    ENTRANCES_AVG: float
    FLOORSMAX_AVG: float
    FLOORSMIN_AVG: float
    LANDAREA_AVG: float
    LIVINGAPARTMENTS_AVG: float
    LIVINGAREA_AVG: float
    NONLIVINGAPARTMENTS_AVG: float
    NONLIVINGAREA_AVG: float
    APARTMENTS_MODE: float
    BASEMENTAREA_MODE: float
    YEARS_BEGINEXPLUATATION_MODE: float
    YEARS_BUILD_MODE: float
    COMMONAREA_MODE: float
    ELEVATORS_MODE: float
    ENTRANCES_MODE: float
    FLOORSMAX_MODE: float
    FLOORSMIN_MODE: float
    LANDAREA_MODE: float
    LIVINGAPARTMENTS_MODE: float
    LIVINGAREA_MODE: float
    NONLIVINGAPARTMENTS_MODE: float
    NONLIVINGAREA_MODE: float
    APARTMENTS_MEDI: float
    BASEMENTAREA_MEDI: float
    YEARS_BEGINEXPLUATATION_MEDI: float
    YEARS_BUILD_MEDI: float
    COMMONAREA_MEDI: float
    ELEVATORS_MEDI: float
    ENTRANCES_MEDI: float
    FLOORSMAX_MEDI: float
    FLOORSMIN_MEDI: float
    LANDAREA_MEDI: float
    LIVINGAPARTMENTS_MEDI: float
    LIVINGAREA_MEDI: float
    NONLIVINGAPARTMENTS_MEDI: float
    NONLIVINGAREA_MEDI: float
    FONDKAPREMONT_MODE: str
    HOUSETYPE_MODE: str
    TOTALAREA_MODE: float
    WALLSMATERIAL_MODE: str
    EMERGENCYSTATE_MODE: str
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_2: int
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_4: int
    FLAG_DOCUMENT_5: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_7: int
    FLAG_DOCUMENT_8: int
    FLAG_DOCUMENT_9: int
    FLAG_DOCUMENT_10: int
    FLAG_DOCUMENT_11: int
    FLAG_DOCUMENT_12: int
    FLAG_DOCUMENT_13: int
    FLAG_DOCUMENT_14: int
    FLAG_DOCUMENT_15: int
    FLAG_DOCUMENT_16: int
    FLAG_DOCUMENT_17: int
    FLAG_DOCUMENT_18: int
    FLAG_DOCUMENT_19: int
    FLAG_DOCUMENT_20: int
    FLAG_DOCUMENT_21: int
    AMT_REQ_CREDIT_BUREAU_HOUR: float
    AMT_REQ_CREDIT_BUREAU_DAY: float
    AMT_REQ_CREDIT_BUREAU_WEEK: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    bureau_count: float
    bureau_active_count: float
    bureau_days_credit_mean: float
    bureau_days_credit_min: float
    bureau_credit_sum_total: float
    bureau_credit_sum_mean: float
    bureau_debt_sum_total: float
    bureau_overdue_max: float
    bureau_overdue_mean: float
    bureau_debt_credit_ratio: float
    prev_count: float
    prev_approved_count: float
    prev_refused_count: float
    prev_canceled_count: float
    prev_amt_credit_mean: float
    prev_amt_credit_max: float
    prev_amt_annuity_mean: float
    prev_days_decision_mean: float
    prev_cnt_payment_mean: float
    prev_refused_ratio: float
    install_count: float
    install_delay_mean: float
    install_delay_max: float
    install_late_count: float
    install_ratio_mean: float
    install_ratio_min: float
    install_late_ratio: float
    pos_count: float
    pos_dpd_mean: float
    pos_dpd_max: float
    pos_cnt_installment_mean: float
    cc_count: float
    cc_balance_mean: float
    cc_balance_max: float
    cc_credit_limit_mean: float
    cc_dpd_mean: float
    cc_drawings_atm_mean: float
    cc_utilization: float
    is_days_employed_anomaly: int


@app.post("/predict")
def predict(applicant: ApplicantData):
    # Convert the incoming request into a DataFrame with exactly one row
    input_df = pd.DataFrame([applicant.dict()])

    # Enforce the same column order the model was trained on — critical, as discussed earlier
    input_df = input_df[feature_columns]

    # Re-apply category dtype to the same columns we treated as categorical during training
    for col in categorical_features:
        input_df[col] = input_df[col].astype('category')

    # Get the prediction
    probability = model.predict_proba(input_df)[0, 1]

    # Get SHAP values for this one applicant, for explainability
    shap_values = explainer.shap_values(input_df)
    top_features = pd.DataFrame({
        'feature': feature_columns,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', key=abs, ascending=False).head(5)

    return {
        "default_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.15 else "Medium" if probability > 0.08 else "Low",
        "top_factors": [
            {"feature": row['feature'], "impact": round(float(row['shap_value']), 4)}
            for _, row in top_features.iterrows()
        ]
    }