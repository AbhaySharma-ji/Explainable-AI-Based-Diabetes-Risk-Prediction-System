import streamlit as st
import pandas as pd
import joblib
import shap
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# LOAD MODEL
pipeline = joblib.load("diabetes_xai_scaled_model.pkl")

# =========================
# UI
st.set_page_config(page_title="Diabetes Risk System")
st.title("Diabetes Risk Explanation System")

# =========================
# INPUT
st.header("Patient Details")

name = st.text_input("Patient Name")

gender = st.selectbox("Gender", ["Female", "Male"])

if gender == "Female":
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
else:
    pregnancies = 0
    st.info("Pregnancy not applicable")

glucose = st.number_input(
    "Blood Sugar Level (mg/dL)",
    min_value=50.0,
    max_value=500.0,   
    value=100.0,
    help="Normal fasting: 70-100 | Diabetic: >126"
)

bp = st.number_input("Blood Pressure (mmHg)", 40.0, 200.0, 70.0)

# =========================
# OPTIONAL FIELDS
st.subheader("Advanced Health Data (Optional)")

use_advanced = st.checkbox("I have medical test data")

if use_advanced:
    skin = st.number_input("Skin Thickness (mm)", 0.0, 100.0, 20.0)
    insulin = st.number_input("Insulin (µU/ml)", 0.0, 900.0, 80.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
else:
    skin = 20.0
    insulin = 80.0
    dpf = 0.5

bmi = st.slider("BMI", 15.0, 45.0, 25.0)
age = st.number_input("Age", 10, 100, 30)

# =========================
# PDF FUNCTION
def generate_pdf(data, risk_percent, explanations, rec):

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Diabetes Risk Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    content.append(Spacer(1, 10))

    table_data = [["Field", "Value"]]
    for k, v in data.items():
        table_data.append([k, str(v)])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    content.append(Paragraph("Patient Details", styles["Heading2"]))
    content.append(table)
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Risk Level: {risk_percent:.1f}%", styles["Heading2"]))

    content.append(Paragraph("Key Factors:", styles["Heading2"]))
    for e in explanations:
        content.append(Paragraph(f"• {e}", styles["Normal"]))

    content.append(Paragraph("Recommendations:", styles["Heading2"]))
    for r in rec:
        content.append(Paragraph(f"• {r}", styles["Normal"]))

    doc.build(content)
    return "report.pdf"

# =========================
# ANALYZE
if st.button("Analyze Diabetes Risk"):

    sample = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    risk = pipeline.predict_proba(sample)[0][1]
    risk_percent = risk * 100

    st.subheader(f"Risk: {risk_percent:.1f}%")

    # =========================
    # CHART
    st.subheader("Health Parameters")

    chart_data = pd.DataFrame({
        "Parameter": ["Glucose", "BMI", "BP", "Age"],
        "Value": [glucose, bmi, bp, age]
    })

    st.bar_chart(chart_data.set_index("Parameter"))

    # =========================
    # SHAP
    explanations = []

    try:
        X = pipeline[:-1].transform(sample)
        model = pipeline.named_steps["model"]

        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        vals = shap_values.values

        if len(vals.shape) == 3:
            vals = vals[:, :, 1]
        vals = vals[0]

        features = sample.columns

        shap_df = pd.DataFrame({
            "Feature": features,
            "Impact": vals
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(shap_df.set_index("Feature")["Impact"].abs())

        st.subheader("Why this result?")

        for f, v in shap_df.values[:3]:
            if f == "Glucose":
                explanations.append("High blood sugar is increasing your risk.")
            elif f == "BMI":
                explanations.append("High BMI is increasing your risk.")
            elif f == "Age":
                explanations.append("Age contributes to higher risk.")
            elif f == "BloodPressure":
                explanations.append("High BP increases risk.")

        for e in explanations:
            st.write("•", e)

    except Exception as e:
        st.error(f"SHAP Error: {e}")
        explanations.append("Explanation not available")

    # =========================
    # RECOMMENDATIONS
    # =========================
    rec = []

    if glucose > 140:
        rec.append("Control blood sugar levels.")
    if bmi > 30:
        rec.append("Reduce weight.")
    if age > 50:
        rec.append("Regular health checkups.")

    if risk > 0.6:
        rec.append("Consult a doctor immediately.")
    else:
        rec.append("Maintain healthy lifestyle.")

    st.subheader("Recommendations")
    for r in rec:
        st.write("•", r)

    # =========================
    # PDF
    # =========================
    patient_data = {
        "Name": name,
        "Gender": gender,
        "Glucose": glucose,
        "BP": bp,
        "BMI": bmi,
        "Age": age
    }

    pdf = generate_pdf(patient_data, risk_percent, explanations, rec)

    with open(pdf, "rb") as f:
        st.download_button("Download Report", f, file_name="diabetes_report.pdf")