"""
Created on Sat May 23 16:00:17 2026

@author: magal
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
#import os
# Preprocesamiento
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline


# Initial web page configuration (Wide layout to optimize visualization of charts)
st.set_page_config(page_title="Academic Alert System", layout="wide")


# =====================================================================
# 1. DATA LOADING AND TRAINING OF THE TWO MODELS
# =====================================================================
@st.cache_resource
# =====================================================================
# 2. DATA PREPARATION
# =====================================================================

def preparar_datos_para_lmm(df_raw):
    df_proc = df_raw.copy()
    
    # === STEP 1: Homogenize grade scales ===
    # Map both grades to a standard scale (base 10) to ensure mathematical comparability
    df_proc['N1_C'] = (df_proc['N1']*10/8).astype(int)
    # Optional: Generate a combined final grade (e.g., average score)
    df_proc['SUM'] = df_proc['N1_C'] + df_proc['N2']

    # === STEP 2: Encode categorical/text variables into numerical format ===
    # Apply Label Encoding for provinces, subjects, and gender attributes
    le_province = LabelEncoder()
    df_proc['COD_PRO'] = le_province.fit_transform(df_proc['COD_PRO'])
    
    le_subject = LabelEncoder()
    df_proc['ID_MAT'] = le_subject.fit_transform(df_proc['ID_MAT'])

    le_sx = LabelEncoder()
    df_proc['ID_SEX'] = le_sx.fit_transform(df_proc['ID_SEX'])
    le_es = LabelEncoder()
    df_proc['COD_EST'] = le_es.fit_transform(df_proc['COD_EST'])
  
    # Extract numerical value from the curricular level (e.g., map "Level 7" to integer 7)
    df_proc['COD_NI'] = df_proc['COD_NI'].astype('category').cat.codes
    
    # Extract numerical value from the academic period (e.g., map "Period 1" to integer 1)
    df_proc['PE_ORD'] = df_proc['ID_PE'].astype('category').cat.codes
 
    return df_proc

# Execute data preprocessing pipeline
# df_listo = preparar_datos_para_lmm(df)

# =====================================================================
# 3. GRAPHICAL USER INTERFACE (DASHBOARD INITIALIZATION)
# =====================================================================

def inicializar_sistema():
    df = pd.read_excel("archivo.xlsx")
    
    df1 = preparar_datos_para_lmm(df)
    # --- STATISTICAL INFRASTRUCTURE (Linear Mixed Model - LMM) ---
    df_mlm = df1.copy()
        
    formula_mlm = "SUM ~ np.log1p(ED_EST)+ PE_ORD + COD_NI + C(ID_SEX) + C(COD_PRO)"
    modelo_mlm = smf.mixedlm(formula_mlm, data=df_mlm, groups=df_mlm["COD_EST"]).fit(reml=True)
    
    # --- PREDICTIVE INFRASTRUCTURE (Unified Random Forest Pipeline) ---
    X = df1[["PE_ORD", "COD_NI","ID_MAT","ED_EST", "ID_SEX", "COD_PRO"]] 
    y = df1["SUM"]
    
    # Define multi-type transformations for feature processing
    preprocessor = ColumnTransformer(
           transformers=[
            ('num', StandardScaler(), ["ED_EST"]),
            ('cat', OneHotEncoder(drop=None, handle_unknown='ignore', min_frequency=10), 
             ["ID_SEX", "PE_ORD", "COD_NI", "COD_PRO"]), # Adjusted strings to match input features
            ('cat_mid', OneHotEncoder(drop=None, handle_unknown='ignore', min_frequency=0.01), 
             ["ID_MAT"])
        ]
    )
    
    pipeline_rf = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    pipeline_rf.fit(X, y)
    
    return df1, modelo_mlm, pipeline_rf

df1, modelo_mlm, pipeline_rf = inicializar_sistema()

# =====================================================================
# 4. USER INTERFACE RENDERING (DASHBOARD)
# =====================================================================
st.title("🛡️ Intelligent Academic Diagnosis and Early Alert Platform")
st.markdown("This platform integrates Statistical Rigor (Linear Mixed Model) and Predictive Performance (Random Forest) to explain academic achievement.")
st.markdown("---")

# Lateral sidebar for handling student data profile selection
st.sidebar.header("🔍 Student Profile Selector")
modo_ingreso = st.sidebar.radio("Data Source:", ["Select from University Historical Record", "Simulate New Profile"])

if modo_ingreso == "Select from University Historical Record":
    # Retrieve features from an empirical student instance in the database
    estudiante_id = st.sidebar.selectbox("Student Code ID:", df1["COD_EST"].unique())
    datos_estudiante = df1[df1["COD_EST"] == estudiante_id].iloc[0]
    
    edad = int(datos_estudiante["ED_EST"])
    sexo = int(datos_estudiante["ID_SEX"])
    periodo = int(datos_estudiante["PE_ORD"])
    nivel = int(datos_estudiante["COD_NI"])
    provincia = int(datos_estudiante["COD_PRO"])
    materia = int(datos_estudiante["ID_MAT"])
else:
    # Interactive widgets for custom scenario simulation
    edad = st.sidebar.slider("Student Age (ED_EST):", 17, 50, 22)
    sexo = st.sidebar.selectbox("Gender Identity (ID_SEX):", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    periodo = st.sidebar.selectbox("Academic Period Index (ID_PE):", [0, 1, 2])
    nivel = st.sidebar.slider("Curricular Level (COD_NI):", 7, 13, 9)
    provincia = st.sidebar.number_input("Province Code (COD_PRO):", 1, 24, 2)
    materia = st.sidebar.number_input("Subject Matter Code (ID_MAT):", 1, 30, 12)

# Consolidate target features into an evaluation DataFrame
X_input = pd.DataFrame([{
    "ED_EST": edad, "ID_SEX": sexo, "PE_ORD": periodo, 
    "COD_NI": nivel, "COD_PRO": provincia, "ID_MAT": materia
}])

# =====================================================================
# 5. INFERENCE CALCULATION AND RISK LEVEL TRIGGER MECHANISM
# =====================================================================
nota_predicha = float(pipeline_rf.predict(X_input)[0])

st.subheader("📊 Diagnostic Summary and Risk Matrix")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Model-Estimated Academic Score", value=f"{nota_predicha:.2f} / 100")

with col2:
    if nota_predicha < 70:
        st.error("🚨 CRITICAL ALERT: HIGH RISK OF FAILURE")
        nivel_riesgo = "High"
    elif 70 <= nota_predicha < 80:
        st.warning("⚠️ PREVENTIVE ALERT: MODERATE ACADEMIC RISK")
        nivel_riesgo = "Moderate"
    else:
        st.success("✅ SECURE STATUS: OPTIMISTIC PERFORMANCE PROJECTION")
        nivel_riesgo = "Low"

with col3:
    st.write("**Suggested Strategic Intervention:**")
    if nivel_riesgo == "High":
        st.write("Immediate Intervention: Mandatory peer-tutoring assignment and pedagogical review.")
    elif nivel_riesgo == "Moderate":
        st.write("Active Monitoring: Continuous virtual platform engagement tracking and early evaluation.")
    else:
        st.write("Maintain Strategy: Profile optimal for academic leadership or peer-mentorship roles.")

st.markdown("---")

# =====================================================================
# 6. INTEGRATED SHAP EXPLAINABILITY PIPELINE
# =====================================================================
st.subheader("🧬 Integrated Causal Analysis via SHAP")
st.write("Utilize the tabs below to audit the root factors determining the student's score from both empirical frameworks.")

tab1, tab2, tab3 = st.tabs(["🏛️ Statistical SHAP (LMM Inference)", "🤖 Predictive SHAP (Random Forest)", "🔄 SHAP Fusion (Root Cause Synthesis)"])

# --- TAB 1: SHAP EXPLANATIONS FOR THE LINEAR MIXED MODEL ---
with tab1:
    st.markdown("#### Factor Attribution at Institutional Macro-Structural Level")
    
    # Reconstruct the structural fixed-effects design matrix from the statistical model
    X_fijos_base = pd.DataFrame(modelo_mlm.model.exog, columns=modelo_mlm.model.exog_names)
    if "Intercept" in X_fijos_base.columns:
        X_fijos_base = X_fijos_base.drop(columns=["Intercept"])
        
    # Standardize input instance across the categories mapping
    fila_mlm_input = pd.DataFrame(0.0, index=[0], columns=X_fijos_base.columns)
    if "np.log1p(ED_EST)" in fila_mlm_input.columns: fila_mlm_input["np.log1p(ED_EST)"] = np.log1p(edad)
    if "PE_ORD" in fila_mlm_input.columns: fila_mlm_input["PE_ORD"] = periodo
    if "COD_NI" in fila_mlm_input.columns: fila_mlm_input["COD_NI"] = nivel
    if f"C(ID_SEX)[T.{sexo}]" in fila_mlm_input.columns: fila_mlm_input[f"C(ID_SEX)[T.{sexo}]"] = 1.0
    if f"C(COD_PRO)[T.{provincia}]" in fila_mlm_input.columns: fila_mlm_input[f"C(COD_PRO)[T.{provincia}]"] = 1.0

    # Initialize LinearExplainer using estimated fixed beta coefficients from LMM
    coeficientes = modelo_mlm.fe_params[X_fijos_base.columns].values
    explainer_mlm = shap.LinearExplainer((coeficientes, modelo_mlm.fe_params["Intercept"]), X_fijos_base)
    shap_values_mlm = explainer_mlm(fila_mlm_input)

    # Sanitize and simplify internal feature tokens for user visualization
    shap_values_mlm.feature_names = [c.replace("C(", "").replace(")[T.", " ID:").replace("]", "") for c in X_fijos_base.columns]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    shap.plots.waterfall(shap_values_mlm[0], max_display=6, show=False)
    st.pyplot(fig)

# --- TAB 2: SHAP EXPLANATIONS FOR RANDOM FOREST ---
with tab2:
    st.markdown("#### Non-Linear Pattern Attribution Based on Machine Learning (XAI)")
    
    prep_step = pipeline_rf.named_steps["prep"]
    model_step = pipeline_rf.named_steps["model"]
    
    # Map raw features into the multi-dimensional expanded space of the pipeline preprocessor
    X_input_trans = prep_step.transform(X_input)
    if hasattr(X_input_trans, "toarray"): X_input_trans = X_input_trans.toarray()
    
    cols_rf = prep_step.get_feature_names_out()
    X_input_trans_df = pd.DataFrame(X_input_trans, columns=cols_rf)
    
    # Compute a representative background sample to condition the TreeExplainer framework
    X_train_trans = prep_step.transform(df1[["ED_EST", "ID_SEX", "PE_ORD", "COD_NI", "COD_PRO", "ID_MAT"]])
    if hasattr(X_train_trans, "toarray"): X_train_trans = X_train_trans.toarray()
    X_bg = shap.sample(X_train_trans, 50, random_state=42)
    
    # Instantiate stable interventional TreeExplainer for ensemble forest architecture
    explainer_rf = shap.TreeExplainer(model_step, data=X_bg, feature_perturbation="interventional")
    shap_values_rf = explainer_rf(X_input_trans_df)
    
    # Clean feature labels by isolating step namespaces
    shap_values_rf.feature_names = [c.split("__")[-1] for c in cols_rf]

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    shap.plots.waterfall(shap_values_rf[0], max_display=6, show=False)
    st.pyplot(fig2)

# --- TAB 3: HYBRID ROOT CAUSE SYNTHESIS FUSION ---
with tab3:
    st.markdown("#### Consolidated Causal Attribution Matrix")
    st.write("Aggregates partial dummy variable weightings to yield net feature importance vectors across the 6 primary dimensions.")
    
    # Compress expanded fixed-effect vectors back into base model features (LMM)
    impactos_mlm = {
        "ED_EST": float(shap_values_mlm.values[0][X_fijos_base.columns.get_loc("np.log1p(ED_EST)")]) if "np.log1p(ED_EST)" in X_fijos_base.columns else 0.0,
        "ID_SEX": float(np.sum([shap_values_mlm.values[0][i] for i, c in enumerate(X_fijos_base.columns) if "ID_SEX" in c])),
        "ID_PE": float(shap_values_mlm.values[0][X_fijos_base.columns.get_loc("PE_ORD")]) if "PE_ORD" in X_fijos_base.columns else 0.0,
        "COD_NI": float(shap_values_mlm.values[0][X_fijos_base.columns.get_loc("COD_NI")]) if "COD_NI" in X_fijos_base.columns else 0.0,
        "COD_PRO": float(np.sum([shap_values_mlm.values[0][i] for i, c in enumerate(X_fijos_base.columns) if "COD_PRO" in c])),
        "ID_MAT": 0.0 # Variable acts purely as random effect in the provided fixed formula context
    }
    
    # Compress expanded hot-encoded columns back into base features (Random Forest)
    impactos_rf = {
        "ED_EST": float(np.sum([shap_values_rf.values[0][i] for i, c in enumerate(cols_rf) if "ED_EST" in c])),
        "ID_SEX": float(np.sum([shap_values_rf.values[0][i] for i, c in enumerate(cols_rf) if "ID_SEX" in c])),
        "ID_PE": float(np.sum([shap_values_rf.values[0][i] for i, c in enumerate(cols_rf) if "PE_ORD" in c])),
        "COD_NI": float(np.sum([shap_values_rf.values[0][i] for i, c in enumerate(cols_rf) if "COD_NI" in c])),
        "COD_PRO": float(np.sum([shap_values_rf.values[0][i] for i, c in enumerate(cols_rf) if "COD_PRO" in c])),
        "ID_MAT": float(np.sum([shap_values_rf.values[0][i] for i, c in enumerate(cols_rf) if "ID_MAT" in c]))
    }
    
    # Build a strategic data matrix cross-referencing both modeling approaches
    df_fusion = pd.DataFrame({
        "Statistical SHAP (LMM)": impactos_mlm,
        "Predictive SHAP (Random Forest)": impactos_rf
    })
    
    # Compute unified prioritized intervention score
    df_fusion["Priority Index"] = df_fusion.mean(axis=1)
    
    # Render aesthetic color styling vectors mapping performance risks
    def pintar_celdas(val):
        return f'background-color: {"#ffcccc" if val < 0 else "#ccffcc"}'
    
    st.dataframe(df_fusion.style.applymap(pintar_celdas, subset=["Statistical SHAP (LMM)", "Predictive SHAP (Random Forest)"]))
    
    # Automatic data-driven clinical diagnostic logic
    causas_reales = df_fusion[df_fusion["Priority Index"] < 0].sort_values(by="Priority Index")
    
    if not causas_reales.empty:
        st.error(f"🔍 **Clinical Root Cause Analysis:** The leading structural dimension depressing this student's performance is **{causas_reales.index[0]}** (Consolidated Score: {causas_reales['Priority Index'].iloc[0]:.2f}). The academic tutor should focus intervention efforts specifically toward mitigating the impacts of this variable.")
    else:
        st.success("✨ No active academic risk factors or negative causal determinants were detected in the evaluated student profile.")
