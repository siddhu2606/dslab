import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Alzheimer's Analytics", page_icon="🧠", layout="wide")

CSV = "alzheimers_disease_data.csv"
TARGET = "Diagnosis"
LABELS = {0: "No Alzheimer's", 1: "Alzheimer's"}
COLORS = {"No Alzheimer's": "#4C78A8", "Alzheimer's": "#E45756"}
BINARY = ["Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
          "Depression", "HeadInjury", "Hypertension", "MemoryComplaints", "BehavioralProblems",
          "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
          "Forgetfulness"]
DROP = ["PatientID", "DoctorInCharge"]


@st.cache_data
def load(src):
    df = pd.read_csv(src)
    # the sheet has stray summary columns on the right (skewness/mean/median notes) - keep real columns only
    df = df.iloc[:, :35].dropna(subset=[TARGET]).drop(columns=DROP, errors="ignore")
    df["Gender"] = df["Gender"].astype(int)
    df["Diagnosis_Label"] = df[TARGET].map(LABELS)
    return df


@st.cache_resource
def train(df, model_name, test_size, seed):
    X = df.drop(columns=[TARGET, "Diagnosis_Label"])
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if model_name == "Random Forest":
        m = RandomForestClassifier(n_estimators=150, random_state=seed)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    return m, X, Xte, yte, proba


# ---------------- DATA + SIDEBAR FILTERS (everything below reacts live) ----------------
upload = st.sidebar.file_uploader("Use another CSV (optional)", type="csv")
try:
    full = load(upload if upload else CSV)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.sidebar.header("Filters")
a_lo, a_hi = int(full.Age.min()), int(full.Age.max())
age = st.sidebar.slider("Age", a_lo, a_hi, (a_lo, a_hi))
gender = st.sidebar.multiselect("Gender", [0, 1], default=[0, 1],
                                format_func=lambda g: "Male" if g == 0 else "Female")
diag = st.sidebar.multiselect("Diagnosis", list(LABELS.values()), default=list(LABELS.values()))
flags = st.sidebar.multiselect("Must have condition", [c for c in BINARY if c != "Gender"])

df = full[full.Age.between(*age) & full.Gender.isin(gender) & full.Diagnosis_Label.isin(diag)]
for f in flags:
    df = df[df[f] == 1]

if df.empty:
    st.warning("No patients match these filters.")
    st.stop()

num_cols = [c for c in df.columns if c not in (TARGET, "Diagnosis_Label")]
cont_cols = [c for c in num_cols if full[c].nunique() > 10]

st.title("🧠 Alzheimer's Disease Analytics")
st.caption(f"Showing **{len(df):,}** of {len(full):,} patients · all charts update live with the sidebar filters")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Explorer", "Distributions & Outliers", "Correlations", "Prediction Model"])

# ---------------- OVERVIEW ----------------
with tab1:
    pos = (df[TARGET] == 1).mean() * 100
    c = st.columns(5)
    c[0].metric("Patients", f"{len(df):,}")
    c[1].metric("Alzheimer's rate", f"{pos:.1f}%", f"{pos - (full[TARGET] == 1).mean() * 100:+.1f} pts vs all", delta_color="inverse")
    c[2].metric("Avg age", f"{df.Age.mean():.1f}")
    c[3].metric("Avg MMSE", f"{df.MMSE.mean():.1f}")
    c[4].metric("Avg BMI", f"{df.BMI.mean():.1f}")

    l, r = st.columns(2)
    with l:
        cnt = df.Diagnosis_Label.value_counts().reset_index()
        fig = px.pie(cnt, names="Diagnosis_Label", values="count", hole=.55,
                     color="Diagnosis_Label", color_discrete_map=COLORS, title="Diagnosis split")
        st.plotly_chart(fig, width="stretch")
    with r:
        d2 = df.assign(AgeGroup=pd.cut(df.Age, [59, 65, 70, 75, 80, 85, 90],
                                       labels=["60-65", "66-70", "71-75", "76-80", "81-85", "86-90"]))
        g = d2.groupby("AgeGroup", observed=True)[TARGET].mean().mul(100).reset_index()
        fig = px.bar(g, x="AgeGroup", y=TARGET, title="Alzheimer's rate by age group (%)",
                     labels={TARGET: "Rate %"}, color_discrete_sequence=["#E45756"])
        st.plotly_chart(fig, width="stretch")

    # prevalence by condition
    rows = []
    for f in [c for c in BINARY if c != "Gender"]:
        for v in (0, 1):
            s = df[df[f] == v]
            if len(s):
                rows.append((f, "Yes" if v else "No", s[TARGET].mean() * 100))
    pv = pd.DataFrame(rows, columns=["Factor", "Present", "Rate"])
    fig = px.bar(pv, x="Factor", y="Rate", color="Present", barmode="group",
                 title="Alzheimer's rate (%) with vs without each factor",
                 color_discrete_map={"Yes": "#E45756", "No": "#4C78A8"})
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, width="stretch")

# ---------------- EXPLORER ----------------
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Chart", ["Scatter", "Box", "Violin", "Histogram", "Density heatmap", "Bar (mean)"])
    x = c2.selectbox("X", num_cols, index=num_cols.index("Age"))
    y = c3.selectbox("Y", num_cols, index=num_cols.index("MMSE"))
    split = c4.checkbox("Colour by diagnosis", True)
    col = "Diagnosis_Label" if split else None
    cm = COLORS if split else None
    if kind == "Scatter":
        fig = px.scatter(df, x=x, y=y, color=col, color_discrete_map=cm, opacity=.6)
    elif kind == "Box":
        fig = px.box(df, x="Diagnosis_Label", y=y, color=col, color_discrete_map=cm, points="outliers")
    elif kind == "Violin":
        fig = px.violin(df, x="Diagnosis_Label", y=y, color=col, color_discrete_map=cm, box=True)
    elif kind == "Histogram":
        fig = px.histogram(df, x=x, color=col, color_discrete_map=cm, barmode="overlay", opacity=.7, nbins=30)
    elif kind == "Density heatmap":
        fig = px.density_heatmap(df, x=x, y=y, nbinsx=25, nbinsy=25, color_continuous_scale="Viridis")
    else:
        key = df[x] if full[x].nunique() <= 15 else pd.cut(df[x], 10).astype(str)
        fig = px.bar(df.groupby(key)[y].mean().reset_index(), x=x, y=y)
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")
    with st.expander("View filtered data"):
        st.dataframe(df.drop(columns="Diagnosis_Label"), width="stretch")
        st.download_button("Download filtered CSV", df.to_csv(index=False), "filtered_alzheimers.csv")

# ---------------- DISTRIBUTIONS ----------------
with tab3:
    v = st.selectbox("Variable", cont_cols, index=cont_cols.index("BMI"))
    s = df[v]
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = s[(s < lo) | (s > hi)]
    sk = s.skew()
    c = st.columns(6)
    c[0].metric("Mean", f"{s.mean():.2f}")
    c[1].metric("Median", f"{s.median():.2f}")
    c[2].metric("Std dev", f"{s.std():.2f}")
    c[3].metric("Skewness", f"{sk:.3f}")
    c[4].metric("Kurtosis", f"{s.kurt():.3f}")
    c[5].metric("Outliers (IQR)", f"{len(out)}")
    shape = "approximately symmetric" if abs(sk) < .5 else ("right-skewed (positive)" if sk > 0 else "left-skewed (negative)")
    st.info(f"**{v}** is {shape}. IQR fences: [{lo:.2f}, {hi:.2f}]. "
            f"Mean {'>' if s.mean() > s.median() else '<' if s.mean() < s.median() else '='} median.")
    l, r = st.columns(2)
    with l:
        fig = px.histogram(df, x=v, marginal="box", nbins=40, color="Diagnosis_Label",
                           color_discrete_map=COLORS, barmode="overlay", opacity=.7)
        fig.add_vline(x=s.mean(), line_dash="dash", annotation_text="mean")
        fig.add_vline(x=s.median(), line_dash="dot", annotation_text="median", annotation_position="bottom")
        st.plotly_chart(fig, width="stretch")
    with r:
        sk_all = full[cont_cols].skew().sort_values().reset_index()
        sk_all.columns = ["Feature", "Skewness"]
        fig = px.bar(sk_all, x="Skewness", y="Feature", orientation="h", title="Skewness of all continuous features",
                     color="Skewness", color_continuous_scale="RdBu_r", range_color=[-1, 1])
        st.plotly_chart(fig, width="stretch")

# ---------------- CORRELATIONS ----------------
with tab4:
    corr = df[num_cols + [TARGET]].corr()
    l, r = st.columns([3, 2])
    with l:
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title="Correlation matrix (Pearson)")
        fig.update_layout(height=700)
        st.plotly_chart(fig, width="stretch")
    with r:
        t = corr[TARGET].drop(TARGET).sort_values()
        t = pd.concat([t.head(8), t.tail(8)]).reset_index()
        t.columns = ["Feature", "Correlation"]
        fig = px.bar(t, x="Correlation", y="Feature", orientation="h", title="Strongest correlations with Diagnosis",
                     color="Correlation", color_continuous_scale="RdBu_r", range_color=[-.5, .5])
        fig.update_layout(height=700)
        st.plotly_chart(fig, width="stretch")

# ---------------- MODEL ----------------
with tab5:
    m1, m2, m3 = st.columns(3)
    mname = m1.selectbox("Model", ["Random Forest", "Logistic Regression"])
    tsize = m2.slider("Test size", .1, .5, .2, .05)
    seed = m3.number_input("Random seed", 0, 999, 42)
    model, X, Xte, yte, proba = train(full, mname, tsize, int(seed))
    pred = (proba >= .5).astype(int)

    c = st.columns(5)
    c[0].metric("Accuracy", f"{accuracy_score(yte, pred):.3f}")
    c[1].metric("Precision", f"{precision_score(yte, pred):.3f}")
    c[2].metric("Recall", f"{recall_score(yte, pred):.3f}")
    c[3].metric("F1", f"{f1_score(yte, pred):.3f}")
    c[4].metric("ROC-AUC", f"{roc_auc_score(yte, proba):.3f}")

    l, m, r = st.columns(3)
    with l:
        cmx = confusion_matrix(yte, pred)
        fig = px.imshow(cmx, text_auto=True, x=["Pred No", "Pred Yes"], y=["Actual No", "Actual Yes"],
                        color_continuous_scale="Blues", title="Confusion matrix")
        st.plotly_chart(fig, width="stretch")
    with m:
        fpr, tpr, _ = roc_curve(yte, proba)
        fig = go.Figure([go.Scatter(x=fpr, y=tpr, name="Model", line=dict(color="#E45756")),
                         go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash", color="grey"))])
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")
    with r:
        if mname == "Random Forest":
            imp = pd.Series(model.feature_importances_, X.columns)
        else:
            imp = pd.Series(np.abs(model[-1].coef_[0]), X.columns)
        imp = imp.sort_values().tail(10).reset_index()
        imp.columns = ["Feature", "Importance"]
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top 10 predictive features")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Live patient risk predictor")
    st.caption("Move the sliders - the risk updates instantly. Other inputs use the dataset median.")
    top = list(imp.Feature[::-1][:6])
    cols = st.columns(3)
    vals = X.median().to_dict()
    for i, f in enumerate(top):
        lo_, hi_ = float(X[f].min()), float(X[f].max())
        with cols[i % 3]:
            if X[f].nunique() <= 2:
                vals[f] = float(st.selectbox(f, [0, 1], key=f"p_{f}"))
            else:
                vals[f] = st.slider(f, lo_, hi_, float(X[f].median()), key=f"p_{f}")
    risk = model.predict_proba(pd.DataFrame([vals])[X.columns])[0, 1]
    g = go.Figure(go.Indicator(mode="gauge+number", value=risk * 100, number={"suffix": "%"},
                               title={"text": "Predicted Alzheimer's probability"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E45756"},
                                      "steps": [{"range": [0, 33], "color": "#d6eadf"},
                                                {"range": [33, 66], "color": "#fdf0c9"},
                                                {"range": [66, 100], "color": "#f6d0d0"}]}))
    g.update_layout(height=300)
    st.plotly_chart(g, width="stretch")
