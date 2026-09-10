"""Central Streamlit lab application for TGS-2020503264."""
from __future__ import annotations

import io
from html import escape

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from ml_core import (
    cross_validate_classifier,
    detect_anomalies,
    feature_ranking,
    impute_dataframe,
    mine_association_rules,
    missingness_profile,
    numeric_columns,
    run_classification,
    run_hierarchical,
    run_kmeans,
    run_pca,
    run_regression,
    silhouette_scan,
)
from workflow_catalog import WORKFLOW_GUIDES, WorkflowGuide

st.set_page_config(page_title="Data Mining & ML Lab", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
  :root {color-scheme: light;}
  .stApp, [data-testid="stAppViewContainer"] {background:#F8FAFC;color:#172B4D;}
  [data-testid="stHeader"] {background:rgba(248,250,252,.94);}
  [data-testid="stSidebar"] {background:#F1F5F9;border-right:1px solid #D7E2F2;}
  [data-testid="stSidebar"] > div:first-child {padding-top:1.35rem;}
  .block-container {max-width:1280px;padding-top:1.8rem;padding-bottom:3rem;}
  h1,h2,h3 {color:#102A43;letter-spacing:-.02em;}
  p, li, label, [data-testid="stCaptionContainer"] {font-size:1rem;line-height:1.55;}
  .course-hero {background:linear-gradient(135deg,#102A43 0%,#1E40AF 70%,#2563EB 100%);color:white;border-radius:20px;padding:28px 32px;margin-bottom:22px;box-shadow:0 14px 32px rgba(30,64,175,.16);}
  .course-hero .tag {display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.28);border-radius:999px;padding:5px 11px;font-size:.78rem;font-weight:700;letter-spacing:.08em;}
  .course-hero h1 {color:white;font-size:clamp(1.75rem,3vw,2.65rem);margin:14px 0 8px;line-height:1.12;}
  .course-hero p {color:#EAF2FF;margin:0;max-width:760px;}
  .method-card {background:white;border:1px solid #DBEAFE;border-radius:18px;padding:24px 26px;margin-bottom:16px;box-shadow:0 5px 18px rgba(15,23,42,.05);}
  .method-eyebrow,.section-kicker,.sidebar-kicker {color:#1E40AF;font-size:.76rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;}
  .method-card h2 {font-size:1.8rem;margin:.35rem 0 .45rem;}
  .method-summary {font-size:1.08rem;color:#334E68;margin:0 0 18px;max-width:980px;}
  .guide-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;}
  .guide-item {background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:14px 15px;}
  .guide-item strong {display:block;color:#102A43;margin-bottom:5px;}
  .guide-item span {color:#486581;font-size:.92rem;line-height:1.48;}
  .lab-strip {background:#FFF9ED;border:1px solid #F3D9A7;border-radius:14px;padding:15px 17px;margin:14px 0 18px;}
  .lab-strip strong {color:#7C4700;}
  .lab-pill {display:inline-block;background:white;color:#694100;border:1px solid #E8C47E;border-radius:999px;padding:5px 10px;margin:7px 6px 0 0;font-size:.86rem;font-weight:650;}
  .how-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:8px;}
  .how-step {border-left:3px solid #3B82F6;padding:7px 12px;color:#334E68;}
  .how-step b {color:#1E40AF;margin-right:5px;}
  .caution {background:#FFF7ED;border-left:4px solid #D97706;border-radius:8px;padding:12px 15px;color:#713F12;margin-top:14px;}
  .dataset-banner {background:#ECFDF5;border:1px solid #A7F3D0;border-radius:14px;padding:15px 18px;color:#065F46;margin:6px 0 16px;}
  .start-card {background:white;border:1px dashed #93B4E8;border-radius:16px;padding:22px 24px;margin-top:12px;}
  .start-card h3 {margin-top:0;}
  .start-card code {background:#EEF4FF;color:#1E3A8A;padding:2px 5px;border-radius:5px;overflow-wrap:anywhere;}
  [data-testid="stMetric"] {background:white;border:1px solid #D7E2F2;border-radius:14px;padding:14px;box-shadow:0 3px 12px rgba(15,23,42,.04);}
  [data-testid="stVerticalBlockBorderWrapper"] {background:white;border-color:#D7E2F2!important;border-radius:16px;}
  div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {background:white!important;color:#172B4D!important;min-height:46px;}
  .stButton button,.stDownloadButton button,.stLinkButton a {min-height:44px;border-radius:10px;font-weight:700;transition:transform .16s ease,box-shadow .16s ease;}
  .stButton button:hover,.stDownloadButton button:hover,.stLinkButton a:hover {transform:translateY(-1px);box-shadow:0 5px 12px rgba(30,64,175,.14);}
  .powered-by {color:#627D98;font-size:.86rem;text-align:center;margin:1.3rem 0 0;}
  .powered-by a {color:#1E40AF;font-weight:700;text-decoration:none;}
  .powered-by a:hover {text-decoration:underline;}
  button:focus-visible,a:focus-visible,input:focus-visible {outline:3px solid #93C5FD!important;outline-offset:2px;}
  @media (prefers-reduced-motion:reduce) {* {scroll-behavior:auto!important;transition:none!important;}}
  @media (max-width:800px) {
    .block-container {padding:1rem .9rem 2rem;}
    .course-hero {padding:22px 20px;border-radius:15px;}
    .method-card {padding:20px 18px;}
    .guide-grid,.how-grid {grid-template-columns:1fr;}
    .lab-pill {display:block;margin-right:0;}
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<section class="course-hero">
  <span class="tag">TGS-2020503264 · CENTRAL LAB APP</span>
  <h1>Data Mining &amp; Machine Learning Lab</h1>
  <p>Understand the method, connect it to your lab, and turn mock data into evidence you can explain.</p>
</section>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def choose_data():
    st.sidebar.markdown('<div class="sidebar-kicker">STEP 2 · ADD LAB DATA</div>', unsafe_allow_html=True)
    upload = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Use the mock dataset supplied in the lab folder. Your upload stays in this local Streamlit session.",
    )
    if upload:
        return read_upload(upload.getvalue(), upload.name), upload.name
    st.sidebar.caption("Use the mock dataset supplied in the current lab folder.")
    return None, None


def require_data(df):
    if df is None:
        st.info("Start by uploading the mock dataset from your lab folder.")
        st.stop()


def preferred_index(columns, preferred: tuple[str, ...] = ()) -> int:
    """Choose a beginner-friendly default without assuming every lab schema."""
    names = list(columns)
    for candidate in preferred:
        if candidate in names:
            return names.index(candidate)
    return max(0, len(names) - 1)


def render_method_guide(guide: WorkflowGuide) -> None:
    lab_pills = "".join(
        f'<span class="lab-pill">Lab {escape(lab.number)} · {escape(lab.title)}</span>'
        for lab in guide.labs
    )
    steps = "".join(
        f'<div class="how-step"><b>{index}</b>{escape(step)}</div>'
        for index, step in enumerate(guide.steps, start=1)
    )
    st.markdown(
        f"""
        <section class="method-card">
          <div class="method-eyebrow">{escape(guide.eyebrow)}</div>
          <h2>{escape(guide.title)}</h2>
          <p class="method-summary">{escape(guide.summary)}</p>
          <div class="guide-grid">
            <div class="guide-item"><strong>Use it when</strong><span>{escape(guide.use_when)}</span></div>
            <div class="guide-item"><strong>How it works</strong><span>{escape(guide.mechanism)}</span></div>
            <div class="guide-item"><strong>Evidence to keep</strong><span>{escape(guide.evidence)}</span></div>
          </div>
          <div class="lab-strip"><strong>Used in these course labs</strong><br>{lab_pills}</div>
          <div class="section-kicker">METHOD IN THREE MOVES</div>
          <div class="how-grid">{steps}</div>
          <div class="caution"><strong>Interpretation check:</strong> {escape(guide.caution)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


workflows = list(WORKFLOW_GUIDES)
st.sidebar.markdown('<div class="sidebar-kicker">STEP 1 · CHOOSE A METHOD</div>', unsafe_allow_html=True)
workflow = st.sidebar.selectbox(
    "Analysis method",
    workflows,
    help="Choose the workflow named in your lab guide.",
)
guide = WORKFLOW_GUIDES[workflow]
df, filename = choose_data()
st.sidebar.markdown('<div class="sidebar-kicker">LAB CONNECTION</div>', unsafe_allow_html=True)
for lab in guide.labs:
    st.sidebar.markdown(f"**Lab {lab.number}** · {lab.title}")
    st.sidebar.caption(f"{lab.folder}/{lab.data}")
st.sidebar.link_button(
    "Course registration",
    "https://www.tertiarycourses.com.sg/wsq-data-mining-and-machine-learning-fundamentals-for-beginners.html",
    use_container_width=True,
)
st.sidebar.link_button(
    "Central Lab Portal",
    "https://alfredang.github.io/data-mining-ml-labs/",
    use_container_width=True,
)

render_method_guide(guide)

if df is not None:
    missing_cells = int(df.isna().sum().sum())
    st.markdown(
        f'<div class="dataset-banner"><strong>Dataset ready:</strong> {escape(filename)} · '
        f'{len(df):,} rows · {len(df.columns)} columns · {missing_cells:,} missing cells</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Preview the uploaded dataset"):
        st.dataframe(df.head(20), use_container_width=True)
else:
    lab_paths = "<br>".join(
        f'<code>labs/{escape(lab.folder)}/{escape(lab.data)}</code>' for lab in guide.labs
    )
    st.markdown(
        f"""
        <section class="start-card">
          <div class="section-kicker">READY TO BEGIN</div>
          <h3>Upload the dataset for your current lab</h3>
          <p>Find the supplied mock data at:</p>
          <p>{lab_paths}</p>
          <p>Then use <strong>Step 2</strong> in the sidebar. The analysis controls will appear here after the file is validated.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown('<div class="section-kicker">ANALYSIS WORKSPACE</div>', unsafe_allow_html=True)
st.subheader("Configure and run")
st.caption("Use the settings specified in your lab guide, run the method, then retain the requested evidence and interpretation.")

if workflow == "Data audit & preparation":
    require_data(df)
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}"); b.metric("Columns", len(df.columns)); c.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df.head(30), use_container_width=True)
    st.subheader("Missingness evidence")
    profile = missingness_profile(df); st.dataframe(profile, use_container_width=True)
    strategy = st.radio("Numeric imputation", ["median", "mean"], horizontal=True)
    cleaned = impute_dataframe(df, strategy)
    st.download_button("Download cleaned CSV", cleaned.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")
    with st.expander("Optional filtering"):
        col = st.selectbox("Column", df.columns)
        if pd.api.types.is_numeric_dtype(df[col]):
            lo, hi = float(df[col].min()), float(df[col].max())
            chosen = st.slider("Range", lo, hi, (lo, hi))
            st.dataframe(df[df[col].between(*chosen)], use_container_width=True)
        else:
            values = st.multiselect("Values", sorted(df[col].dropna().astype(str).unique()))
            st.dataframe(df[df[col].astype(str).isin(values)] if values else df, use_container_width=True)

elif workflow == "Join & aggregate":
    require_data(df)
    second = st.file_uploader("Upload the second table", type=["csv", "xlsx", "xls"], key="second", help="For Lab 03, upload customers.csv after orders.csv is already loaded in the sidebar.")
    if second:
        right = read_upload(second.getvalue(), second.name)
        common = sorted(set(df.columns) & set(right.columns))
        if not common: st.error("No common key column was found."); st.stop()
        key = st.selectbox("Join key", common); how = st.selectbox("Join type", ["inner", "left", "right", "outer"])
        joined = df.merge(right, on=key, how=how, indicator=True)
        st.metric("Joined rows", len(joined)); st.dataframe(joined.head(50), use_container_width=True)
        cat = st.selectbox("Group by", joined.columns)
        nums = numeric_columns(joined)
        if nums:
            val = st.selectbox("Measure", nums); agg = joined.groupby(cat, dropna=False)[val].agg(["count", "mean", "sum"]).reset_index()
            st.dataframe(agg, use_container_width=True)
            st.plotly_chart(px.bar(agg, x=cat, y="sum", title=f"Sum of {val} by {cat}"), use_container_width=True)

elif workflow == "Regression":
    require_data(df)
    nums = numeric_columns(df)
    target = st.selectbox("1. Numeric target to predict", nums, index=preferred_index(nums, ("price_sgd", "price", "monthly_spend")), help="Choose the continuous outcome, such as price_sgd.")
    features = st.multiselect("2. Input features", [c for c in df.columns if c != target], default=[c for c in nums if c != target][:4], help="Choose columns available at prediction time. Avoid target leakage.")
    model = st.selectbox("3. Regression model", ["Linear", "Ridge", "Lasso"], help="Linear is the baseline. Ridge and Lasso add regularisation to reduce overfitting.")
    model_notes = {
        "Linear": "Linear regression fits one weighted line or plane through the feature space. Start here for a transparent baseline.",
        "Ridge": "Ridge shrinks large coefficients toward zero, which can stabilise a model when features overlap or overfit.",
        "Lasso": "Lasso can shrink some coefficients exactly to zero, creating a simpler model that uses fewer features.",
    }
    st.info(model_notes[model])
    alpha = st.slider("4. Regularisation strength (alpha)", 0.01, 10.0, 1.0, disabled=model == "Linear", help="Higher alpha means stronger coefficient shrinkage. It applies only to Ridge and Lasso.")
    if st.button("Train regression model", type="primary", disabled=not features, use_container_width=True):
        result = run_regression(df, target, features, model, alpha=alpha)
        cols = st.columns(3)
        for box, (name, value) in zip(cols, result.metrics.items()): box.metric(name, f"{value:.4f}")
        st.plotly_chart(px.scatter(result.predictions, x="actual", y="predicted", trendline="ols", title="Actual vs predicted"), use_container_width=True)
        st.dataframe(result.predictions, use_container_width=True)

elif workflow == "Classification":
    require_data(df)
    target = st.selectbox("1. Target class", df.columns, index=preferred_index(df.columns, ("diagnosis", "churned", "target")))
    features = st.multiselect("2. Input features", [c for c in df.columns if c != target], default=[c for c in numeric_columns(df) if c != target][:5])
    model = st.selectbox("3. Classifier", ["Logistic Regression", "K-Nearest Neighbours", "Decision Tree", "Random Forest", "Support Vector Machine"])
    st.caption("Compare models on the same target, features and evidence. The best choice depends on error costs, stability and interpretability—not accuracy alone.")
    if st.button("Train classifier", type="primary", disabled=not features, use_container_width=True):
        result = run_classification(df, target, features, model)
        cols = st.columns(len(result.metrics))
        for box, (name, value) in zip(cols, result.metrics.items()): box.metric(name, f"{value:.3f}")
        cm = result.predictions.attrs["confusion_matrix"]; labels = result.predictions.attrs["labels"]
        fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix"); st.pyplot(fig)
        st.dataframe(result.predictions, use_container_width=True)

elif workflow == "Cross-validation":
    require_data(df)
    target = st.selectbox("1. Target class", df.columns, index=preferred_index(df.columns, ("diagnosis", "churned", "target")))
    features = st.multiselect("2. Input features", [c for c in df.columns if c != target], default=[c for c in numeric_columns(df) if c != target][:5])
    model = st.selectbox("3. Classifier", ["Logistic Regression", "K-Nearest Neighbours", "Decision Tree", "Random Forest", "Support Vector Machine"])
    folds = st.slider("4. Number of folds", 3, 10, 5)
    if st.button("Run cross-validation", type="primary", disabled=not features, use_container_width=True):
        scores = cross_validate_classifier(df, target, features, model, folds)
        st.metric("Mean accuracy", f"{scores.accuracy.mean():.3f} ± {scores.accuracy.std():.3f}")
        st.plotly_chart(px.bar(scores, x="fold", y="accuracy", range_y=[0, 1], title="Accuracy by fold"), use_container_width=True)

elif workflow == "Clustering":
    require_data(df)
    nums = numeric_columns(df); features = st.multiselect("Numeric features", nums, default=nums[:3])
    if len(features) >= 2:
        scan = silhouette_scan(df, features); st.plotly_chart(px.line(scan, x="k", y="silhouette", markers=True, title="Choose k at the strongest valid silhouette score"), use_container_width=True)
        k = st.slider("Number of clusters", 2, min(8, max(2, len(df)-1)), 3)
        clustered, score, _ = run_kmeans(df, features, k); st.metric("Silhouette score", f"{score:.3f}")
        st.plotly_chart(px.scatter(clustered, x=features[0], y=features[1], color="cluster", hover_data=features, title="Cluster evidence"), use_container_width=True)
        st.download_button("Download clustered data", clustered.to_csv(index=False).encode(), "clustered_data.csv", "text/csv")

elif workflow == "Hierarchical clustering":
    require_data(df)
    nums = numeric_columns(df); features = st.multiselect("Numeric features", nums, default=nums[:4])
    clusters = st.slider("Clusters to cut", 2, min(8, max(2, len(df)-1)), 2)
    if len(features) >= 2:
        clustered, matrix, _ = run_hierarchical(df, features, clusters)
        fig, ax = plt.subplots(figsize=(10, 4)); dendrogram(matrix, truncate_mode="lastp", p=min(30, len(df)), ax=ax)
        ax.set(title="Ward-linkage dendrogram", xlabel="Merged observations", ylabel="Distance"); st.pyplot(fig)
        st.plotly_chart(px.scatter(clustered, x=features[0], y=features[1], color="cluster", title="Selected cluster cut"), use_container_width=True)

elif workflow == "PCA & feature ranking":
    require_data(df)
    mode = st.radio("Analysis", ["Principal Component Analysis", "Feature ranking"], horizontal=True)
    if mode == "Principal Component Analysis":
        nums = numeric_columns(df); features = st.multiselect("Numeric features", nums, default=nums[:6])
        variance_target = st.slider("Retained variance", 0.50, 0.99, 0.70)
        if len(features) >= 2:
            projected, variance, loadings, _ = run_pca(df, features, variance_target)
            st.metric("Components retained", len(projected.columns)); st.plotly_chart(px.line(variance, x="component", y="cumulative", markers=True, title="Cumulative explained variance"), use_container_width=True)
            if len(projected.columns) >= 2: st.plotly_chart(px.scatter(projected, x="PC1", y="PC2", title="Projection onto first two components"), use_container_width=True)
            st.subheader("Component loadings"); st.dataframe(loadings.style.background_gradient(cmap="GnBu"), use_container_width=True)
    else:
        target = st.selectbox("Target class", df.columns, index=preferred_index(df.columns, ("diagnosis", "churned", "target"))); features = st.multiselect("Candidate features", [c for c in df.columns if c != target], default=[c for c in numeric_columns(df) if c != target][:8])
        if st.button("Rank features", type="primary", disabled=not features, use_container_width=True):
            ranking = feature_ranking(df, target, features); st.plotly_chart(px.bar(ranking, x="importance", y="feature", orientation="h", title="Permutation importance"), use_container_width=True); st.dataframe(ranking)

elif workflow == "Association rules":
    require_data(df)
    tx = st.selectbox("Transaction ID", df.columns); item = st.selectbox("Item", [c for c in df.columns if c != tx])
    support = st.slider("Minimum support", 0.01, 0.80, 0.10); confidence = st.slider("Minimum confidence", 0.10, 1.00, 0.50)
    if st.button("Mine rules", type="primary", use_container_width=True):
        itemsets, rules = mine_association_rules(df, tx, item, support, confidence)
        st.subheader("Frequent itemsets"); st.dataframe(itemsets, use_container_width=True)
        st.subheader("Rules"); st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]] if not rules.empty else rules, use_container_width=True)

elif workflow == "Anomaly detection":
    require_data(df)
    anomaly_nums = numeric_columns(df)
    col = st.selectbox("Numeric measure", anomaly_nums, index=preferred_index(anomaly_nums, ("temperature", "value", "reading"))); method = st.radio("Method", ["Z-score", "IQR"], horizontal=True)
    default = 3.0 if method == "Z-score" else 1.5; threshold = st.slider("Threshold", 0.5, 5.0, default)
    result = detect_anomalies(df[col], method, threshold); view = df.copy(); view["score"] = result.score; view["is_anomaly"] = result.is_anomaly
    st.metric("Flagged rows", int(view.is_anomaly.sum()))
    plot = view.reset_index(names="row"); st.plotly_chart(px.scatter(plot, x="row", y=col, color="is_anomaly", color_discrete_map={False:"#0f766e",True:"#f59e0b"}, title="Anomaly evidence"), use_container_width=True)
    st.dataframe(view[view.is_anomaly], use_container_width=True)

st.divider()
st.caption("Outputs are learning evidence, not automatic business decisions. Verify data quality, assumptions, and error costs before acting.")
st.markdown(
    '<p class="powered-by">Powered by <a href="https://www.tertiaryinfotech.com/" target="_blank" rel="noopener">Tertiary Infotech Academy Pte Ltd</a></p>',
    unsafe_allow_html=True,
)
