"""Central Streamlit lab application for TGS-2020503264."""
from __future__ import annotations

import io

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

st.set_page_config(page_title="Data Mining & ML Lab", page_icon="📊", layout="wide")
st.markdown("""
<style>
  .stApp {background: #f7faf9;} h1,h2,h3 {color:#102a43}
  [data-testid="stMetric"] {background:white;border:1px solid #d9e2ec;border-radius:14px;padding:14px}
  .lab-note {background:#e6fffa;border-left:5px solid #0f766e;padding:12px 16px;border-radius:8px}
</style>
""", unsafe_allow_html=True)

st.title("Data Mining & Machine Learning Lab")
st.caption("Central learner application · TGS-2020503264 · Python + Streamlit")


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def choose_data():
    upload = st.sidebar.file_uploader("Upload a lab CSV or Excel file", type=["csv", "xlsx", "xls"])
    if upload:
        return read_upload(upload.getvalue(), upload.name), upload.name
    st.sidebar.info("Upload the mock dataset supplied in the current lab folder.")
    return None, None


def require_data(df):
    if df is None:
        st.info("Start by uploading the mock dataset from your lab folder.")
        st.stop()


workflows = [
    "Data audit & preparation", "Join & aggregate", "Regression", "Classification",
    "Cross-validation", "Clustering", "Hierarchical clustering", "PCA & feature ranking",
    "Association rules", "Anomaly detection",
]
workflow = st.sidebar.selectbox("Workflow", workflows)
df, filename = choose_data()
st.sidebar.markdown("[Course registration](https://www.tertiarycourses.com.sg/wsq-data-mining-and-machine-learning-fundamentals-for-beginners.html)")

if df is not None:
    st.success(f"Loaded **{filename}** · {len(df):,} rows × {len(df.columns)} columns")

if workflow == "Data audit & preparation":
    require_data(df)
    st.header("Data audit & preparation")
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
    st.header("Join & aggregate")
    second = st.file_uploader("Upload the second table", type=["csv", "xlsx", "xls"], key="second")
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
    require_data(df); st.header("Regression")
    nums = numeric_columns(df)
    target = st.selectbox("Numeric target", nums)
    features = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in nums if c != target][:4])
    model = st.selectbox("Model", ["Linear", "Ridge", "Lasso"]); alpha = st.slider("Regularisation α", 0.01, 10.0, 1.0)
    if st.button("Train regression model", type="primary") and features:
        result = run_regression(df, target, features, model, alpha=alpha)
        cols = st.columns(3)
        for box, (name, value) in zip(cols, result.metrics.items()): box.metric(name, f"{value:.4f}")
        st.plotly_chart(px.scatter(result.predictions, x="actual", y="predicted", trendline="ols", title="Actual vs predicted"), use_container_width=True)
        st.dataframe(result.predictions, use_container_width=True)

elif workflow == "Classification":
    require_data(df); st.header("Classification")
    target = st.selectbox("Target class", df.columns)
    features = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in numeric_columns(df) if c != target][:5])
    model = st.selectbox("Classifier", ["Logistic Regression", "K-Nearest Neighbours", "Decision Tree", "Random Forest", "Support Vector Machine"])
    if st.button("Train classifier", type="primary") and features:
        result = run_classification(df, target, features, model)
        cols = st.columns(len(result.metrics))
        for box, (name, value) in zip(cols, result.metrics.items()): box.metric(name, f"{value:.3f}")
        cm = result.predictions.attrs["confusion_matrix"]; labels = result.predictions.attrs["labels"]
        fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix"); st.pyplot(fig)
        st.dataframe(result.predictions, use_container_width=True)

elif workflow == "Cross-validation":
    require_data(df); st.header("K-fold cross-validation")
    target = st.selectbox("Target class", df.columns)
    features = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in numeric_columns(df) if c != target][:5])
    model = st.selectbox("Classifier", ["Logistic Regression", "K-Nearest Neighbours", "Decision Tree", "Random Forest", "Support Vector Machine"])
    folds = st.slider("Folds", 3, 10, 5)
    if st.button("Run cross-validation", type="primary") and features:
        scores = cross_validate_classifier(df, target, features, model, folds)
        st.metric("Mean accuracy", f"{scores.accuracy.mean():.3f} ± {scores.accuracy.std():.3f}")
        st.plotly_chart(px.bar(scores, x="fold", y="accuracy", range_y=[0, 1], title="Accuracy by fold"), use_container_width=True)

elif workflow == "Clustering":
    require_data(df); st.header("K-means clustering & silhouette analysis")
    nums = numeric_columns(df); features = st.multiselect("Numeric features", nums, default=nums[:3])
    if len(features) >= 2:
        scan = silhouette_scan(df, features); st.plotly_chart(px.line(scan, x="k", y="silhouette", markers=True, title="Choose k at the strongest valid silhouette score"), use_container_width=True)
        k = st.slider("Number of clusters", 2, min(8, max(2, len(df)-1)), 3)
        clustered, score, _ = run_kmeans(df, features, k); st.metric("Silhouette score", f"{score:.3f}")
        st.plotly_chart(px.scatter(clustered, x=features[0], y=features[1], color="cluster", hover_data=features, title="Cluster evidence"), use_container_width=True)
        st.download_button("Download clustered data", clustered.to_csv(index=False).encode(), "clustered_data.csv", "text/csv")

elif workflow == "Hierarchical clustering":
    require_data(df); st.header("Hierarchical clustering")
    nums = numeric_columns(df); features = st.multiselect("Numeric features", nums, default=nums[:4])
    clusters = st.slider("Clusters to cut", 2, min(8, max(2, len(df)-1)), 2)
    if len(features) >= 2:
        clustered, matrix, _ = run_hierarchical(df, features, clusters)
        fig, ax = plt.subplots(figsize=(10, 4)); dendrogram(matrix, truncate_mode="lastp", p=min(30, len(df)), ax=ax)
        ax.set(title="Ward-linkage dendrogram", xlabel="Merged observations", ylabel="Distance"); st.pyplot(fig)
        st.plotly_chart(px.scatter(clustered, x=features[0], y=features[1], color="cluster", title="Selected cluster cut"), use_container_width=True)

elif workflow == "PCA & feature ranking":
    require_data(df); st.header("Dimension reduction & feature ranking")
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
        target = st.selectbox("Target class", df.columns); features = st.multiselect("Candidate features", [c for c in df.columns if c != target], default=[c for c in numeric_columns(df) if c != target][:8])
        if st.button("Rank features", type="primary") and features:
            ranking = feature_ranking(df, target, features); st.plotly_chart(px.bar(ranking, x="importance", y="feature", orientation="h", title="Permutation importance"), use_container_width=True); st.dataframe(ranking)

elif workflow == "Association rules":
    require_data(df); st.header("Frequent itemsets & association rules")
    tx = st.selectbox("Transaction ID", df.columns); item = st.selectbox("Item", [c for c in df.columns if c != tx])
    support = st.slider("Minimum support", 0.01, 0.80, 0.10); confidence = st.slider("Minimum confidence", 0.10, 1.00, 0.50)
    if st.button("Mine rules", type="primary"):
        itemsets, rules = mine_association_rules(df, tx, item, support, confidence)
        st.subheader("Frequent itemsets"); st.dataframe(itemsets, use_container_width=True)
        st.subheader("Rules"); st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]] if not rules.empty else rules, use_container_width=True)

elif workflow == "Anomaly detection":
    require_data(df); st.header("Anomaly detection")
    col = st.selectbox("Numeric measure", numeric_columns(df)); method = st.radio("Method", ["Z-score", "IQR"], horizontal=True)
    default = 3.0 if method == "Z-score" else 1.5; threshold = st.slider("Threshold", 0.5, 5.0, default)
    result = detect_anomalies(df[col], method, threshold); view = df.copy(); view["score"] = result.score; view["is_anomaly"] = result.is_anomaly
    st.metric("Flagged rows", int(view.is_anomaly.sum()))
    plot = view.reset_index(names="row"); st.plotly_chart(px.scatter(plot, x="row", y=col, color="is_anomaly", color_discrete_map={False:"#0f766e",True:"#f59e0b"}, title="Anomaly evidence"), use_container_width=True)
    st.dataframe(view[view.is_anomaly], use_container_width=True)

st.divider()
st.caption("Outputs are learning evidence, not automatic business decisions. Verify data quality, assumptions, and error costs before acting.")
