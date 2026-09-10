"""Testable analytics functions for the Data Mining & ML Lab central app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame
    model: object


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    return pd.DataFrame({"column": counts.index, "missing": counts.values,
                         "missing_pct": (counts.values / max(len(df), 1) * 100).round(2)})


def impute_dataframe(df: pd.DataFrame, numeric_strategy: str = "median") -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            value = out[col].mean() if numeric_strategy == "mean" else out[col].median()
        else:
            mode = out[col].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else "Unknown"
        out[col] = out[col].fillna(value)
    return out


def _preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def run_regression(df: pd.DataFrame, target: str, features: list[str], model_name: str = "Linear",
                   test_size: float = 0.2, random_state: int = 42, alpha: float = 1.0) -> ModelResult:
    work = df.dropna(subset=[target]).copy()
    X, y = work[features], pd.to_numeric(work[target], errors="coerce")
    keep = y.notna(); X, y = X.loc[keep], y.loc[keep]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    estimator = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=alpha), "Lasso": Lasso(alpha=alpha)}[model_name]
    pipe = Pipeline([("prep", _preprocessor(work, features)), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {"MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5,
               "R2": r2_score(y_test, pred)}
    return ModelResult(metrics, pd.DataFrame({"actual": y_test, "predicted": pred}, index=y_test.index), pipe)


def classification_estimator(name: str, random_state: int = 42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Support Vector Machine": SVC(probability=True, random_state=random_state),
    }[name]


def run_classification(df: pd.DataFrame, target: str, features: list[str], model_name: str,
                       test_size: float = 0.25, random_state: int = 42) -> ModelResult:
    work = df.dropna(subset=[target]).copy(); X, y = work[features], work[target].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    pipe = Pipeline([("prep", _preprocessor(work, features)),
                     ("model", classification_estimator(model_name, random_state))])
    pipe.fit(X_train, y_train); pred = pipe.predict(X_test)
    labels = sorted(y.unique().tolist())
    average = "binary" if len(labels) == 2 else "weighted"
    pos = labels[-1] if len(labels) == 2 else None
    kwargs = {"average": average, "zero_division": 0}
    if pos is not None: kwargs["pos_label"] = pos
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, **kwargs),
        "Recall": recall_score(y_test, pred, **kwargs),
        "F1": f1_score(y_test, pred, **kwargs),
    }
    if len(labels) == 2 and hasattr(pipe, "predict_proba"):
        metrics["ROC AUC"] = roc_auc_score((y_test == pos).astype(int), pipe.predict_proba(X_test)[:, 1])
    result = pd.DataFrame({"actual": y_test, "predicted": pred}, index=y_test.index)
    result.attrs["confusion_matrix"] = confusion_matrix(y_test, pred, labels=labels)
    result.attrs["labels"] = labels
    return ModelResult(metrics, result, pipe)


def cross_validate_classifier(df: pd.DataFrame, target: str, features: list[str], model_name: str,
                              folds: int = 5) -> pd.DataFrame:
    work = df.dropna(subset=[target]).copy(); X, y = work[features], work[target].astype(str)
    pipe = Pipeline([("prep", _preprocessor(work, features)), ("model", classification_estimator(model_name))])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return pd.DataFrame({"fold": np.arange(1, folds + 1), "accuracy": scores})


def run_kmeans(df: pd.DataFrame, features: list[str], k: int = 3) -> tuple[pd.DataFrame, float, KMeans]:
    X = SimpleImputer(strategy="median").fit_transform(df[features])
    Xs = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xs)
    out = df.copy(); out["cluster"] = model.labels_.astype(str)
    return out, float(silhouette_score(Xs, model.labels_)), model


def silhouette_scan(df: pd.DataFrame, features: list[str], candidates: Iterable[int] = range(2, 9)) -> pd.DataFrame:
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[features]))
    rows = []
    for k in candidates:
        if k >= len(X): continue
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(rows)


def run_hierarchical(df: pd.DataFrame, features: list[str], clusters: int = 2):
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[features]))
    model = AgglomerativeClustering(n_clusters=clusters).fit(X)
    out = df.copy(); out["cluster"] = model.labels_.astype(str)
    return out, linkage(X, method="ward"), model


def run_pca(df: pd.DataFrame, features: list[str], variance_target: float = 0.70):
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[features]))
    full = PCA().fit(X); cumulative = np.cumsum(full.explained_variance_ratio_)
    n = int(np.argmax(cumulative >= variance_target) + 1)
    fitted = PCA(n_components=n).fit(X); coords = fitted.transform(X)
    projected = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n)], index=df.index)
    variance = pd.DataFrame({"component": np.arange(1, len(full.explained_variance_ratio_) + 1),
                             "explained": full.explained_variance_ratio_, "cumulative": cumulative})
    loadings = pd.DataFrame(fitted.components_.T, index=features, columns=projected.columns)
    return projected, variance, loadings, fitted


def feature_ranking(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    work = df.dropna(subset=[target]).copy(); X = work[features]; y = work[target].astype(str)
    pipe = Pipeline([("prep", _preprocessor(work, features)),
                     ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
    pipe.fit(X, y)
    perm = permutation_importance(pipe, X, y, n_repeats=8, random_state=42, scoring="accuracy")
    return pd.DataFrame({"feature": features, "importance": perm.importances_mean}).sort_values("importance", ascending=False)


def mine_association_rules(df: pd.DataFrame, transaction_col: str, item_col: str,
                           min_support: float = 0.1, min_confidence: float = 0.5):
    basket = pd.crosstab(df[transaction_col], df[item_col]).astype(bool)
    itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return itemsets, pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        for c in ["antecedents", "consequents"]:
            rules[c] = rules[c].apply(lambda x: ", ".join(sorted(x)))
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
    return itemsets.sort_values("support", ascending=False), rules


def detect_anomalies(series: pd.Series, method: str = "Z-score", threshold: float = 3.0) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    if method == "Z-score":
        score = ((s - s.mean()) / s.std(ddof=1)).abs(); flag = score > threshold
    else:
        q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
        lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
        score = pd.concat([(lo - s), (s - hi)], axis=1).max(axis=1).clip(lower=0); flag = (s < lo) | (s > hi)
    return pd.DataFrame({"value": s, "score": score, "is_anomaly": flag.fillna(False)})
