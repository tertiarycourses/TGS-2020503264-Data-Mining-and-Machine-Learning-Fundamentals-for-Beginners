"""Learner-facing method explanations and lab mappings for the central app."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabReference:
    number: str
    title: str
    folder: str
    data: str


@dataclass(frozen=True)
class WorkflowGuide:
    eyebrow: str
    title: str
    summary: str
    use_when: str
    mechanism: str
    evidence: str
    steps: tuple[str, ...]
    caution: str
    labs: tuple[LabReference, ...]


WORKFLOW_GUIDES: dict[str, WorkflowGuide] = {
    "Data audit & preparation": WorkflowGuide(
        eyebrow="FOUNDATION",
        title="Data audit & preparation",
        summary="Inspect a dataset before modelling, then handle missing values consistently so later results are based on usable evidence.",
        use_when="You have received a new dataset or need a clean, reproducible input for another method.",
        mechanism="The app profiles rows, columns and missingness, then replaces missing numeric values with a median or mean and categorical values with the most frequent value.",
        evidence="A missingness table, a preview of the records and an exported cleaned CSV.",
        steps=("Inspect the dataset profile", "Choose an imputation rule", "Verify and export the cleaned data"),
        caution="Imputation makes data usable; it does not recover the unknown true value. Record the rule you used.",
        labs=(
            LabReference("01", "Data audit and business question", "lab-01-audit-business-data", "customer_quality.csv"),
            LabReference("02", "Clean, impute, and export", "lab-02-clean-impute-export", "customer_quality.csv"),
        ),
    ),
    "Join & aggregate": WorkflowGuide(
        eyebrow="DATA INTEGRATION",
        title="Join & aggregate",
        summary="Combine related tables through a shared key, then summarise detailed records into decision-ready totals and averages.",
        use_when="The evidence is split across tables, such as customers in one file and their orders in another.",
        mechanism="A join matches key values between two tables. Aggregation then groups the joined rows and calculates count, mean and sum for a numeric measure.",
        evidence="Join-status counts, a joined-table preview, a grouped summary and a comparison chart.",
        steps=("Choose the common key", "Select the join type", "Group and aggregate a measure"),
        caution="Duplicate or missing keys can multiply or discard rows. Compare row counts before and after the join.",
        labs=(LabReference("03", "Filter, join, and aggregate", "lab-03-filter-join-aggregate", "orders.csv + customers.csv"),),
    ),
    "Regression": WorkflowGuide(
        eyebrow="PREDICT A NUMBER",
        title="Regression",
        summary="Regression learns how input features relate to a continuous numeric outcome—for example, estimating a housing price rather than choosing a category.",
        use_when="The target is a number with meaningful distance between values, such as price, demand, time or monthly spend.",
        mechanism="The app splits data into training and test sets, imputes and scales features, fits a line or regularised model, then predicts unseen test rows.",
        evidence="MAE shows typical absolute error, RMSE penalises large errors and R² shows how much target variation the model explains.",
        steps=("Choose a numeric target", "Select useful features", "Compare predictions with actual values"),
        caution="A strong fit does not prove causation. Check error size, residual patterns, leakage and whether the test data represents future cases.",
        labs=(
            LabReference("04", "Scale features before modelling", "lab-04-scale-and-compare", "housing_scale.csv"),
            LabReference("05", "Build a regression baseline", "lab-05-regression-baseline", "housing_regression.csv"),
            LabReference("06", "Control overfitting with Ridge and Lasso", "lab-06-regularization-overfit", "housing_regression.csv"),
        ),
    ),
    "Classification": WorkflowGuide(
        eyebrow="PREDICT A CLASS",
        title="Classification",
        summary="Classification learns to assign a record to a discrete class, such as churn or stay, positive or negative, or one of several named categories.",
        use_when="The target is a label and different prediction errors may carry different business costs.",
        mechanism="The app learns class boundaries from labelled training rows and evaluates predictions on a held-out test set.",
        evidence="Accuracy, precision, recall, F1, optional ROC AUC and a confusion matrix showing each error type.",
        steps=("Choose the target class", "Select candidate features", "Compare metrics and error types"),
        caution="Accuracy alone can hide poor minority-class performance. Use the confusion matrix and the cost of false positives and false negatives.",
        labs=(
            LabReference("07", "Compare classification algorithms", "lab-07-classification-baselines", "breast_cancer.csv"),
            LabReference("09", "Interpret confusion metrics and error costs", "lab-09-confusion-roc-costs", "churn.csv"),
        ),
    ),
    "Cross-validation": WorkflowGuide(
        eyebrow="TEST STABILITY",
        title="K-fold cross-validation",
        summary="Cross-validation repeats evaluation on several train-and-test splits so you can see whether a classifier performs consistently.",
        use_when="A single test split may be misleading or you need stronger evidence before selecting a model.",
        mechanism="The data is divided into K folds. Each fold becomes the test set once while the remaining folds train the model; stratification preserves class proportions.",
        evidence="Accuracy for every fold plus the mean and standard deviation across all folds.",
        steps=("Choose target and features", "Set the number of folds", "Inspect average performance and spread"),
        caution="Cross-validation estimates generalisation only when preprocessing stays inside each fold and the data is representative.",
        labs=(LabReference("08", "Use stratified K-fold validation", "lab-08-kfold-validation", "breast_cancer.csv"),),
    ),
    "Clustering": WorkflowGuide(
        eyebrow="DISCOVER GROUPS",
        title="K-means clustering",
        summary="K-means discovers groups of similar records without a target label and assigns each record to its nearest cluster centre.",
        use_when="You want exploratory segments from numeric behaviour or characteristics and do not already have class labels.",
        mechanism="The app scales the selected features, tests several values of K, then repeatedly updates cluster centres and record assignments.",
        evidence="A silhouette curve for selecting K, a silhouette score and a coloured two-feature cluster plot.",
        steps=("Select comparable numeric features", "Choose K using silhouette evidence", "Interpret and name the resulting groups"),
        caution="Clusters are mathematical groupings, not automatically meaningful customer types. Validate stability and business usefulness.",
        labs=(LabReference("10", "Select K with silhouette evidence", "lab-10-kmeans-silhouette", "customer_clusters.csv"),),
    ),
    "Hierarchical clustering": WorkflowGuide(
        eyebrow="EXPLORE NESTED GROUPS",
        title="Hierarchical clustering",
        summary="Hierarchical clustering builds a tree of progressively merged records so you can explore grouping structure at more than one level.",
        use_when="You want to inspect how groups combine, compare possible cuts or explain clusters through a dendrogram.",
        mechanism="Ward linkage merges the pair of groups that causes the smallest increase in within-cluster variance at each step.",
        evidence="A dendrogram showing merge distance and a scatter plot coloured by the selected cluster cut.",
        steps=("Select numeric features", "Read large jumps in merge distance", "Choose and interpret a cluster cut"),
        caution="The dendrogram depends on scaling, distance and linkage choices; a visible branch is not proof of a real-world segment.",
        labs=(LabReference("11", "Cut a hierarchical dendrogram", "lab-11-hierarchical-dendrogram", "patient_profiles.csv"),),
    ),
    "PCA & feature ranking": WorkflowGuide(
        eyebrow="REDUCE COMPLEXITY",
        title="PCA & feature ranking",
        summary="PCA compresses correlated numeric features into components, while feature ranking measures which original features most influence prediction quality.",
        use_when="You need fewer dimensions for exploration or want to prioritise original predictors for a supervised model.",
        mechanism="PCA rotates scaled data toward directions of greatest variance. Permutation ranking shuffles one feature at a time and measures the loss in model accuracy.",
        evidence="Cumulative explained variance, a component projection, loading values or a ranked feature-importance chart.",
        steps=("Choose PCA or ranking", "Select the input features", "Interpret variance, loadings or importance"),
        caution="PCA components can be hard to explain, and feature importance is not causality. Interpret results in the dataset context.",
        labs=(
            LabReference("12", "Retain variance with PCA", "lab-12-pca-variance", "patient_profiles.csv"),
            LabReference("13", "Rank predictive features", "lab-13-feature-ranking", "breast_cancer.csv"),
        ),
    ),
    "Association rules": WorkflowGuide(
        eyebrow="FIND CO-OCCURRENCE",
        title="Association rules",
        summary="Association-rule mining finds items that frequently occur together in transactions and expresses their directional relationships as rules.",
        use_when="You have transaction-item data and want basket patterns for merchandising, recommendations or process investigation.",
        mechanism="Apriori prunes infrequent item combinations, then the app forms rules that meet minimum support and confidence thresholds.",
        evidence="Frequent itemsets plus rules measured by support, confidence and lift.",
        steps=("Identify transaction and item fields", "Set support and confidence", "Prioritise rules with useful lift and context"),
        caution="A high-lift rule is an association, not proof that one item causes another. Rare rules may also be unstable.",
        labs=(LabReference("14", "Mine market-basket rules", "lab-14-association-rules", "market_basket.csv"),),
    ),
    "Anomaly detection": WorkflowGuide(
        eyebrow="FLAG UNUSUAL VALUES",
        title="Anomaly detection",
        summary="Anomaly detection flags observations far from the usual range so a person can investigate possible faults, risks or data-quality issues.",
        use_when="You need a transparent first-pass screen for unusual numeric readings or records.",
        mechanism="Z-score measures distance from the mean in standard deviations; IQR uses the middle 50% of values to create robust lower and upper fences.",
        evidence="The number of flagged rows, an anomaly plot, a score and the underlying records for review.",
        steps=("Choose a numeric measure", "Select a transparent rule", "Review flagged rows in context"),
        caution="An anomaly is a review signal, not proof of fraud or failure. Thresholds must reflect domain risk and expected variation.",
        labs=(LabReference("15", "Detect and explain anomalies", "lab-15-anomaly-evidence", "sensor_readings.csv"),),
    ),
}
