# Data Mining and Machine Learning Fundamentals for Beginners

Build beginner-ready data-mining evidence with one Python and Streamlit application across 15 guided labs.

| Course detail | Information |
|---|---|
| Course code | `TGS-2020503264` |
| Programme | WSQ |
| Duration | 2 days · 16 training hours · 2 assessment hours |
| Registration | [View course details and register](https://www.tertiarycourses.com.sg/wsq-data-mining-and-machine-learning-fundamentals-for-beginners.html) |
| Funding | Up to 70% course-fee funding for eligible learners or employers. Eligibility and terms apply; funding is currently listed as valid through 28 February 2027. |

## About the course

This practical beginner course moves from business questions and reliable data preparation to regression, classification, clustering, dimension reduction, and association-rule evidence. Learners use a central Streamlit application on localhost, while every lab supplies its own scenario, mock dataset, verification criteria, troubleshooting guidance, and cleanup.

## Learning outcomes

- Apply data-mining and machine-learning principles to assess business insights.
- Integrate and aggregate information from multiple datasets to build data models.
- Apply predictive modelling techniques to identify trends.
- Apply classification techniques and interpret their errors.
- Apply clustering and interactive visualisation to discover defensible patterns.
- Prototype dimension-reduction and feature-ranking analyses.
- Construct and evaluate association rules from transaction data.

## Topics covered

1. Data-mining evidence loops, machine-learning boundaries, and business impact
2. Data import, schema validation, filtering, joins, imputation, scaling, sampling, and aggregation
3. Linear regression, residuals, MAE/RMSE/R², overfitting, Ridge, and Lasso
4. Decision trees, Logistic Regression, KNN, SVM, Naive Bayes, ensembles, cross-validation, confusion metrics, ROC/AUC, and threshold policy
5. Distance, K-means, silhouette analysis, hierarchical clustering, and cluster interpretation
6. PCA, retained variance, component loadings, and permutation feature ranking
7. Basket encoding, support, confidence, lift, Apriori pruning, and rule deployment

## Run the central lab app

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, choose the workflow named in the lab guide, and upload that lab's mock data.

## Labs

- [Lab 01 — Data audit and business question](labs/lab-01-audit-business-data/README.md)
- [Lab 02 — Clean, impute, and export](labs/lab-02-clean-impute-export/README.md)
- [Lab 03 — Filter, join, and aggregate](labs/lab-03-filter-join-aggregate/README.md)
- [Lab 04 — Scale features before modelling](labs/lab-04-scale-and-compare/README.md)
- [Lab 05 — Build a regression baseline](labs/lab-05-regression-baseline/README.md)
- [Lab 06 — Control overfitting with Ridge and Lasso](labs/lab-06-regularization-overfit/README.md)
- [Lab 07 — Compare classification algorithms](labs/lab-07-classification-baselines/README.md)
- [Lab 08 — Use stratified K-fold validation](labs/lab-08-kfold-validation/README.md)
- [Lab 09 — Interpret confusion metrics and error costs](labs/lab-09-confusion-roc-costs/README.md)
- [Lab 10 — Select K with silhouette evidence](labs/lab-10-kmeans-silhouette/README.md)
- [Lab 11 — Cut a hierarchical dendrogram](labs/lab-11-hierarchical-dendrogram/README.md)
- [Lab 12 — Retain variance with PCA](labs/lab-12-pca-variance/README.md)
- [Lab 13 — Rank predictive features](labs/lab-13-feature-ranking/README.md)
- [Lab 14 — Mine market-basket rules](labs/lab-14-association-rules/README.md)
- [Lab 15 — Detect and explain anomalies](labs/lab-15-anomaly-evidence/README.md)

## Public package

- `app.py` — central Streamlit learner interface
- `ml_core.py` — testable analytics functions
- `requirements.txt` — shared runtime dependencies
- `labs/` — one self-contained folder per lab, including mock data
- `courseware/` — current trainer slide deck, Learner Guide, Lesson Plan, Facilitator Guide, and Assessment Plan in editable and PDF formats

The public repository intentionally excludes assessments and answer keys, source references, credentials, private build tooling, QA renders, and local environment files. Candidate assessment papers are distributed through the LMS and controlled courseware Drive, not GitHub.

## Provider

Developed and delivered by [Tertiary Infotech Academy Pte Ltd](https://www.tertiarycourses.com.sg/) · UEN `201200696W`.
