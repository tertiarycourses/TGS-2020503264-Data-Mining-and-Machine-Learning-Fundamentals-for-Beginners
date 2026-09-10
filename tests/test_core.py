from pathlib import Path

import pandas as pd

from ml_core import (
    cross_validate_classifier, detect_anomalies, feature_ranking, impute_dataframe,
    mine_association_rules, run_classification, run_hierarchical, run_kmeans,
    run_pca, run_regression, silhouette_scan,
)

ROOT=Path(__file__).resolve().parents[1]


def csv(lab,name): return pd.read_csv(next((ROOT/"labs").glob(lab))/name)


def test_preparation_and_regression():
    q=csv("lab-01-*","customer_quality.csv")
    assert impute_dataframe(q).isna().sum().sum()==0
    h=csv("lab-05-*","housing_regression.csv")
    r=run_regression(h,"price_sgd",["floor_area_sqm","floor","building_age","distance_to_mrt_km"])
    assert r.metrics["R2"]>.75


def test_classification_and_validation():
    d=csv("lab-07-*","breast_cancer.csv")
    feats=["mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness"]
    r=run_classification(d,"diagnosis",feats,"Logistic Regression")
    assert r.metrics["Accuracy"]>.85
    cv=cross_validate_classifier(d,"diagnosis",feats,"Random Forest",5)
    assert len(cv)==5 and cv.accuracy.between(0,1).all()
    rank=feature_ranking(d,"diagnosis",feats); assert len(rank)==len(feats)


def test_unsupervised_association_and_anomaly():
    d=csv("lab-10-*","customer_clusters.csv"); scan=silhouette_scan(d,list(d.columns))
    assert int(scan.loc[scan.silhouette.idxmax(),"k"])==3
    out,score,_=run_kmeans(d,list(d.columns),3); assert len(out)==len(d) and score>.5
    p=csv("lab-11-*","patient_profiles.csv"); feats=list(p.columns)
    h,link,_=run_hierarchical(p,feats,2); assert h.cluster.nunique()==2 and link.shape[0]==len(p)-1
    projected,var,loadings,_=run_pca(p,feats,.70); assert var.loc[len(projected.columns)-1,"cumulative"]>=.70
    b=csv("lab-14-*","market_basket.csv"); itemsets,rules=mine_association_rules(b,"transaction_id","item",.20,.60)
    assert not itemsets.empty and not rules.empty
    s=csv("lab-15-*","sensor_readings.csv"); a=detect_anomalies(s.temperature,"Z-score",3.0)
    assert a.is_anomaly.sum()>=3
