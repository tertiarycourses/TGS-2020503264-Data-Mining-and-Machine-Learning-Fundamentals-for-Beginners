# Lab 12 — Retain variance with PCA

**Alignment:** LO6 · K8 · A4/A5
**Central-app workflow:** `PCA & feature ranking`
**Mock data:** `patient_profiles.csv`
**Central Lab Portal:** [alfredang.github.io/data-mining-ml-labs](https://alfredang.github.io/data-mining-ml-labs/)

## Objective

Standardise numeric features, retain at least 70% cumulative variance, and inspect component loadings.

## Before you begin

- Use Python 3.11 or later.
- Work from the repository root so every lab uses the same central application.
- Keep the supplied mock data unchanged; export results under a new filename.

## Detailed procedure

1. Open Terminal and change to the repository root (the folder containing `app.py`).
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the shared dependencies once:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the central application:

   ```bash
   streamlit run app.py
   ```

5. Open `http://localhost:8501`, choose **PCA & feature ranking**, and upload `labs/lab-12-pca-variance/patient_profiles.csv`.
6. Adjust the available model or threshold control and compare the evidence rather than judging from one metric.
8. Record the chosen inputs, parameter values, output metrics, and one screenshot of the visual evidence.
9. Export the result when the workflow provides a download button, then state one business interpretation and one limitation.

## Verification

The chosen component count is the first point where cumulative variance reaches at least 0.70.

Your evidence is complete when it includes the uploaded filename, selected fields, parameters, result table or chart, and a plain-language interpretation.

## Troubleshooting

- **The page is blank:** confirm Streamlit is running and refresh `http://localhost:8501`.
- **A column is missing:** verify that you uploaded this lab's file, not a file from another lab.
- **A model fails:** remove identifier/text fields from numeric-only feature selections and confirm the target is not also selected as a feature.
- **Metrics differ slightly:** use the default random state and unchanged mock data.
- **Port 8501 is busy:** run `streamlit run app.py --server.port 8502` and open the printed URL.

## Cleanup

1. Return to Terminal and press `Control-C` to stop Streamlit.
2. Keep only your exported evidence files; do not overwrite the supplied CSV.
3. Run `deactivate` when you have finished the lab session.
