# Lab 05 — Build a regression baseline

**Alignment:** LO3 · K2/K5 · A2
**Central-app workflow:** `Regression`
**Mock data:** `housing_regression.csv`

## Objective

Train a linear regression model and interpret residual error using MAE, RMSE, and R².

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

5. Open `http://localhost:8501`, choose **Regression**, and upload `labs/lab-05-regression-baseline/housing_regression.csv`.
6. Adjust the available model or threshold control and compare the evidence rather than judging from one metric.
8. Record the chosen inputs, parameter values, output metrics, and one screenshot of the visual evidence.
9. Export the result when the workflow provides a download button, then state one business interpretation and one limitation.

## Verification

R² is above 0.75 on the supplied deterministic dataset.

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
