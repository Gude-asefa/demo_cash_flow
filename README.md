# 💵 Cash-Out Prediction Dashboard

A Streamlit dashboard to evaluate your LightGBM cash-out prediction model across 941+ bank branches.

---

## 📁 Files You Need

Place all these files in the **same folder** as `app.py`:

| File | Description |
|------|-------------|
| `app.py` | The dashboard application |
| `all_cash_out.joblib` | Your trained LightGBM model |
| `features_for_test_dataset.csv` | Test dataset features (108,794 rows × 18 columns) |
| `target_for_test_dataset.csv` | Actual cash-out values for the test set |
| `branch_id_to_name_mapping.csv` | Maps branch IDs (e.g. ET0010003) to branch names |

---

## 🚀 Setup & Run

### Step 1 — Install Python dependencies

Open your terminal / command prompt and run:

```bash
pip install streamlit lightgbm pandas numpy plotly scikit-learn joblib
```

> ✅ You already have `lightgbm` installed (you used it to train the model), so this should be quick.

### Step 2 — Navigate to the dashboard folder

```bash
cd path/to/cash_prediction_dashboard
```

### Step 3 — Launch the dashboard

```bash
streamlit run app.py
```

Your browser will automatically open at **http://localhost:8501** 🎉

---

## 🖥️ What the Dashboard Shows

| Section | Description |
|---------|-------------|
| **Overall Metrics** | MAE, RMSE, MAPE, R² score across all branches |
| **Predicted vs Actual Scatter** | Visual comparison — dots on the diagonal = perfect |
| **Error Distribution** | Histogram showing if the model over/under-predicts |
| **% Error by District** | Which districts are hardest to predict |
| **Top / Bottom Branches** | Best and worst performing branches by MAPE |
| **Per-Branch Explorer** | Deep-dive into any single branch |
| **Full Branch Table** | Sortable & downloadable table for all 941 branches |

---

## 🎛️ Using the Sidebar

- **Filter by District** — Focus on a specific geographic area
- **Top N slider** — Control how many branches appear in the top/bottom charts
- **Show raw table** — Toggle a raw prediction table for debugging

---

## 📖 Metric Glossary

| Metric | Meaning | Good value |
|--------|---------|------------|
| **MAE** | Mean Absolute Error — average prediction error in ETB | As low as possible |
| **RMSE** | Root Mean Squared Error — punishes large errors more | As low as possible |
| **MAPE** | Mean Absolute Percentage Error — error as % of actual value | < 15% is great |
| **R²** | How much variance the model explains (0 = nothing, 1 = perfect) | > 0.85 is good |
| **Bias** | Average over/under-prediction per branch | Close to 0 is ideal |

---

## ❓ Troubleshooting

**"ModuleNotFoundError: No module named 'lightgbm'"**
→ Run `pip install lightgbm`

**"File not found" error**
→ Update the file paths in the sidebar (they default to the same folder as app.py)

**Dashboard is slow to load**
→ Normal on first run — the model predicts 108,794 rows. Subsequent runs are cached and instant.
