"""
Cash Prediction Model Dashboard
=================================
A Streamlit app to evaluate and visualize the performance of the
LightGBM cash-out prediction model across 941+ branches.

HOW TO RUN:
    pip install streamlit lightgbm pandas numpy plotly scikit-learn
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cash Prediction Dashboard",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Base */
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    /* App background */
    .stApp { background: #0a0f1e; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1426 !important;
        border-right: 1px solid #1e2d4a;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1a36 0%, #111d3a 100%);
        border: 1px solid #1e3a6e;
        border-radius: 12px;
        padding: 16px 20px !important;
        box-shadow: 0 4px 24px rgba(0,100,255,0.08);
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        color: #4fc3f7 !important;
    }
    [data-testid="stMetricLabel"] { color: #7a8aaa !important; font-size: 0.75rem !important; }
    [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }

    /* Headers */
    h1 { color: #e8f4fd !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
    h2 { color: #c5d8f0 !important; font-weight: 600 !important; }
    h3 { color: #a0bbd8 !important; font-weight: 600 !important; }

    /* Selectbox / inputs */
    [data-baseweb="select"] > div { background: #0d1426 !important; border-color: #1e3a6e !important; color: #e0eaf8 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #1e3a6e; border-radius: 8px; overflow: hidden; }

    /* Dividers */
    hr { border-color: #1e3a6e !important; }

    /* Section boxes */
    .section-box {
        background: linear-gradient(135deg, #0d1a36 0%, #0f1e3d 100%);
        border: 1px solid #1e3a6e;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    /* Status badge */
    .badge-good { background: #0d3d2a; color: #34d399; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .badge-warn { background: #3d2a0d; color: #fbbf24; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .badge-bad  { background: #3d0d0d; color: #f87171; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

    /* Info box */
    .info-box {
        background: #0d1e3a;
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #a0bbd8;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helper: Format ETB currency ─────────────────────────────────────────────
def fmt_etb(value):
    """Format a number as Ethiopian Birr."""
    if abs(value) >= 1_000_000:
        return f"ETB {value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"ETB {value/1_000:.1f}K"
    return f"ETB {value:,.0f}"

# ─── Helper: MAPE ────────────────────────────────────────────────────────────
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ─── Load data (cached so it only runs once) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data(show_spinner="Reading data files...")
def load_data(features_path, target_path, mapping_path):
    features = pd.read_csv(features_path)
    target   = pd.read_csv(target_path)
    mapping  = pd.read_csv(mapping_path)
    return features, target, mapping

@st.cache_data(show_spinner="Running predictions...")
def run_predictions(_model, features_path):
    """
    Run the model on the feature data.
    We pass features_path (a string) instead of the DataFrame so that
    Streamlit's cache key is stable across re-runs.

    IMPORTANT: During training, 5 columns were cast to pandas 'category' dtype
    before being passed to LightGBM.  When we reload a CSV those columns come
    back as plain str/bool, which causes LightGBM to raise
    "train and valid dataset categorical_feature do not match".
    We must re-apply the exact same dtype conversion here.
    """
    # Columns that were categorical during training
    CAT_COLS = [
        "branch_id",
        "day_of_the_week",
        "branch_district",
        "is_weekend",
        "is_payday_window",
    ]

    features = pd.read_csv(features_path)

    for col in CAT_COLS:
        if col in features.columns:
            features[col] = features[col].astype("category")

    predictions = _model.predict(features)
    return predictions

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**📁 Data Files**")
    st.markdown('<div class="info-box">Point these to your local file paths. Defaults use the files in the same folder as app.py.</div>', unsafe_allow_html=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path    = st.text_input("Model (.joblib)",    value=os.path.join(base_dir, "all_cash_out.joblib"))
    features_path = st.text_input("Features CSV",       value=os.path.join(base_dir, "features_for_test_dataset.csv"))
    target_path   = st.text_input("Actuals CSV",        value=os.path.join(base_dir, "target_for_test_dataset.csv"))
    mapping_path  = st.text_input("Branch Mapping CSV", value=os.path.join(base_dir, "branch_id_to_name_mapping.csv"))

    st.markdown("---")
    st.markdown("**🔍 Filters**")
    district_filter = st.multiselect("Filter by District", options=[], key="district_filter")

    st.markdown("---")
    st.markdown("**📊 Chart Options**")
    top_n = st.slider("Top N branches (over/under-performing)", 5, 30, 10)
    show_raw_table = st.checkbox("Show raw prediction table", value=False)

    st.markdown("---")
    st.caption("Cash Prediction Dashboard v1.0")


# ─── Title ───────────────────────────────────────────────────────────────────
st.markdown("# 💵 Cash-Out Prediction Dashboard")
st.markdown("**Model performance evaluation across 941+ branches · Test dataset**")
st.markdown("---")


# ─── Load everything ─────────────────────────────────────────────────────────
files_ok = all(os.path.exists(p) for p in [model_path, features_path, target_path, mapping_path])

if not files_ok:
    st.error("⚠️ One or more files not found. Please update the paths in the sidebar.")
    st.stop()

try:
    model               = load_model(model_path)
    features, target_df, mapping = load_data(features_path, target_path, mapping_path)
    raw_predictions     = run_predictions(model, features_path)
except Exception as e:
    st.error(f"Error loading files or running model: {e}")
    st.stop()


# ─── Build master DataFrame ──────────────────────────────────────────────────
df = features.copy()
df["actual"]     = target_df["teller_cash_out"].values
df["predicted"]  = raw_predictions
df["error"]      = df["actual"] - df["predicted"]
df["abs_error"]  = df["error"].abs()
df["pct_error"]  = ((df["actual"] - df["predicted"]) / df["actual"].replace(0, np.nan)).abs() * 100

# Merge branch names
df = df.merge(mapping, on="branch_id", how="left")
df["branch_name"] = df["branch_name"].fillna(df["branch_id"])

# Update district filter options once we have data
available_districts = sorted(df["branch_district"].dropna().unique().tolist())
district_filter = st.session_state.get("district_filter", [])

# ─── Apply filters ───────────────────────────────────────────────────────────
# Re-render sidebar district options (must do after data load)
with st.sidebar:
    district_filter = st.multiselect(
        "Filter by District",
        options=available_districts,
        default=[],
        key="district_filter_real",
    )

df_filtered = df.copy()
if district_filter:
    df_filtered = df_filtered[df_filtered["branch_district"].isin(district_filter)]

# ─── Overall metrics ─────────────────────────────────────────────────────────
y_true = df_filtered["actual"]
y_pred = df_filtered["predicted"]

mae   = mean_absolute_error(y_true, y_pred)
rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
r2    = r2_score(y_true, y_pred)
mape  = mean_absolute_percentage_error(y_true, y_pred)
n_branches = df_filtered["branch_id"].nunique()
n_records  = len(df_filtered)

st.markdown("### 📈 Overall Model Performance")
# Two rows of 3 — gives each card enough room to display without truncation
row1_c1, row1_c2, row1_c3 = st.columns(3)
row1_c1.metric("🏦 Branches",  f"{n_branches:,}",  help="Number of unique branches evaluated")
row1_c2.metric("📋 Records",   f"{n_records:,}",   help="Total prediction points in the test dataset")
row1_c3.metric("📉 MAPE",      f"{mape:.2f}%",     help="Mean Absolute Percentage Error — average % difference between predicted and actual")

row2_c1, row2_c2, row2_c3 = st.columns(3)
row2_c1.metric("📏 MAE",       fmt_etb(mae),        help="Mean Absolute Error — average prediction error in ETB")
row2_c2.metric("📐 RMSE",      fmt_etb(rmse),       help="Root Mean Squared Error — penalises large errors more heavily than MAE")
row2_c3.metric("🎯 R² Score",  f"{r2:.4f}",         help="R² of 1.0 = perfect model, 0 = no better than the mean")

st.markdown("---")


# ─── Predicted vs Actual scatter ─────────────────────────────────────────────
st.markdown("### 🔵 Predicted vs Actual Cash-Out")

# Sample for performance (max 5,000 points on scatter)
df_sample = df_filtered.sample(min(5000, len(df_filtered)), random_state=42)

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=df_sample["actual"],
    y=df_sample["predicted"],
    mode="markers",
    marker=dict(
        color=df_sample["pct_error"],
        colorscale="RdYlGn_r",
        size=4,
        opacity=0.6,
        colorbar=dict(title=dict(text="% Error", font=dict(color="#a0bbd8")), tickfont=dict(color="#a0bbd8")),
        showscale=True,
    ),
    text=df_sample["branch_name"],
    hovertemplate="<b>%{text}</b><br>Actual: ETB %{x:,.0f}<br>Predicted: ETB %{y:,.0f}<extra></extra>",
    name="Predictions",
))
# Perfect prediction line
max_val = max(df_sample["actual"].max(), df_sample["predicted"].max())
fig_scatter.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val],
    mode="lines",
    line=dict(color="#3b82f6", dash="dash", width=1.5),
    name="Perfect Prediction",
))
fig_scatter.update_layout(
    paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
    font=dict(color="#a0bbd8", family="Sora"),
    xaxis=dict(title="Actual Cash-Out (ETB)", gridcolor="#1e2d4a", tickformat=",.0f"),
    yaxis=dict(title="Predicted Cash-Out (ETB)", gridcolor="#1e2d4a", tickformat=",.0f"),
    height=480,
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d4a"),
    margin=dict(t=30, b=50, l=60, r=20),
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown('<div class="info-box">💡 Each dot is one prediction. Color shows % error — green = accurate, red = large error. Dots on the dashed blue line are perfect predictions.</div>', unsafe_allow_html=True)
st.markdown("---")


# ─── Error distribution ───────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📊 Error Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_filtered["error"] / 1_000_000,
        nbinsx=80,
        marker=dict(color="#3b82f6", line=dict(width=0.3, color="#1e3a6e")),
        opacity=0.85,
        name="Prediction Error",
    ))
    fig_hist.add_vline(x=0, line_color="#f87171", line_dash="dash", line_width=2, annotation_text="Zero Error", annotation_font_color="#f87171")
    fig_hist.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="Error (Millions ETB)", gridcolor="#1e2d4a"),
        yaxis=dict(title="Frequency", gridcolor="#1e2d4a"),
        height=360,
        showlegend=False,
        margin=dict(t=20, b=50, l=60, r=20),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('<div class="info-box">💡 Bars to the left = we over-predicted. Bars to the right = we under-predicted. Narrow bell around zero = good model.</div>', unsafe_allow_html=True)

with col_b:
    st.markdown("### 📊 % Error by District")
    district_mape = (
        df_filtered.groupby("branch_district")
        .apply(lambda g: mean_absolute_percentage_error(g["actual"], g["predicted"]))
        .reset_index()
        .rename(columns={0: "MAPE"})
        .sort_values("MAPE", ascending=True)
    )
    colors = ["#34d399" if v < 15 else "#fbbf24" if v < 30 else "#f87171" for v in district_mape["MAPE"]]
    fig_district = go.Figure(go.Bar(
        x=district_mape["MAPE"],
        y=district_mape["branch_district"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in district_mape["MAPE"]],
        textposition="outside",
        textfont=dict(color="#e0eaf8", size=11),
    ))
    fig_district.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="MAPE (%)", gridcolor="#1e2d4a"),
        yaxis=dict(title=""),
        height=360,
        showlegend=False,
        margin=dict(t=20, b=50, l=180, r=60),
    )
    st.plotly_chart(fig_district, use_container_width=True)
    st.markdown('<div class="info-box">💡 Green = MAPE &lt;15% (good). Yellow = 15–30% (acceptable). Red = &gt;30% (needs attention).</div>', unsafe_allow_html=True)

st.markdown("---")


# ─── Top over/under-performing branches ──────────────────────────────────────
st.markdown("### 🏆 Top Performing vs ⚠️ Under-Performing Branches")

branch_stats = (
    df_filtered.groupby(["branch_id", "branch_name", "branch_district"])
    .agg(
        actual_mean    = ("actual", "mean"),
        predicted_mean = ("predicted", "mean"),
        mae_val        = ("abs_error", "mean"),
        mape_val       = ("pct_error", "mean"),
        n_records      = ("actual", "count"),
    )
    .reset_index()
)
branch_stats["bias"] = branch_stats["predicted_mean"] - branch_stats["actual_mean"]  # positive = over-predict

col_top, col_bot = st.columns(2)

with col_top:
    st.markdown(f"#### ✅ Top {top_n} Most Accurate Branches (lowest MAPE)")
    top_branches = branch_stats.nsmallest(top_n, "mape_val")
    fig_top = go.Figure(go.Bar(
        y=top_branches["branch_name"],
        x=top_branches["mape_val"],
        orientation="h",
        marker=dict(
            color=top_branches["mape_val"],
            colorscale=[[0, "#0d3d2a"], [0.5, "#059669"], [1, "#34d399"]],
        ),
        text=[f"{v:.1f}%" for v in top_branches["mape_val"]],
        textposition="outside",
        textfont=dict(color="#e0eaf8", size=10),
        hovertemplate="<b>%{y}</b><br>MAPE: %{x:.1f}%<extra></extra>",
    ))
    fig_top.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="MAPE (%)", gridcolor="#1e2d4a"),
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 32),
        margin=dict(t=10, b=40, l=200, r=80),
        showlegend=False,
    )
    st.plotly_chart(fig_top, use_container_width=True)

with col_bot:
    st.markdown(f"#### ⚠️ Top {top_n} Hardest Branches (highest MAPE)")
    bot_branches = branch_stats.nlargest(top_n, "mape_val")
    fig_bot = go.Figure(go.Bar(
        y=bot_branches["branch_name"],
        x=bot_branches["mape_val"],
        orientation="h",
        marker=dict(
            color=bot_branches["mape_val"],
            colorscale=[[0, "#fbbf24"], [0.5, "#ef4444"], [1, "#991b1b"]],
        ),
        text=[f"{v:.1f}%" for v in bot_branches["mape_val"]],
        textposition="outside",
        textfont=dict(color="#e0eaf8", size=10),
        hovertemplate="<b>%{y}</b><br>MAPE: %{x:.1f}%<extra></extra>",
    ))
    fig_bot.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="MAPE (%)", gridcolor="#1e2d4a"),
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 32),
        margin=dict(t=10, b=40, l=200, r=80),
        showlegend=False,
    )
    st.plotly_chart(fig_bot, use_container_width=True)

st.markdown('<div class="info-box">💡 MAPE = Mean Absolute Percentage Error per branch. Lower is better. Branches with high MAPE may have unusual cash-flow patterns the model hasn\'t fully learned.</div>', unsafe_allow_html=True)
st.markdown("---")


# ─── Per-branch prediction breakdown ─────────────────────────────────────────
st.markdown("### 🔍 Per-Branch Prediction Explorer")

search_col, sort_col = st.columns([3, 1])
with search_col:
    selected_branch = st.selectbox(
        "Select a branch to inspect",
        options=sorted(df_filtered["branch_name"].unique()),
    )
with sort_col:
    chart_metric = st.selectbox("Compare by", ["Actual vs Predicted", "Error Over Time"])

branch_data = df_filtered[df_filtered["branch_name"] == selected_branch].reset_index(drop=True)
branch_data["period"] = range(1, len(branch_data) + 1)  # sequential day index

b_mae  = mean_absolute_error(branch_data["actual"], branch_data["predicted"])
b_mape = mean_absolute_percentage_error(branch_data["actual"], branch_data["predicted"])
b_r2   = r2_score(branch_data["actual"], branch_data["predicted"]) if len(branch_data) > 2 else 0

bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Branch MAE",        fmt_etb(b_mae))
bc2.metric("Branch MAPE",       f"{b_mape:.1f}%")
bc3.metric("R² (this branch)",  f"{b_r2:.4f}")
bc4.metric("Records",           f"{len(branch_data)}")

if chart_metric == "Actual vs Predicted":
    fig_branch = go.Figure()
    fig_branch.add_trace(go.Scatter(
        x=branch_data["period"], y=branch_data["actual"],
        mode="lines", name="Actual",
        line=dict(color="#34d399", width=2),
    ))
    fig_branch.add_trace(go.Scatter(
        x=branch_data["period"], y=branch_data["predicted"],
        mode="lines", name="Predicted",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    ))
    fig_branch.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="Day Index (sequential)", gridcolor="#1e2d4a"),
        yaxis=dict(title="Cash-Out (ETB)", gridcolor="#1e2d4a", tickformat=",.0f"),
        height=360,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=50, l=80, r=20),
    )
else:
    fig_branch = go.Figure()
    fig_branch.add_trace(go.Bar(
        x=branch_data["period"], y=branch_data["error"] / 1000,
        marker=dict(color=["#f87171" if e < 0 else "#34d399" for e in branch_data["error"]]),
        name="Error (K ETB)",
    ))
    fig_branch.add_hline(y=0, line_color="#7a8aaa", line_width=1)
    fig_branch.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="Day Index", gridcolor="#1e2d4a"),
        yaxis=dict(title="Error (Thousands ETB)", gridcolor="#1e2d4a"),
        height=360,
        showlegend=False,
        margin=dict(t=20, b=50, l=80, r=20),
    )

st.plotly_chart(fig_branch, use_container_width=True)
st.markdown("---")


# ─── Full branch table ────────────────────────────────────────────────────────
st.markdown("### 📋 Full Branch Performance Table")

display_cols = ["branch_name", "branch_district", "n_records", "actual_mean", "predicted_mean", "mae_val", "mape_val", "bias"]
table_df = branch_stats[display_cols].copy()
table_df.columns = ["Branch Name", "District", "Records", "Avg Actual (ETB)", "Avg Predicted (ETB)", "MAE (ETB)", "MAPE (%)", "Bias (ETB)"]

# Format numeric columns
for col in ["Avg Actual (ETB)", "Avg Predicted (ETB)", "MAE (ETB)", "Bias (ETB)"]:
    table_df[col] = table_df[col].apply(lambda x: f"{x:,.0f}")
table_df["MAPE (%)"] = table_df["MAPE (%)"].apply(lambda x: f"{x:.2f}%")

sort_by = st.selectbox("Sort table by", ["MAPE (%)", "Branch Name", "District", "MAE (ETB)"])
asc = st.checkbox("Ascending order", value=True)
table_df = table_df.sort_values(sort_by, ascending=asc)

st.dataframe(table_df, use_container_width=True, height=400)
st.download_button(
    label="📥 Download Table as CSV",
    data=table_df.to_csv(index=False).encode("utf-8"),
    file_name="branch_prediction_performance.csv",
    mime="text/csv",
)
st.markdown("---")


# ─── Optional: raw predictions table ─────────────────────────────────────────
if show_raw_table:
    st.markdown("### 🗃️ Raw Prediction Data")
    raw_display = df_filtered[["branch_id", "branch_name", "branch_district", "day_of_the_week", "actual", "predicted", "error", "pct_error"]].copy()
    raw_display.columns = ["Branch ID", "Branch Name", "District", "Day", "Actual (ETB)", "Predicted (ETB)", "Error (ETB)", "MAPE (%)"]
    st.dataframe(raw_display.head(1000), use_container_width=True, height=400)
    st.caption("Showing first 1,000 rows. Download the branch table above for full data.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#3a5070; font-size:0.78rem;">Cash-Out Prediction Dashboard · LightGBM Model · 941+ Branches · Test Dataset Evaluation</div>',
    unsafe_allow_html=True,
)