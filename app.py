import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import io
import hashlib
import tempfile
from sklearn.metrics import mean_absolute_error, r2_score

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
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp { background: #0a0f1e; }

    [data-testid="stSidebar"] {
        background: #0d1426 !important;
        border-right: 1px solid #1e2d4a;
    }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1a36 0%, #111d3a 100%);
        border: 1px solid #1e3a6e;
        border-radius: 12px;
        padding: 16px 20px !important;
        box-shadow: 0 4px 24px rgba(0,100,255,0.08);
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        color: #4fc3f7 !important;
    }
    [data-testid="stMetricLabel"] { color: #7a8aaa !important; font-size: 0.75rem !important; }

    h1 { color: #e8f4fd !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
    h2 { color: #c5d8f0 !important; font-weight: 600 !important; }
    h3 { color: #a0bbd8 !important; font-weight: 600 !important; }

    [data-baseweb="select"] > div { background: #0d1426 !important; border-color: #1e3a6e !important; color: #e0eaf8 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #1e3a6e; border-radius: 8px; overflow: hidden; }
    hr { border-color: #1e3a6e !important; }

    .info-box {
        background: #0d1e3a;
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #a0bbd8;
    }
    .upload-box {
        background: #0d1e3a;
        border: 1px dashed #1e3a6e;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.8rem;
        color: #4a6080;
    }
    [data-testid="stExpander"] {
        background: #0d1426 !important;
        border: 1px solid #1e2d4a !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def fmt_etb(value):
    if abs(value) >= 1_000_000:
        return f"ETB {value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"ETB {value/1_000:.1f}K"
    return f"ETB {value:,.0f}"

def mape_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def file_hash(b: bytes) -> str:
    """MD5 of raw bytes — used as cache key so uploads bust the cache."""
    return hashlib.md5(b).hexdigest()

# ─── Cached loaders (keyed on content hash, not path) ────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model_from_bytes(model_bytes: bytes, _hash: str):
    """Write bytes to a temp file and load with joblib."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    model = joblib.load(tmp_path)
    os.unlink(tmp_path)
    return model

@st.cache_data(show_spinner="Reading data files...")
def load_data_from_bytes(features_bytes, target_bytes, mapping_bytes,
                          _fhash, _thash, _mhash):
    features = pd.read_csv(io.BytesIO(features_bytes))
    target   = pd.read_csv(io.BytesIO(target_bytes))
    mapping  = pd.read_csv(io.BytesIO(mapping_bytes))
    return features, target, mapping

@st.cache_data(show_spinner="Running predictions...")
def run_predictions_from_bytes(model_bytes, features_bytes, _mhash, _fhash):
    CAT_COLS = [
        "branch_id", "day_of_the_week", "branch_district",
        "is_weekend", "is_payday_window", "is_spike",
    ]
    # Re-load model from bytes
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    model = joblib.load(tmp_path)
    os.unlink(tmp_path)

    features = pd.read_csv(io.BytesIO(features_bytes))
    for col in CAT_COLS:
        if col in features.columns:
            features[col] = features[col].astype("category")
    return model.predict(features)

# ─── Helper: resolve a file to bytes ─────────────────────────────────────────
def resolve_file(uploaded_file, local_path: str) -> bytes | None:
    """
    Return file bytes from whichever source is available:
      1. An uploaded file object (from st.file_uploader)
      2. A local path on disk (works when running locally with files present)
    Returns None if neither is available.
    """
    if uploaded_file is not None:
        return uploaded_file.read()
    if local_path and os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()
    return None

# ─── Default model registry ──────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODELS = [
    {
        "label":      "All_branches_Cash Out",
        "model":      "all_cash_out.joblib",
        "features":   "cash_out/all_out_features_for_test_dataset.csv",
        "target":     "cash_out/all_out_target_for_test_dataset.csv",
        "mapping":    "branch_id_to_name_mapping.csv",
        "target_col": "teller_cash_out",
    },
    {
        "label":      "All_branches_Cash In",
        "model":      "all_cash_in.joblib",
        "features":   "cash_in/all_in_features_for_test_dataset.csv",
        "target":     "cash_in/all_in_target_for_test_dataset.csv",
        "mapping":    "branch_id_to_name_mapping.csv",
        "target_col": "teller_cash_in",
    },
    {
        "label":      "Addis_Ababa_region_Cash In",
        "model":      "addis_cash_in.joblib",
        "features":   "cash_in/addis_in_features_for_test_dataset.csv",
        "target":     "cash_in/addis_in_target_for_test_dataset.csv",
        "mapping":    "branch_id_to_name_mapping.csv",
        "target_col": "teller_cash_in",
    },
    {
        "label":      "Addis_Ababa_region_Cash Out",
        "model":      "addis_cash_out.joblib",
        "features":   "cash_out/addis_out_features_for_test_dataset.csv",
        "target":     "cash_out/addis_out_target_for_test_dataset.csv",
        "mapping":    "branch_id_to_name_mapping.csv",
        "target_col": "teller_cash_out",
    },
    {
        "label":      "Other_regions_Cash In",
        "model":      "regions_cash_in.joblib",
        "features":   "cash_in/other_regions_in_features_for_test_dataset.csv",
        "target":     "cash_in/other_regions_in_target_for_test_dataset.csv",
        "mapping":    "branch_id_to_name_mapping.csv",
        "target_col": "teller_cash_in",
    },
    {
        "label":      "Other_regions_Cash Out",
        "model":      "regions_cash_out.joblib",
        "features":   "cash_out/other_regions_out_features_for_test_dataset.csv",
        "target":     "cash_out/other_regions_out_target_for_test_dataset.csv",
        "mapping":    "branch_id_to_name_mapping.csv",
        "target_col": "teller_cash_out",
    },
]

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Model selector ──
    st.markdown("**🤖 Active Model**")
    model_labels = [m["label"] for m in DEFAULT_MODELS]
    selected_model_label = st.selectbox("Choose model to evaluate", model_labels)
    active_model_cfg = next(m for m in DEFAULT_MODELS if m["label"] == selected_model_label)

    # Absolute local paths for this model slot
    local_model    = os.path.join(base_dir, active_model_cfg["model"])
    local_features = os.path.join(base_dir, active_model_cfg["features"])
    local_target   = os.path.join(base_dir, active_model_cfg["target"])
    local_mapping  = os.path.join(base_dir, active_model_cfg["mapping"])

    # Check how many local files exist so we can show the right prompt
    local_files_exist = all(os.path.exists(p) for p in [local_model, local_features, local_target, local_mapping])

    st.markdown("---")

    # ── File upload section ──
    # Only shown when local files are NOT present (i.e. deployed / other machine)
    if local_files_exist:
        st.markdown('<div class="info-box">✅ Local files detected — running automatically.</div>', unsafe_allow_html=True)
        uploaded_model    = None
        uploaded_features = None
        uploaded_target   = None
        uploaded_mapping  = None
    else:
        st.markdown("**📁 Upload Model Files**")
        st.markdown('<div class="upload-box">Local files not found. Please upload the four files for the selected model.</div>', unsafe_allow_html=True)
        uploaded_model    = st.file_uploader(f"Model (.joblib) — {active_model_cfg['model']}",    type=["joblib"], key=f"model_{selected_model_label}")
        uploaded_features = st.file_uploader(f"Features CSV — {active_model_cfg['features']}",   type=["csv"],    key=f"feat_{selected_model_label}")
        uploaded_target   = st.file_uploader(f"Actuals CSV — {active_model_cfg['target']}",      type=["csv"],    key=f"tgt_{selected_model_label}")
        uploaded_mapping  = st.file_uploader(f"Branch Mapping CSV — {active_model_cfg['mapping']}", type=["csv"], key=f"map_{selected_model_label}")

    st.markdown("---")
    st.markdown("**🔍 Filters**")
    district_placeholder = st.empty()

    st.markdown("---")
    st.markdown("**📊 Chart Options**")
    top_n          = st.slider("Top N branches (over/under-performing)", 5, 30, 10)
    show_raw_table = st.checkbox("Show raw prediction table", value=False)

    st.markdown("---")
    st.caption("Cash Prediction Dashboard v2.1")

# ─── Title ───────────────────────────────────────────────────────────────────
st.markdown(f"# 💵 Cash Prediction Dashboard — {selected_model_label}")
st.markdown("**Model performance evaluation across branches · Test dataset**")
st.markdown("---")

# ─── Resolve all four files to bytes ─────────────────────────────────────────
model_bytes    = resolve_file(uploaded_model,    local_model)
features_bytes = resolve_file(uploaded_features, local_features)
target_bytes   = resolve_file(uploaded_target,   local_target)
mapping_bytes  = resolve_file(uploaded_mapping,  local_mapping)

missing_labels = []
if model_bytes    is None: missing_labels.append(f"Model (.joblib) — `{active_model_cfg['model']}`")
if features_bytes is None: missing_labels.append(f"Features CSV — `{active_model_cfg['features']}`")
if target_bytes   is None: missing_labels.append(f"Actuals CSV — `{active_model_cfg['target']}`")
if mapping_bytes  is None: missing_labels.append(f"Branch Mapping CSV — `{active_model_cfg['mapping']}`")

if missing_labels:
    st.warning("⚠️ Missing files. Please upload the following in the sidebar:")
    for label in missing_labels:
        st.markdown(f"- {label}")
    st.stop()

# ─── Load & run model (cache busted by content hash) ─────────────────────────
mh = file_hash(model_bytes)
fh = file_hash(features_bytes)
th = file_hash(target_bytes)
mah = file_hash(mapping_bytes)

try:
    features, target_df, mapping = load_data_from_bytes(
        features_bytes, target_bytes, mapping_bytes, fh, th, mah
    )
    raw_predictions = run_predictions_from_bytes(model_bytes, features_bytes, mh, fh)
except Exception as e:
    st.error(f"Error loading or running model: {e}")
    st.stop()

target_col = active_model_cfg["target_col"]

# ─── Build master DataFrame ───────────────────────────────────────────────────
df = features.copy()
df["actual"]    = target_df[target_col].values
df["predicted"] = raw_predictions
df["error"]     = df["actual"] - df["predicted"]
df["abs_error"] = df["error"].abs()
df["pct_error"] = ((df["actual"] - df["predicted"]) / df["actual"].replace(0, np.nan)).abs() * 100

df = df.merge(mapping, on="branch_id", how="left")
df["branch_name"] = df["branch_name"].fillna(df["branch_id"])

# ─── District filter ──────────────────────────────────────────────────────────
available_districts = sorted(df["branch_district"].dropna().unique().tolist())
with st.sidebar:
    district_filter = st.multiselect("Filter by District", options=available_districts, default=[])

df_filtered = df[df["branch_district"].isin(district_filter)] if district_filter else df.copy()

# ─── Overall metrics ─────────────────────────────────────────────────────────
y_true     = df_filtered["actual"]
y_pred     = df_filtered["predicted"]
mae        = mean_absolute_error(y_true, y_pred)
r2         = r2_score(y_true, y_pred)
mape       = mape_score(y_true, y_pred)
n_branches = df_filtered["branch_id"].nunique()
n_records  = len(df_filtered)

st.markdown("### 📈 Overall Model Performance")

row1_c1, row1_c2, row1_c3 = st.columns(3)
row1_c1.metric("🏦 Branches",  f"{n_branches:,}",   help="Number of unique branches evaluated")
row1_c2.metric("📋 Records",   f"{n_records:,}",    help="Total prediction rows in the test dataset")
row1_c3.metric("📉 MAPE",      f"{mape:.2f}%",      help="Mean Absolute % Error")

row2_c1, row2_c2, row2_c3 = st.columns(3)
row2_c1.metric("📏 MAE",       fmt_etb(mae),         help="Mean Absolute Error in ETB")
row2_c2.metric("🎯 R² Score",  f"{r2:.4f}",          help="R² of 1.0 = perfect model")
over_pct = (df_filtered["error"] < 0).mean() * 100
row2_c3.metric("⬆️ Over-predicted", f"{over_pct:.1f}%", help="% of time the model predicted MORE than actual")

st.markdown("---")

# ─── % Error by District ──────────────────────────────────────────────────────
st.markdown("### 🗺️ Prediction Accuracy by District")

district_mape = (
    df_filtered.groupby("branch_district")
    .apply(lambda g: mape_score(g["actual"], g["predicted"]))
    .reset_index()
    .rename(columns={0: "MAPE"})
    .sort_values("MAPE", ascending=True)
)
bar_colors = ["#34d399" if v < 15 else "#fbbf24" if v < 30 else "#f87171" for v in district_mape["MAPE"]]

fig_district = go.Figure(go.Bar(
    x=district_mape["MAPE"],
    y=district_mape["branch_district"],
    orientation="h",
    marker=dict(color=bar_colors),
    text=[f"{v:.1f}%" for v in district_mape["MAPE"]],
    textposition="outside",
    textfont=dict(color="#e0eaf8", size=11),
    hovertemplate="<b>%{y}</b><br>MAPE: %{x:.2f}%<extra></extra>",
))
fig_district.update_layout(
    paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
    font=dict(color="#a0bbd8", family="Sora"),
    xaxis=dict(title="MAPE (%)", gridcolor="#1e2d4a"),
    yaxis=dict(title=""),
    height=max(300, len(district_mape) * 38),
    showlegend=False,
    margin=dict(t=20, b=50, l=200, r=80),
)
st.plotly_chart(fig_district, use_container_width=True)
st.markdown('<div class="info-box">💡 Green = MAPE &lt;15% (accurate). Yellow = 15–30% (acceptable). Red = &gt;30% (needs attention).</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Top / Bottom performing branches ────────────────────────────────────────
st.markdown("### 🏆 Top Performing vs ⚠️ Hardest Branches")

branch_stats = (
    df_filtered.groupby(["branch_id", "branch_name", "branch_district"])
    .agg(
        actual_mean    = ("actual",    "mean"),
        predicted_mean = ("predicted", "mean"),
        mae_val        = ("abs_error", "mean"),
        mape_val       = ("pct_error", "mean"),
        n_records      = ("actual",    "count"),
    )
    .reset_index()
)
branch_stats["bias"] = branch_stats["predicted_mean"] - branch_stats["actual_mean"]

col_top, col_bot = st.columns(2)

with col_top:
    st.markdown(f"#### ✅ Top {top_n} Most Accurate (lowest MAPE)")
    top_branches = branch_stats.nsmallest(top_n, "mape_val")
    fig_top = go.Figure(go.Bar(
        y=top_branches["branch_name"],
        x=top_branches["mape_val"],
        orientation="h",
        marker=dict(color=top_branches["mape_val"],
                    colorscale=[[0,"#0d3d2a"],[0.5,"#059669"],[1,"#34d399"]]),
        text=[f"{v:.1f}%" for v in top_branches["mape_val"]],
        textposition="outside",
        textfont=dict(color="#e0eaf8", size=10),
        hovertemplate="<b>%{y}</b><br>MAPE: %{x:.2f}%<extra></extra>",
    ))
    fig_top.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="MAPE (%)", gridcolor="#1e2d4a"),
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 34),
        margin=dict(t=10, b=40, l=200, r=80),
        showlegend=False,
    )
    st.plotly_chart(fig_top, use_container_width=True)

with col_bot:
    st.markdown(f"#### ⚠️ Top {top_n} Hardest (highest MAPE)")
    bot_branches = branch_stats.nlargest(top_n, "mape_val")
    fig_bot = go.Figure(go.Bar(
        y=bot_branches["branch_name"],
        x=bot_branches["mape_val"],
        orientation="h",
        marker=dict(color=bot_branches["mape_val"],
                    colorscale=[[0,"#fbbf24"],[0.5,"#ef4444"],[1,"#991b1b"]]),
        text=[f"{v:.1f}%" for v in bot_branches["mape_val"]],
        textposition="outside",
        textfont=dict(color="#e0eaf8", size=10),
        hovertemplate="<b>%{y}</b><br>MAPE: %{x:.2f}%<extra></extra>",
    ))
    fig_bot.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="MAPE (%)", gridcolor="#1e2d4a"),
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 34),
        margin=dict(t=10, b=40, l=200, r=80),
        showlegend=False,
    )
    st.plotly_chart(fig_bot, use_container_width=True)

st.markdown('<div class="info-box">💡 Branches with high MAPE may have unusual or highly variable cash-flow patterns the model hasn\'t fully learned yet.</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Per-branch prediction explorer ──────────────────────────────────────────
st.markdown("### 🔍 Per-Branch Prediction Explorer")

search_col, sort_col = st.columns([3, 1])
with search_col:
    selected_branch = st.selectbox(
        "Select a branch to inspect",
        options=sorted(df_filtered["branch_name"].unique()),
    )
with sort_col:
    chart_type = st.selectbox("View", ["Actual vs Predicted", "Error Over Time"])

branch_data = df_filtered[df_filtered["branch_name"] == selected_branch].reset_index(drop=True)
branch_data["period"] = range(1, len(branch_data) + 1)

b_mae  = mean_absolute_error(branch_data["actual"], branch_data["predicted"])
b_mape = mape_score(branch_data["actual"], branch_data["predicted"])
b_r2   = r2_score(branch_data["actual"], branch_data["predicted"]) if len(branch_data) > 2 else 0

bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Branch MAE",       fmt_etb(b_mae))
bc2.metric("Branch MAPE",      f"{b_mape:.2f}%")
bc3.metric("R² (this branch)", f"{b_r2:.4f}")
bc4.metric("Records",          f"{len(branch_data)}")

if chart_type == "Actual vs Predicted":
    fig_branch = go.Figure()
    fig_branch.add_trace(go.Scatter(
        x=branch_data["period"],
        y=branch_data["actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#34d399", width=2),
        marker=dict(size=4),
        customdata=np.stack([branch_data["predicted"], branch_data["error"]], axis=1),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "🟢 Actual:    <b>ETB %{y:,.0f}</b><br>"
            "🟡 Predicted: <b>ETB %{customdata[0]:,.0f}</b><br>"
            "📐 Error:     ETB %{customdata[1]:,.0f}"
            "<extra></extra>"
        ),
    ))
    fig_branch.add_trace(go.Scatter(
        x=branch_data["period"],
        y=branch_data["predicted"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        marker=dict(size=4),
        customdata=np.stack([branch_data["actual"], branch_data["error"]], axis=1),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "🟡 Predicted: <b>ETB %{y:,.0f}</b><br>"
            "🟢 Actual:    <b>ETB %{customdata[0]:,.0f}</b><br>"
            "📐 Error:     ETB %{customdata[1]:,.0f}"
            "<extra></extra>"
        ),
    ))
    fig_branch.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="Day Index (sequential)", gridcolor="#1e2d4a"),
        yaxis=dict(title="Cash (ETB)", gridcolor="#1e2d4a", tickformat=",.0f"),
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a6e"),
        hovermode="x unified",
        margin=dict(t=20, b=50, l=90, r=20),
    )
    st.markdown('<div class="info-box">💡 Hover anywhere on the chart to see Actual, Predicted, and Error for that day side by side.</div>', unsafe_allow_html=True)

else:
    fig_branch = go.Figure()
    fig_branch.add_trace(go.Bar(
        x=branch_data["period"],
        y=branch_data["error"] / 1000,
        marker=dict(color=["#f87171" if e < 0 else "#34d399" for e in branch_data["error"]]),
        name="Error (K ETB)",
        customdata=np.stack([branch_data["actual"], branch_data["predicted"]], axis=1),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "📐 Error: ETB %{y:.1f}K<br>"
            "🟢 Actual:    ETB %{customdata[0]:,.0f}<br>"
            "🟡 Predicted: ETB %{customdata[1]:,.0f}"
            "<extra></extra>"
        ),
    ))
    fig_branch.add_hline(y=0, line_color="#7a8aaa", line_width=1)
    fig_branch.update_layout(
        paper_bgcolor="#0d1426", plot_bgcolor="#0a0f1e",
        font=dict(color="#a0bbd8", family="Sora"),
        xaxis=dict(title="Day Index", gridcolor="#1e2d4a"),
        yaxis=dict(title="Error (Thousands ETB)", gridcolor="#1e2d4a"),
        height=400,
        showlegend=False,
        margin=dict(t=20, b=50, l=90, r=20),
    )
    st.markdown('<div class="info-box">💡 Green bars = model under-predicted. Red bars = model over-predicted.</div>', unsafe_allow_html=True)

st.plotly_chart(fig_branch, use_container_width=True)
st.markdown("---")

# ─── Full branch performance table ───────────────────────────────────────────
st.markdown("### 📋 Full Branch Performance Table")

display_cols = ["branch_name","branch_district","n_records","actual_mean","predicted_mean","mae_val","mape_val","bias"]
table_df = branch_stats[display_cols].copy()
table_df.columns = ["Branch Name","District","Records","Avg Actual (ETB)","Avg Predicted (ETB)","MAE (ETB)","MAPE (%)","Bias (ETB)"]

for col in ["Avg Actual (ETB)","Avg Predicted (ETB)","MAE (ETB)","Bias (ETB)"]:
    table_df[col] = table_df[col].apply(lambda x: f"{x:,.0f}")
table_df["MAPE (%)"] = table_df["MAPE (%)"].apply(lambda x: f"{x:.2f}%")

tbl_sort = st.selectbox("Sort table by", ["MAPE (%)","Branch Name","District","MAE (ETB)"])
tbl_asc  = st.checkbox("Ascending order", value=True)
table_df = table_df.sort_values(tbl_sort, ascending=tbl_asc)

st.dataframe(table_df, use_container_width=True, height=420)
st.download_button(
    label="📥 Download as CSV",
    data=table_df.to_csv(index=False).encode("utf-8"),
    file_name=f"branch_performance_{selected_model_label.replace(' ','_')}.csv",
    mime="text/csv",
)
st.markdown("---")

# ─── Optional raw table ───────────────────────────────────────────────────────
if show_raw_table:
    st.markdown("### 🗃️ Raw Prediction Data")
    raw = df_filtered[["branch_id","branch_name","branch_district","day_of_the_week","actual","predicted","error","pct_error"]].copy()
    raw.columns = ["Branch ID","Branch Name","District","Day","Actual (ETB)","Predicted (ETB)","Error (ETB)","MAPE (%)"]
    st.dataframe(raw.head(1000), use_container_width=True, height=400)
    st.caption("Showing first 1,000 rows.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div style="text-align:center;color:#3a5070;font-size:0.78rem;">'
    f'Cash Prediction Dashboard · {selected_model_label} Model · LightGBM · Test Dataset Evaluation'
    f'</div>',
    unsafe_allow_html=True,
)