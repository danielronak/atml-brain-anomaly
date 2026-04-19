"""
🧠 Brain Anomaly Detection — Streamlit Dashboard
=================================================
Run with:
    conda activate atml
    streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    st.error(
        "**plotly not found.**  "
        "Install it in your conda env:\n"
        "```\nconda activate atml\npip install plotly streamlit\n```"
    )
    st.stop()

try:
    import torch
except ModuleNotFoundError:
    st.error(
        "**PyTorch not found in this Python environment.**\n\n"
        "You are running the wrong Python. Use the atml conda environment:\n"
        "```\nconda activate atml\nstreamlit run app.py\n```"
    )
    st.stop()

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT       = Path(__file__).parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"
RESULTS_DIR = ROOT / "results"

VQVAE_METRICS = RESULTS_DIR / "vqvae"  / "metrics.csv"
SWIN_METRICS  = RESULTS_DIR / "swin_gan" / "metrics.csv"
ALL_METRICS   = RESULTS_DIR / "all_models_metrics.csv"

MODEL_COLS   = {"vqvae": "#5cc47a", "swin_gan": "#5c8de0"}
MODEL_LABELS = {"vqvae": "VQ-VAE ⭐", "swin_gan": "Swin-UNET GAN"}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Brain Anomaly Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Background ── */
  [data-testid="stAppViewContainer"] { background: #080c18; }
  [data-testid="stSidebar"]          { background: #0f1424; border-right: 1px solid #1e2a42; }
  .block-container                   { padding: 1.5rem 2.5rem 3rem; }

  /* ── Metric cards ── */
  .kpi-card {
    background: linear-gradient(145deg, #141929, #1c2640);
    border: 1px solid #253050;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,.35);
  }
  .kpi-label { color:#6b7faa; font-size:.72rem; text-transform:uppercase;
                letter-spacing:1.2px; margin-bottom:6px; }
  .kpi-value { color:#dde4ff; font-size:2rem; font-weight:800;
                font-family:'JetBrains Mono',monospace; }
  .kpi-sub   { color:#5cc47a; font-size:.72rem; margin-top:4px; }

  /* ── Hero header ── */
  .hero-title { font-size:2.8rem; font-weight:900;
    background: linear-gradient(90deg,#5cc47a 30%,#5c8de0);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .hero-sub   { color:#6b7faa; font-size:.95rem; margin-top:-8px; }

  /* ── Section headings ── */
  h2,h3 { color:#c8d0f0 !important; }

  /* ── Info box ── */
  .info-box {
    background:#131a30; border:1px solid #2a3a60; border-radius:10px;
    padding:1rem 1.2rem; color:#8899bb; font-size:.875rem; line-height:1.6;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"]                     { color:#6b7faa; font-weight:500; }
  .stTabs [data-baseweb="tab"][aria-selected="true"]{ color:#5cc47a;
    border-bottom: 2px solid #5cc47a !important; }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

  /* ── Button ── */
  div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#2d5a3d,#2a3e78);
    border:none; color:#e8f0ff; font-weight:600; border-radius:8px;
    padding:.5rem 1.5rem; transition: all .2s ease;
  }
  div.stButton > button[kind="primary"]:hover { filter: brightness(1.2); }

  hr { border-color:#1e2a42 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

@st.cache_data
def load_all_metrics() -> pd.DataFrame:
    if ALL_METRICS.exists():
        return pd.read_csv(ALL_METRICS)
    frames = []
    for name in ("vqvae", "swin_gan"):
        p = RESULTS_DIR / name / "metrics.csv"
        if p.exists():
            df = pd.read_csv(p); df["model"] = name; frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data
def load_model_metrics(name: str) -> pd.DataFrame:
    p = RESULTS_DIR / name / "metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def plotly_dark_layout(**kw) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899bb"),
        margin=dict(t=24, b=10, l=10, r=10),
        **kw,
    )

def arr_slice(tensor, s: int, ch: int = 0) -> np.ndarray:
    """Robustly extract a 2-D axial slice from any tensor shape."""
    import torch
    if isinstance(tensor, torch.Tensor):
        a = tensor.float().cpu().numpy()
    else:
        a = np.array(tensor, dtype=np.float32)
    if a.ndim == 5: a = a[0]          # (1,C,D,H,W) → (C,D,H,W)
    if a.ndim == 4:                    # (C,D,H,W) → (D,H,W)  average or pick channel
        a = a[ch] if ch < a.shape[0] else a.mean(0)
    return a[s]                        # (D,H,W) → (H,W)

def heatmap(z, colorscale="gray", showscale=False, **kw):
    return go.Heatmap(z=z, colorscale=colorscale, showscale=showscale,
                      zmin=0, zmax=1, **kw)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Brain Anomaly")
    st.caption("Unsupervised 3D MRI · IXI + BraTS 2021")
    st.markdown("---")

    st.markdown("### ⚙️ Inference Settings")
    st.caption("Required only for the **Live Inference** tab")

    vqvae_ckpt_str = st.text_input(
        "VQ-VAE checkpoint (.pth)",
        placeholder="/path/to/vqvae/final.pth",
    )
    swin_ckpt_str = st.text_input(
        "Swin GAN checkpoint (.pth)",
        placeholder="/path/to/swin_gan/generator_final.pth",
    )
    brats_dir_str = st.text_input(
        "BraTS data folder",
        placeholder="/path/to/BraTS2021_Training_Data",
    )

    st.markdown("---")
    # Tensor availability check
    vq_tensors = RESULTS_DIR / "vqvae" / "patient_tensors"
    n_tensors = len(list(vq_tensors.glob("*.pt"))) if vq_tensors.exists() else 0
    if n_tensors:
        st.success(f"✅ {n_tensors} patient tensors found locally")
    else:
        st.info("ℹ️ Download patient tensors from Drive for the slice viewer")
        st.caption("`atml/results/*/patient_tensors/*.pt`")

    st.markdown("---")
    st.caption("**ATML · Brain Anomaly Detection**")
    st.caption("VQ-VAE vs Swin-UNET GAN · 50 BraTS patients")


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 Brain Anomaly Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Unsupervised 3-D MRI anomaly detection &nbsp;·&nbsp; '
            'VQ-VAE vs Swin-UNET GAN &nbsp;·&nbsp; 50 BraTS 2021 glioma patients</p>',
            unsafe_allow_html=True)
st.markdown("---")

tab_overview, tab_explorer, tab_infer = st.tabs(
    ["📊  Overview", "🔬  Patient Explorer", "⚡  Live Inference"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab_overview:
    df_all = load_all_metrics()

    # ── KPI cards ────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    def kpi_card(col, label, value, subtitle="", colour="#5cc47a"):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub" style="color:{colour}">{subtitle}</div>
        </div>""", unsafe_allow_html=True)

    if not df_all.empty:
        vq  = df_all[df_all["model"] == "vqvae"]
        sw  = df_all[df_all["model"] == "swin_gan"]
        kpi_card(kpi1, "VQ-VAE Dice",    f"{vq['best_dice'].mean():.3f}",  f"± {vq['best_dice'].std():.3f}")
        kpi_card(kpi2, "VQ-VAE AUROC",   f"{vq['auroc'].mean():.3f}",      "voxel-level ROC")
        kpi_card(kpi3, "VQ-VAE HD95",    f"{vq['hausdorff95'].replace(np.inf,np.nan).mean():.0f} mm", "lower = better", "#e09c55")
        kpi_card(kpi4, "Swin GAN Dice",  f"{sw['best_dice'].mean():.3f}",  f"± {sw['best_dice'].std():.3f}", "#5c8de0")
        kpi_card(kpi5, "Swin GAN AUROC", f"{sw['auroc'].mean():.3f}",      "voxel-level ROC", "#5c8de0")
        best_d = df_all["best_dice"].max()
        best_p = df_all.loc[df_all["best_dice"].idxmax(), "patient_id"]
        kpi_card(kpi6, "Best Patient",   f"{best_d:.3f}", best_p, "#f0c060")
    else:
        for col in [kpi1,kpi2,kpi3,kpi4,kpi5,kpi6]:
            kpi_card(col, "—", "N/A", "run evaluation first", "#888")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Violin plots ─────────────────────────────────────────────
    chart_l, chart_r = st.columns(2)

    def violin(col, title, metric, yaxis_title):
        with col:
            st.markdown(f"#### {title}")
            if df_all.empty:
                st.info("No data yet"); return
            fig = go.Figure()
            for mname in ["vqvae", "swin_gan"]:
                sub = df_all[df_all["model"] == mname][metric]
                if sub.empty: continue
                fig.add_trace(go.Violin(
                    y=sub, name=MODEL_LABELS[mname],
                    line_color=MODEL_COLS[mname], fillcolor=MODEL_COLS[mname],
                    opacity=0.55, box_visible=True, meanline_visible=True,
                    points="all", pointpos=-0.5,
                    marker=dict(color=MODEL_COLS[mname], size=5, opacity=0.55),
                ))
            if metric == "auroc":
                fig.add_hline(y=0.5, line_dash="dot", line_color="#cc4444",
                              annotation_text="Random (0.5)", annotation_position="top right")
            fig.update_layout(**plotly_dark_layout(yaxis_title=yaxis_title, height=360,
                                                   showlegend=True,
                                                   legend=dict(x=0.02, y=0.98)))
            st.plotly_chart(fig, width='stretch')

    violin(chart_l, "Dice Score Distribution",  "best_dice", "Dice ↑")
    violin(chart_r, "AUROC Distribution",        "auroc",     "AUROC ↑")

    st.markdown("---")

    # ── Per-patient table ─────────────────────────────────────────
    st.markdown("#### Per-Patient Results")
    if not df_all.empty:
        disp = df_all.copy()
        disp["Model"]   = disp["model"].map(MODEL_LABELS)
        disp["Patient"] = disp["patient_id"]
        disp = disp.rename(columns={
            "best_dice": "Dice ↑", "auroc": "AUROC ↑",
            "hausdorff95": "HD95 ↓ (mm)", "best_threshold": "Threshold",
        }).sort_values("Dice ↑", ascending=False)
        st.dataframe(
            disp[["Patient","Model","Dice ↑","AUROC ↑","HD95 ↓ (mm)"]],
            width='stretch', hide_index=True,
        )
    else:
        st.info("Run the evaluation notebook and place CSVs in `results/`.")

    st.markdown("---")

    # ── Figures ───────────────────────────────────────────────────
    st.markdown("#### Generated Figures")
    fig_data = [
        ("dice_boxplot.png",              "Dice Box Plot"),
        ("threshold_curves.png",          "Threshold vs Dice"),
        ("reconstruction_grid_best.png",  "Reconstruction Grid (Best Patient)"),
        ("training_curves.png",           "Training Curves"),
    ]
    cols = st.columns(2)
    for i, (fname, title) in enumerate(fig_data):
        fp = RESULTS_DIR / "figures" / fname
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if fp.exists():
                st.image(str(fp), width='stretch')
            else:
                st.markdown(f'<div class="info-box">Figure not found:<br><code>results/figures/{fname}</code></div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown("### 🔬 Patient Explorer")
    st.caption("Browse pre-computed results. Download patient tensors from Drive to enable the slice viewer.")

    df_vq = load_model_metrics("vqvae")
    df_sw = load_model_metrics("swin_gan")

    if df_vq.empty and df_sw.empty:
        st.warning("No per-model metrics CSVs found in `results/`. Run Notebook 05 first.")
        st.stop()

    all_pids = sorted(set(
        (df_vq["patient_id"].tolist() if not df_vq.empty else []) +
        (df_sw["patient_id"].tolist() if not df_sw.empty else [])
    ))

    sel_col, sort_col = st.columns([3, 1])
    with sort_col:
        sort_key = st.selectbox("Sort by", ["Patient ID", "VQ-VAE Dice ↓", "AUROC ↓"], label_visibility="collapsed")
    with sel_col:
        if sort_key == "VQ-VAE Dice ↓" and not df_vq.empty:
            all_pids = df_vq.sort_values("best_dice", ascending=False)["patient_id"].tolist()
        elif sort_key == "AUROC ↓" and not df_vq.empty:
            all_pids = df_vq.sort_values("auroc", ascending=False)["patient_id"].tolist()
        selected_pid = st.selectbox("Select patient", all_pids)

    if not selected_pid:
        st.stop()

    # ── Metrics row ───────────────────────────────────────────────
    row_vq = df_vq[df_vq["patient_id"] == selected_pid].iloc[0] if (not df_vq.empty and selected_pid in df_vq["patient_id"].values) else None
    row_sw = df_sw[df_sw["patient_id"] == selected_pid].iloc[0] if (not df_sw.empty and selected_pid in df_sw["patient_id"].values) else None

    st.markdown(f"#### Patient: `{selected_pid}`")
    mc = st.columns(6)
    def show_metric(col, label, val, delta=None, color="#5cc47a"):
        if val is not None:
            col.metric(label, f"{val:.4f}", delta)

    if row_vq is not None:
        show_metric(mc[0], "🟢 VQ-VAE Dice",  row_vq["best_dice"])
        show_metric(mc[1], "🟢 VQ-VAE AUROC", row_vq["auroc"])
        hd = row_vq["hausdorff95"] if not np.isinf(row_vq["hausdorff95"]) else None
        mc[2].metric("🟢 VQ-VAE HD95", f"{hd:.1f} mm" if hd else "∞")
    if row_sw is not None:
        show_metric(mc[3], "🔵 Swin Dice",  row_sw["best_dice"])
        show_metric(mc[4], "🔵 Swin AUROC", row_sw["auroc"])
        hd2 = row_sw["hausdorff95"] if not np.isinf(row_sw["hausdorff95"]) else None
        mc[5].metric("🔵 Swin HD95", f"{hd2:.1f} mm" if hd2 else "∞")

    st.markdown("---")

    # ── Slice viewer ──────────────────────────────────────────────
    tensor_paths = {
        mn: RESULTS_DIR / mn / "patient_tensors" / f"{selected_pid}.pt"
        for mn in ("vqvae", "swin_gan")
    }
    loaded_tensors = {}
    for mn, tp in tensor_paths.items():
        if tp.exists():
            import torch
            loaded_tensors[mn] = torch.load(tp, map_location="cpu", weights_only=False)

    if not loaded_tensors:
        st.markdown("""
        <div class="info-box">
        <b>Patient tensor files not found locally.</b><br><br>
        To enable the interactive slice viewer, download these files from Google Drive:<br>
        <code>atml/results/vqvae/patient_tensors/{patient_id}.pt</code><br>
        <code>atml/results/swin_gan/patient_tensors/{patient_id}.pt</code><br><br>
        Place them in <code>results/vqvae/patient_tensors/</code> and
        <code>results/swin_gan/patient_tensors/</code>.
        </div>""", unsafe_allow_html=True)
    else:
        import torch
        first = next(iter(loaded_tensors.values()))
        vol_np = first["volume"].float().numpy()
        if vol_np.ndim == 5: vol_np = vol_np[0]
        depth = vol_np.shape[1]   # (C, D, H, W)

        ctrl, viewer = st.columns([1, 3])
        with ctrl:
            st.markdown("**Controls**")
            s        = st.slider("Axial slice", 0, depth-1, depth//2, key="exp_slice")
            ch_label = st.radio("Input channel", ["T1", "T2"], key="exp_ch")
            ch       = 0 if ch_label == "T1" else 1
            active_m = st.radio("Anomaly map from", list(loaded_tensors.keys()),
                                format_func=lambda x: MODEL_LABELS.get(x, x), key="exp_model")

        with viewer:
            t = loaded_tensors[active_m]
            orig_sl  = arr_slice(t["volume"],   s, ch)
            recon_sl = arr_slice(t["recon"],    s, ch)
            res_sl   = arr_slice(t["residual"], s, 0)
            mask_sl  = arr_slice(t["mask"],     s, 0)

            fig_slices = make_subplots(
                rows=1, cols=4,
                subplot_titles=["Input MRI", "Reconstruction", "Anomaly Map", "True Mask"],
                horizontal_spacing=0.02,
            )
            fig_slices.add_trace(heatmap(orig_sl,  "gray"),       row=1, col=1)
            fig_slices.add_trace(heatmap(recon_sl, "gray"),       row=1, col=2)
            fig_slices.add_trace(go.Heatmap(z=res_sl, colorscale="Hot",
                                             showscale=True, zmin=0, zmax=1,
                                             colorbar=dict(len=0.5, thickness=12,
                                                           tickfont=dict(color="#888"))),
                                 row=1, col=3)
            fig_slices.add_trace(heatmap(mask_sl, "Reds"),        row=1, col=4)

            fig_slices.update_layout(
                **plotly_dark_layout(height=310),
            )
            fig_slices.update_xaxes(showticklabels=False)
            fig_slices.update_yaxes(showticklabels=False, autorange="reversed")
            st.plotly_chart(fig_slices, width='stretch')

        # ── Model diff comparison ─────────────────────────────────
        if len(loaded_tensors) == 2:
            st.markdown("#### Side-by-side Anomaly Maps")
            diff_fig = make_subplots(rows=1, cols=3,
                                     subplot_titles=["VQ-VAE Anomaly", "Swin GAN Anomaly", "Difference (VQ - Swin)"],
                                     horizontal_spacing=0.03)
            vq_res = arr_slice(loaded_tensors["vqvae"]["residual"],    s, 0)
            sw_res = arr_slice(loaded_tensors["swin_gan"]["residual"], s, 0)
            diff   = vq_res - sw_res

            diff_fig.add_trace(go.Heatmap(z=vq_res, colorscale="Hot",  showscale=False, zmin=0, zmax=1), row=1, col=1)
            diff_fig.add_trace(go.Heatmap(z=sw_res, colorscale="Blues", showscale=False, zmin=0, zmax=1), row=1, col=2)
            diff_fig.add_trace(go.Heatmap(z=diff,   colorscale="RdBu",  showscale=True,
                                           zmid=0,
                                           colorbar=dict(len=0.6, thickness=12,
                                                         tickfont=dict(color="#888"))),
                                row=1, col=3)
            diff_fig.update_layout(**plotly_dark_layout(height=280))
            diff_fig.update_xaxes(showticklabels=False)
            diff_fig.update_yaxes(showticklabels=False, autorange="reversed")
            st.plotly_chart(diff_fig, width='stretch')

    # ── Threshold-Dice curves from CSV columns ────────────────────
    import ast
    has_sweep = False
    if row_vq is not None and "sweep_thresholds" in row_vq:
        try:
            ts_vq = ast.literal_eval(str(row_vq["sweep_thresholds"]))
            ds_vq = ast.literal_eval(str(row_vq["sweep_dices"]))
            has_sweep = bool(ts_vq)
        except Exception:
            pass

    if has_sweep:
        st.markdown("#### Threshold vs Dice")
        fig_sweep = go.Figure()
        for mname, row, ts_key in [("vqvae", row_vq, "sweep_thresholds"),
                                     ("swin_gan", row_sw, "sweep_thresholds")]:
            if row is None: continue
            try:
                ts = ast.literal_eval(str(row.get("sweep_thresholds", "[]")))
                ds = ast.literal_eval(str(row.get("sweep_dices", "[]")))
                if ts and ds:
                    best_i = int(np.argmax(ds))
                    fig_sweep.add_trace(go.Scatter(
                        x=ts, y=ds, mode="lines",
                        name=f"{MODEL_LABELS.get(mname,mname)} (best={ds[best_i]:.3f} @ t={ts[best_i]:.2f})",
                        line=dict(color=MODEL_COLS.get(mname,"#fff"), width=2.5),
                    ))
                    fig_sweep.add_vline(x=ts[best_i], line_dash="dot",
                                        line_color=MODEL_COLS.get(mname,"#fff"), opacity=0.5)
            except Exception:
                pass
        fig_sweep.update_layout(**plotly_dark_layout(
            xaxis_title="Threshold", yaxis_title="Dice", height=280, showlegend=True,
        ))
        st.plotly_chart(fig_sweep, width='stretch')


# ══════════════════════════════════════════════════════════════════
# TAB 3 — LIVE INFERENCE
# ══════════════════════════════════════════════════════════════════
with tab_infer:
    st.markdown("### ⚡ Live Inference")
    st.caption("Run models on any BraTS patient directly. Needs local checkpoints + BraTS data.")

    # ── Pre-req status ─────────────────────────────────────────────
    vqvae_path = Path(vqvae_ckpt_str).expanduser() if vqvae_ckpt_str else None
    swin_path  = Path(swin_ckpt_str).expanduser()  if swin_ckpt_str  else None
    brats_path = Path(brats_dir_str).expanduser()  if brats_dir_str  else None

    has_vqvae = bool(vqvae_path and vqvae_path.exists())
    has_swin  = bool(swin_path  and swin_path.exists())
    has_brats = bool(brats_path and brats_path.exists())

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("VQ-VAE ckpt",  "✅ Ready" if has_vqvae else "❌ Not set")
    rc2.metric("Swin GAN ckpt","✅ Ready" if has_swin  else "❌ Not set")
    rc3.metric("BraTS folder", "✅ Ready" if has_brats else "❌ Not set")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    rc4.metric("Device", f"🖥 {device_str.upper()}")

    st.markdown("---")

    if not has_brats:
        st.markdown("""
        <div class="info-box">
        <b>Set the BraTS data folder path in the sidebar to continue.</b><br><br>
        The path should point to the folder containing patient subfolders:<br>
        <code>BraTS2021_00000/</code>, <code>BraTS2021_00002/</code>, …<br><br>
        Also set the model checkpoint paths in the sidebar for inference to work.
        </div>""", unsafe_allow_html=True)
        st.stop()

    patient_dirs = sorted(brats_path.glob("BraTS2021_*"))
    if not patient_dirs:
        st.error(f"No `BraTS2021_*` folders found in `{brats_path}`.")
        st.stop()

    # ── Patient + model selection ─────────────────────────────────
    sel_l, sel_r = st.columns([3, 1])
    with sel_l:
        inf_pid = st.selectbox("BraTS patient for inference",
                               [p.name for p in patient_dirs], key="inf_pid")
    with sel_r:
        models_to_run = st.multiselect("Models", ["VQ-VAE", "Swin-UNET GAN"],
                                       default=["VQ-VAE"], key="inf_models")

    p_dir = brats_path / inf_pid
    t1_f  = p_dir / f"{inf_pid}_t1.nii.gz"
    t2_f  = p_dir / f"{inf_pid}_t2.nii.gz"
    seg_f = p_dir / f"{inf_pid}_seg.nii.gz"
    has_seg = seg_f.exists()

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("T1", "✅" if t1_f.exists() else "❌")
    fc2.metric("T2", "✅" if t2_f.exists() else "❌")
    fc3.metric("Segmentation", "✅ (metrics enabled)" if has_seg else "⚠️  (no metrics)")

    run_btn = st.button("🚀  Run Inference", type="primary",
                        disabled=(not models_to_run or not (has_vqvae or has_swin)))

    if run_btn:
        model_map = {"VQ-VAE": "vqvae", "Swin-UNET GAN": "swin_gan"}
        device    = torch.device(device_str)
        config    = load_config()

        try:
            from monai.transforms import (
                Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
                Orientationd, ScaleIntensityRangePercentilesd,
                CropForegroundd, Resized, ConcatItemsd, DeleteItemsd, ToTensord,
            )
            from monai.data import Dataset as MonaiDataset
            from src.evaluation.anomaly_scorer import AnomalyScorer
            from src.evaluation.eval_utils import load_state_dict_flexible, infer_swin_feature_size
            from src.models.vqvae import get_vqvae
            from src.models.swin_generator import get_swin_generator

            sys.path.insert(0, str(ROOT))

            res     = tuple(config["data"]["resolution"])
            pct_lo  = config["data"]["intensity_percentile_low"]
            pct_hi  = config["data"]["intensity_percentile_high"]
            spacing = tuple(config["data"]["spacing"])

            keys = ["t1", "t2", "mask"] if has_seg else ["t1", "t2"]
            seg  = ["mask"] if has_seg else []
            modes = ["bilinear", "bilinear"] + ["nearest"] * len(seg)
            sizes_mode = ["trilinear", "trilinear"] + ["nearest"] * len(seg)

            tfm = Compose([
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Spacingd(keys=keys, pixdim=spacing, mode=modes),
                Orientationd(keys=keys, axcodes="RAS"),
                CropForegroundd(keys=keys, source_key="t1"),
                ScaleIntensityRangePercentilesd(
                    keys=["t1","t2"], lower=pct_lo, upper=pct_hi,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                Resized(keys=keys, spatial_size=res, mode=sizes_mode),
                ConcatItemsd(keys=["t1","t2"], name="image", dim=0),
                DeleteItemsd(keys=["t1","t2"]),
                ToTensord(keys=["image"] + seg),
            ])

            item = {"t1": str(t1_f), "t2": str(t2_f)}
            if has_seg:
                item["mask"] = str(seg_f)

            with st.spinner("⏳ Preprocessing MRI (MONAI pipeline)…"):
                ds      = MonaiDataset([item], transform=tfm)
                batch   = ds[0]
                volume  = batch["image"].unsqueeze(0)
                mask    = batch["mask"].unsqueeze(0) if has_seg else torch.zeros(1, 1, *res)

            depth = volume.shape[2]

            inf_results = {}
            for mdisplay in models_to_run:
                mname = model_map[mdisplay]
                ckpt  = vqvae_path if mname == "vqvae" else swin_path

                if ckpt is None or not ckpt.exists():
                    st.warning(f"⚠️  No checkpoint for {mdisplay}. Set path in sidebar.")
                    continue

                with st.spinner(f"⏳ Running {mdisplay}…"):
                    if mname == "vqvae":
                        model = get_vqvae(config).to(device)
                        model = load_state_dict_flexible(model, ckpt, device)
                        scorer = AnomalyScorer(model, "vqvae", config, device)
                    else:
                        detected_fs = infer_swin_feature_size(ckpt, device)
                        config["swin"]["feature_size"] = detected_fs
                        model = get_swin_generator(config).to(device)
                        model = load_state_dict_flexible(model, ckpt, device)
                        scorer = AnomalyScorer(model, "gan", config, device)

                    result = scorer.score_patient(volume, mask, inf_pid)
                    inf_results[mname] = result
                    m = result["metrics"]
                    st.success(
                        f"✅ **{mdisplay}** — "
                        f"Dice: `{m['best_dice']:.4f}` | "
                        f"AUROC: `{m['auroc']:.4f}` | "
                        f"HD95: `{m['hausdorff95']:.1f}` mm"
                    )

            # ── Results viewer ─────────────────────────────────────
            if inf_results:
                st.markdown("---")
                st.markdown("#### Slice Viewer")

                ctrl_col, view_col = st.columns([1, 3])
                with ctrl_col:
                    s_inf  = st.slider("Axial slice", 0, depth-1, depth//2, key="inf_s")
                    ch_inf = st.radio("Channel", ["T1", "T2"], key="inf_ch")
                    ch_i   = 0 if ch_inf == "T1" else 1

                n_m = len(inf_results)
                with view_col:
                    subplot_titles = (
                        ["Original"] + [MODEL_LABELS.get(m, m) for m in inf_results] +
                        [""] * (n_m + 1) +
                        ["True Mask" if has_seg else "N/A"] + ["Anomaly Map"] * n_m
                    )
                    fig_inf = make_subplots(
                        rows=3, cols=n_m + 1,
                        subplot_titles=subplot_titles[:3*(n_m+1)],
                        row_titles=["Input", "Reconstruction", "Anomaly Map"],
                        horizontal_spacing=0.02, vertical_spacing=0.06,
                    )
                    orig_sl = arr_slice(volume, s_inf, ch_i)
                    mask_sl = arr_slice(mask,   s_inf, 0)

                    # Col 1: input / (blank) / mask
                    fig_inf.add_trace(heatmap(orig_sl, "gray"),  row=1, col=1)
                    fig_inf.add_trace(heatmap(orig_sl, "gray"),  row=2, col=1)
                    fig_inf.add_trace(heatmap(mask_sl, "Reds"),  row=3, col=1)

                    for ci, (mname, res_data) in enumerate(inf_results.items(), 2):
                        recon_sl = arr_slice(res_data["recon"],    s_inf, ch_i)
                        anom_sl  = arr_slice(res_data["residual"], s_inf, 0)
                        fig_inf.add_trace(heatmap(orig_sl,  "gray"), row=1, col=ci)
                        fig_inf.add_trace(heatmap(recon_sl, "gray"), row=2, col=ci)
                        fig_inf.add_trace(go.Heatmap(z=anom_sl, colorscale="Hot",
                                                      showscale=True, zmin=0, zmax=1,
                                                      colorbar=dict(len=0.4, thickness=10,
                                                                    tickfont=dict(color="#888"))),
                                          row=3, col=ci)

                    fig_inf.update_layout(**plotly_dark_layout(height=620))
                    fig_inf.update_xaxes(showticklabels=False)
                    fig_inf.update_yaxes(showticklabels=False, autorange="reversed")
                    st.plotly_chart(fig_inf, width='stretch')

        except ImportError as e:
            st.error(f"Import error: {e}")
            st.info("Make sure you run with: `conda activate atml && streamlit run app.py`")
        except Exception as e:
            st.error(f"Inference failed: {e}")
            import traceback
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
