import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path("data/output")
PLOTS_ROOT  = Path("plots")
SKIP_LOGS   = set()

# Style
PALETTE = {
    "accumulated":   "#2ecc71",
    "baseline_cf":   "#3498db",
    "noisy":         "#e74c3c",
    "corrected":     "#e67e22",
    "offline_base":  "#9b59b6",
    "accum":         "#1abc9c",
    "filtered_sa_accum":   "#27ae60",
    "windowed":      "#e67e22",
    "baseline_win":  "#2980b9"
}
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
    "figure.dpi":        150,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe(d, *keys, default=np.nan):
    """Safely traverse nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, None)
        if d is None:
            return default
    return d if d is not None else default

def sort_w_labels(labels):
    """Sort window size labels numerically ascending (e.g. W1%, W5%, W10%)."""
    def _parse_w(val):
        m = re.search(r"(\d+)", str(val))
        return int(m.group(1)) if m else 0
    return sorted(labels, key=_parse_w)

def path_to_config_name(data: dict, is_time: bool) -> str:
    """Derive a config label from parameters block."""
    params = data.get("parameters", {})
    wf = params.get("window_size", 0) / max(params.get("total_events", 1), 1)
    wl = f"W{round(wf*100)}%"
    l  = params.get("max_trace_events", "?")
    if is_time:
        schedule = params.get("publish_interval", "unknown")
        return f"{schedule} {wl} L={l}"
    w  = params.get("window_size", 1)
    p  = params.get("publish_period", 1)
    if p == 0: p = 1
    r  = round(w / p)
    return f"{wl} r={r} L={l}"

def extract_w_label(config_name, is_time):
    if is_time:
        return "Unknown"
    m = re.match(r"(W\d+%)", config_name)
    return m.group(1) if m else "Unknown"

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def extract_rows(data: dict, log_name: str, is_time: bool, filename: str = "") -> list[dict]:
    """Flatten one result JSON into a list of per-publication dicts."""
    params = data.get("parameters", {})
    total_events = params.get("total_events", np.nan)
    config_name  = path_to_config_name(data, is_time)
    schedule = params.get("publish_interval", "unknown") if is_time else None
    
    if is_time and schedule not in ["1d", "5d", "10d"]:
        return []

    w_frac = params.get("window_size", 0) / max(params.get("total_events", 1), 1)
    w_label = f"W{round(w_frac*100)}%"
    
    m_lq = re.search(r"(LQ\d+)", filename)
    if m_lq:
        l_label = f"L={m_lq.group(1)}"
    else:
        l_label = f"L={params.get('max_trace_events', '?')}"

    rows = []
    for pub in data.get("publications", []):
        ep = pub.get("events_processed", np.nan)
        row = {
            "log":            log_name,
            "config":         config_name,
            "schedule":       schedule,
            "is_time":        is_time,
            "w_label":        w_label,
            "l_label":        l_label,
            "pub_index":      pub.get("publication_index", np.nan),
            "events_processed": ep,
            "pct_processed":  ep / total_events if total_events else np.nan,
            # budget
            "eps_spent":      safe(pub, "epsilon_budget", "epsilon_spent_total"),
            "eps_this":       safe(pub, "epsilon_budget", "epsilon_this_pub"),
            "laplace_scale":  safe(pub, "noise_cost", "theoretical_laplace_scale"),
            "noise_mae":      safe(pub, "noise_cost", "MAE"),
            "mre":            safe(pub, "noise_cost", "MRE"),
            
            # DFG structural F1 (0-100)
            "dfg_f1_windowed":    safe(pub, "dfg_metrics_x_vs_oracle", "noisy_dfg_kirchhoff", "f1"),
            "dfg_f1_raw_accum":   safe(pub, "dfg_metrics_x_vs_oracle", "raw_accumulated_dfg", "f1"),
            "dfg_f1_filtered_sa_accum": safe(pub, "dfg_metrics_x_vs_oracle", "filtered_sa_accumulated_dfg", "f1"),
            
            # Model quality – windowed (0-1)
            "f1_noisy_win_heur":  safe(pub, "quality_noisy_window_dfg", "windowed_log", "heuristic", "F1"),
            "f1_noisy_win_ind":   safe(pub, "quality_noisy_window_dfg", "windowed_log", "inductive", "F1"),
            "f1_baseline_win_heur": safe(pub, "baseline_offline_windowed", "windowed_log", "heuristic", "F1"),
            "f1_baseline_win_ind":  safe(pub, "baseline_offline_windowed", "windowed_log", "inductive", "F1"),
            
            # Model quality – accumulating (0-1)
            "f1_raw_accum_heur":  safe(pub, "quality_raw_accumulated_dfg", "seen_log", "heuristic", "F1"),
            "f1_raw_accum_ind":   safe(pub, "quality_raw_accumulated_dfg", "seen_log", "inductive", "F1"),
            
            # Model quality – clean / filtered sa accumulating (0-1)
            "f1_clean_accum_heur": safe(pub, "quality_accumulated_dfg", "seen_log", "heuristic", "F1"),
            "f1_clean_accum_ind":  safe(pub, "quality_accumulated_dfg", "seen_log", "inductive", "F1"),
            "f1_filtered_sa_accum_heur": safe(pub, "quality_accumulated_dfg", "full_log_filtered_sa", "heuristic", "F1"),
            "f1_filtered_sa_accum_ind":  safe(pub, "quality_accumulated_dfg", "full_log_filtered_sa", "inductive", "F1"),
            
            # Baselines for accumulating & offline baseline (0-1)
            "f1_baseline_accum_heur": safe(pub, "baseline_accumulating", "seen_log", "heuristic", "F1"),
            "f1_baseline_accum_ind":  safe(pub, "baseline_accumulating", "seen_log", "inductive", "F1"),
            "f1_offline_base_heur":   safe(pub, "baseline_oracle", "seen_log", "heuristic", "F1"),
            "f1_offline_base_ind":    safe(pub, "baseline_oracle", "seen_log", "inductive", "F1"),
            
            # Generality & Simplicity (0-1)
            "gen_raw_accum_heur":   safe(pub, "quality_raw_accumulated_dfg", "seen_log", "heuristic", "generalization"),
            "gen_raw_accum_ind":    safe(pub, "quality_raw_accumulated_dfg", "seen_log", "inductive", "generalization"),
            "simp_raw_accum_heur":  safe(pub, "quality_raw_accumulated_dfg", "seen_log", "heuristic", "simplicity"),
            "simp_raw_accum_ind":   safe(pub, "quality_raw_accumulated_dfg", "seen_log", "inductive", "simplicity"),
            
            "gen_filtered_sa_accum_heur":   safe(pub, "quality_accumulated_dfg", "full_log_filtered_sa", "heuristic", "generalization"),
            "gen_filtered_sa_accum_ind":    safe(pub, "quality_accumulated_dfg", "full_log_filtered_sa", "inductive", "generalization"),
            "simp_filtered_sa_accum_heur":  safe(pub, "quality_accumulated_dfg", "full_log_filtered_sa", "heuristic", "simplicity"),
            "simp_filtered_sa_accum_ind":   safe(pub, "quality_accumulated_dfg", "full_log_filtered_sa", "inductive", "simplicity"),
            
            "gen_offline_base_heur":   safe(pub, "baseline_oracle", "seen_log", "heuristic", "generalization"),
            "gen_offline_base_ind":    safe(pub, "baseline_oracle", "seen_log", "inductive", "generalization"),
            "simp_offline_base_heur":  safe(pub, "baseline_oracle", "seen_log", "heuristic", "simplicity"),
            "simp_offline_base_ind":   safe(pub, "baseline_oracle", "seen_log", "inductive", "simplicity"),
            
            # MRE vs offline baseline for accumulating DFGs
            "mre_raw_accum":          safe(pub, "dfg_metrics_x_vs_oracle", "raw_accumulated_dfg", "MRE"),
            "mre_accum_vs_offline_base":    safe(pub, "dfg_metrics_x_vs_oracle", "clean_accumulated_dfg", "MRE"),
            "mre_windowed_vs_offline_base": safe(pub, "dfg_metrics_x_vs_oracle", "noisy_dfg_kirchhoff", "MRE"),
        }
        rows.append(row)
    return rows

def load_all(output_root: Path, skip: set) -> pd.DataFrame:
    rows = []
    for log_dir in sorted(output_root.iterdir()):
        if not log_dir.is_dir() or log_dir.name in skip:
            continue
        log_name = log_dir.name
        for jf in sorted(log_dir.glob("*.json")):
            if jf.name.startswith("_"):
                continue
            data = load_json(jf)
            rows.extend(extract_rows(data, log_name, is_time=False, filename=jf.name))
        tb_dir = log_dir / "time_based"
        if tb_dir.is_dir():
            for jf in sorted(tb_dir.glob("*.json")):
                data = load_json(jf)
                rows.extend(extract_rows(data, log_name, is_time=True, filename=jf.name))
    return pd.DataFrame(rows)

def savefig(fig, out_dir: Path, name: str):
    dirname = out_dir.name
    suffix = ""
    
    if dirname.startswith("comparison_grid"):
        is_time = "time_based" in dirname
        suffix = "_tb" if is_time else "_ecb"
    else:
        if dirname.endswith("_timebased"):
            log_name = dirname.replace("_timebased", "")
            is_time = True
        else:
            log_name = dirname
            is_time = False
        suffix = f"_{log_name}_" + ("tb" if is_time else "ecb")
        
    miner_str = ""
    if "_heur" in name:
        name = name.replace("_heur", "")
        miner_str = "_HM"
    elif "_ind" in name:
        name = name.replace("_ind", "")
        miner_str = "_IM"
        
    final_name = f"{name}{suffix}{miner_str}"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{final_name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{final_name}.png", bbox_inches="tight")
    plt.close(fig)

def get_best_config(df: pd.DataFrame) -> str:
    """Find the best config based on overall F1 performance (Heuristic & Inductive) and non-null Inductive score availability across the stream."""
    if df.empty or "config" not in df.columns:
        return None
        
    configs = df["config"].unique()
    best_cfg = None
    best_score = -float("inf")
    
    for cfg in configs:
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
            
        ind_acc = sub["f1_filtered_sa_accum_ind"].dropna() if "f1_filtered_sa_accum_ind" in sub.columns else pd.Series(dtype=float)
        heur_acc = sub["f1_filtered_sa_accum_heur"].dropna() if "f1_filtered_sa_accum_heur" in sub.columns else pd.Series(dtype=float)
        ind_win = sub["f1_noisy_win_ind"].dropna() if "f1_noisy_win_ind" in sub.columns else pd.Series(dtype=float)
        heur_win = sub["f1_noisy_win_heur"].dropna() if "f1_noisy_win_heur" in sub.columns else pd.Series(dtype=float)
        
        valid_ind_ratio = len(ind_acc) / max(len(sub), 1)
        
        mean_ind_acc  = ind_acc.mean()  if len(ind_acc) > 0 else 0
        mean_heur_acc = heur_acc.mean() if len(heur_acc) > 0 else 0
        mean_ind_win  = ind_win.mean()  if len(ind_win) > 0 else 0
        mean_heur_win = heur_win.mean() if len(heur_win) > 0 else 0
        
        # Combined score: penalize missing inductive F1, reward high overall F1 across stream
        score = (valid_ind_ratio * 3.0) + (mean_ind_acc * 2.0) + (mean_heur_acc * 1.5) + (mean_ind_win * 1.0) + (mean_heur_win * 0.5)
        
        if score > best_score:
            best_score = score
            best_cfg = cfg
            
    return best_cfg

# ---------------------------------------------------------------------------
# Plot 1: Windowed Private Binary DFG vs. Windowed Baseline
# ---------------------------------------------------------------------------
def plot_windowed_comparison(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "f1_noisy_win_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_noisy_win_heur"], 
                color="#e74c3c", label="DP Windowed Heuristic F1", marker="o")
    if "f1_noisy_win_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_noisy_win_ind"], 
                color="#c0392b", label="DP Windowed Inductive F1", marker="s")
                
    if "f1_baseline_win_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_baseline_win_heur"], 
                color="#3498db", linestyle="--", label="Baseline Windowed Heuristic F1")
    if "f1_baseline_win_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_baseline_win_ind"], 
                color="#2980b9", linestyle="--", label="Baseline Windowed Inductive F1")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"Windowed Model Quality ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '01_windowed_vs_baseline')

# ---------------------------------------------------------------------------
# Plot 1b: Difference of Baseline & Private Window F1 Score
# ---------------------------------------------------------------------------
def plot_windowed_f1_difference(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "f1_baseline_win_heur" in sub.columns and "f1_noisy_win_heur" in sub.columns:
        diff_heur = sub["f1_baseline_win_heur"] - sub["f1_noisy_win_heur"]
        ax.plot(sub["pct_processed"], diff_heur, 
                color="#e74c3c", label="Δ Windowed Heuristic F1 (Baseline − DP)", marker="o")
    if "f1_baseline_win_ind" in sub.columns and "f1_noisy_win_ind" in sub.columns:
        diff_ind = sub["f1_baseline_win_ind"] - sub["f1_noisy_win_ind"]
        ax.plot(sub["pct_processed"], diff_ind, 
                color="#2980b9", label="Δ Windowed Inductive F1 (Baseline − DP)", marker="s")

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("F1 Score Difference (Baseline − DP)")
    ax.set_title(f"Windowed F1 Score Difference ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '01b_windowed_f1_difference')

# ---------------------------------------------------------------------------
# Plot 2: Accumulating DFG vs. Offline Baseline
# ---------------------------------------------------------------------------
def plot_accumulating_comparison(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "f1_raw_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_raw_accum_heur"], 
                color="#f39c12", label="DP Accum. Heuristic F1", marker="o")
    if "f1_raw_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_raw_accum_ind"], 
                color="#d35400", label="DP Accum. Inductive F1", marker="s")
                
    # offline_base constants
    off_h = "f1_offline_base_heur" if "f1_offline_base_heur" in sub.columns else "f1_oracle_heur"
    off_i = "f1_offline_base_ind" if "f1_offline_base_ind" in sub.columns else "f1_oracle_ind"
    
    if off_h in sub.columns and not sub[off_h].isna().all():
        val = sub[off_h].dropna().iloc[-1]
        ax.axhline(val, color="#9b59b6", linestyle=":", label="Offline Baseline Heuristic F1")
    if off_i in sub.columns and not sub[off_i].isna().all():
        val = sub[off_i].dropna().iloc[-1]
        ax.axhline(val, color="#8e44ad", linestyle=":", label="Offline Baseline Inductive F1")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"Accumulating vs Offline Baseline ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '02_accumulating_vs_baseline')

# ---------------------------------------------------------------------------
# Plot 2b: Difference of Offline Baseline & Accumulating F1 Score
# ---------------------------------------------------------------------------
def plot_accumulating_f1_difference(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    off_h = "f1_offline_base_heur" if "f1_offline_base_heur" in sub.columns else "f1_oracle_heur"
    off_i = "f1_offline_base_ind" if "f1_offline_base_ind" in sub.columns else "f1_oracle_ind"

    if off_h in sub.columns and "f1_filtered_sa_accum_heur" in sub.columns:
        diff = sub[off_h] - sub["f1_filtered_sa_accum_heur"]
        ax.plot(sub["pct_processed"], diff, 
                color="#2ecc71", label="Δ Clean Accum. Heuristic F1 (Offline Base − DP)", marker="o")
    if off_i in sub.columns and "f1_filtered_sa_accum_ind" in sub.columns:
        diff = sub[off_i] - sub["f1_filtered_sa_accum_ind"]
        ax.plot(sub["pct_processed"], diff, 
                color="#27ae60", label="Δ Clean Accum. Inductive F1 (Offline Base − DP)", marker="s")

    if off_h in sub.columns and "f1_raw_accum_heur" in sub.columns:
        diff = sub[off_h] - sub["f1_raw_accum_heur"]
        ax.plot(sub["pct_processed"], diff, 
                color="#f39c12", label="Δ Accum. Heuristic F1 (Offline Base − DP)", marker="^", linestyle="--")
    if off_i in sub.columns and "f1_raw_accum_ind" in sub.columns:
        diff = sub[off_i] - sub["f1_raw_accum_ind"]
        ax.plot(sub["pct_processed"], diff, 
                color="#d35400", label="Δ Accum. Inductive F1 (Offline Base − DP)", marker="d", linestyle="--")

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("F1 Score Difference (Offline Base − DP)")
    ax.set_title(f"Accumulating F1 Score Difference ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '02b_accumulating_f1_difference')

# ---------------------------------------------------------------------------
# Plot 2c: Accumulating DFG vs. Accumulating Peer Baseline
# ---------------------------------------------------------------------------
def plot_accumulating_peer_comparison(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "f1_filtered_sa_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_filtered_sa_accum_heur"], 
                color="#2ecc71", label="DP Clean Accum. Heuristic F1", marker="o")
    if "f1_filtered_sa_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_filtered_sa_accum_ind"], 
                color="#27ae60", label="DP Clean Accum. Inductive F1", marker="s")

    if "f1_baseline_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_baseline_accum_heur"], 
                color="#3498db", linestyle="--", label="Baseline Accumulating Heuristic F1")
    if "f1_baseline_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_baseline_accum_ind"], 
                color="#2980b9", linestyle="--", label="Baseline Accumulating Inductive F1")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"Accumulating vs Peer Baseline ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '02c_accumulating_vs_peer_baseline')

# ---------------------------------------------------------------------------
# Plot 2d: Difference of Accumulating Peer Baseline & DP Accumulating F1 Score
# ---------------------------------------------------------------------------
def plot_accumulating_peer_f1_difference(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "f1_baseline_accum_heur" in sub.columns and "f1_filtered_sa_accum_heur" in sub.columns:
        diff_heur = sub["f1_baseline_accum_heur"] - sub["f1_filtered_sa_accum_heur"]
        ax.plot(sub["pct_processed"], diff_heur, 
                color="#2ecc71", label="Δ Clean Accum. Heuristic F1 (Peer Baseline − DP)", marker="o")
    if "f1_baseline_accum_ind" in sub.columns and "f1_filtered_sa_accum_ind" in sub.columns:
        diff_ind = sub["f1_baseline_accum_ind"] - sub["f1_filtered_sa_accum_ind"]
        ax.plot(sub["pct_processed"], diff_ind, 
                color="#27ae60", label="Δ Clean Accum. Inductive F1 (Peer Baseline − DP)", marker="s")

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("F1 Score Difference (Peer Baseline − DP)")
    ax.set_title(f"Accumulating Peer F1 Difference ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '02d_accumulating_peer_f1_difference')

# ---------------------------------------------------------------------------
# Plot 3: Filtered SA Accumulating DFG vs. Full Baseline + Offline Baseline
# ---------------------------------------------------------------------------
def plot_clean_accumulating_comparison(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "f1_clean_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_clean_accum_heur"], 
                color="#2ecc71", label="DP Clean Accum. Heuristic F1", marker="o")
    if "f1_clean_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["f1_clean_accum_ind"], 
                color="#27ae60", label="DP Clean Accum. Inductive F1", marker="s")
                
    off_h = "f1_offline_base_heur" if "f1_offline_base_heur" in sub.columns else "f1_oracle_heur"
    off_i = "f1_offline_base_ind" if "f1_offline_base_ind" in sub.columns else "f1_oracle_ind"

    if off_h in sub.columns and not sub[off_h].isna().all():
        val = sub[off_h].dropna().iloc[-1]
        ax.axhline(val, color="#9b59b6", linestyle=":", label="Offline Baseline Heuristic F1")
    if off_i in sub.columns and not sub[off_i].isna().all():
        val = sub[off_i].dropna().iloc[-1]
        ax.axhline(val, color="#8e44ad", linestyle=":", label="Offline Baseline Inductive F1")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"Clean Accumulating vs Offline Baseline ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '03_clean_accumulating_vs_baseline')

# ---------------------------------------------------------------------------
# Plot 4: DFG F1 Scores over Stream (Windowed, Accumulating, Filtered SA Accumulating)
# ---------------------------------------------------------------------------
def plot_dfg_f1_scores(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "dfg_f1_windowed" in sub.columns:
        ax.plot(sub["pct_processed"], (sub["dfg_f1_windowed"] / 100.0), 
                color="#e74c3c", label="DP Windowed DFG F1", marker="x")
    if "dfg_f1_raw_accum" in sub.columns:
        ax.plot(sub["pct_processed"], (sub["dfg_f1_raw_accum"] / 100.0), 
                color="#f39c12", label="DP Accum. DFG F1", marker="^")
    if "dfg_f1_filtered_sa_accum" in sub.columns:
        ax.plot(sub["pct_processed"], (sub["dfg_f1_filtered_sa_accum"] / 100.0), 
                color="#2ecc71", label="DP Filtered SA Accum. DFG F1", marker="o")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("DFG Structural F1 Score (%)")
    ax.set_title(f"Structural DFG F1 Scores ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '04_dfg_f1_scores')

# ---------------------------------------------------------------------------
# Plot 5: Mean Relative Error (MRE) over Stream
# ---------------------------------------------------------------------------
def plot_mre(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "mre" in sub.columns:
        ax.plot(sub["pct_processed"], sub["mre"], 
                color="#8e44ad", label="Mean Relative Error (MRE)", marker="d")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("MRE (Noise Cost)")
    ax.set_title(f"Noise Cost MRE over Time ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '05_mre')

# ---------------------------------------------------------------------------
# Plot 6: Budget Decay vs Noise Scale
# ---------------------------------------------------------------------------
def plot_budget_decay(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    max_pub = df.groupby("config")["pub_index"].max()
    multi_configs = max_pub[max_pub > 0].index
    if len(multi_configs) == 0:
        return
        
    multi = df[df["config"].isin(multi_configs)]
    if not best_cfg:
        best_cfg = get_best_config(multi)
    if not best_cfg: return
    
    sub = multi[multi["config"] == best_cfg].sort_values("pub_index")
    if len(sub) < 2: return

    is_local_ax = ax is None
    if is_local_ax:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
    else:
        ax1 = ax
    ax2 = ax1.twinx()

    color_eps = "#2980b9"   # blue
    color_lap = "#e74c3c"   # red

    line1 = ax1.plot(sub["pub_index"], sub["eps_spent"],
             color=color_eps, linestyle="-", linewidth=2, marker="o", markersize=4, label="ε spent (cumulative)")
    
    mre_acc_col = "mre_accum_vs_offline_base" if "mre_accum_vs_offline_base" in sub.columns else "mre_accum_vs_oracle"
    mre_win_col = "mre_windowed_vs_offline_base" if "mre_windowed_vs_offline_base" in sub.columns else "mre_windowed_vs_oracle"

    line2 = ax2.plot(sub["pub_index"], sub[mre_acc_col],
             color=color_lap, linestyle="--", linewidth=2, marker="s", markersize=4, label="MRE accum. DFG vs offline baseline")
    line3 = ax2.plot(sub["pub_index"], sub[mre_win_col],
             color="#e67e22", linestyle=":", linewidth=2, marker="^", markersize=4, label="MRE windowed DFG vs offline baseline")

    ax1.set_xlabel("Publication index")
    ax1.set_ylabel("ε spent", color=color_eps, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_eps)
    
    ax2.set_ylabel("MRE vs offline baseline", color=color_lap, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_lap)
    
    ax1.set_title(f"Budget Decay & Noise ({best_cfg}) — {log_name}", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '06_budget_decay')

# ---------------------------------------------------------------------------
# Plot 7: Privacy–Utility Heatmap (Event-based & Time-based)
# ---------------------------------------------------------------------------
def _parse_w_r(config_str: str):
    """Extract (W%, r) from event-based config strings like 'W5% r=2 L=10'."""
    mw = re.search(r"W(\d+)%", config_str)
    mr = re.search(r"r=(\d+)", config_str)
    if mw and mr:
        return f"W{mw.group(1)}%", int(mr.group(1))
    return None, None

def plot_heatmap(df: pd.DataFrame, log_name: str, out_dir: Path, is_time: bool, ax=None, best_cfg: str = None, miner_type: str = "heur"):
    """Final publication F1 heatmap over W × r or W × schedule (Heuristic or Inductive)."""
    final = df.sort_values("pub_index").groupby("config", as_index=False).last()
    if final.empty:
        return
        
    metric_col = f"f1_filtered_sa_accum_{miner_type}"
    if metric_col not in final.columns:
        return

    if is_time:
        piv = final.groupby(["w_label", "schedule"])[metric_col].mean().reset_index()
        if piv.empty: return
        piv = piv.pivot(index="schedule", columns="w_label", values=metric_col)
        sched_order = ["1d", "5d", "10d"]
        piv = piv.reindex([s for s in sched_order if s in piv.index])
        y_label = "Schedule"
        y_ticks = list(piv.index)
    else:
        parsed = final["config"].apply(lambda c: pd.Series(_parse_w_r(c), index=["w_label_parsed","r_val"]))
        final = pd.concat([final, parsed], axis=1)
        final = final.dropna(subset=["w_label", "r_val"])
        if final.empty: return

        piv = final.groupby(["w_label", "r_val"])[metric_col].mean().reset_index()
        piv = piv.pivot(index="r_val", columns="w_label", values=metric_col)
        piv = piv.sort_index()
        y_label = "Publications per window (r)"
        y_ticks = [f"r={int(v)}" for v in piv.index]

    if piv.empty: return
    
    # Sort W% columns numerically (1%, 5%, 10%)
    sorted_cols = sort_w_labels(piv.columns)
    piv = piv.reindex(columns=sorted_cols)

    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

    im = ax.imshow(piv.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, label=f"Final Filtered SA Accum-DFG {miner_type.capitalize()} F1")

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("Window size (% of log)")
    ax.set_ylabel(y_label)
    mode_title = "Time-based" if is_time else "Event-based"
    miner_title = "Heuristic Miner" if miner_type == "heur" else "Inductive Miner"
    ax.set_title(f"Model Quality Heatmap ({miner_title}, {mode_title}) — {log_name}", fontweight="bold")

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.65 else "black", fontsize=9)

    # Reference box for Offline Baseline
    off_col = f"f1_offline_base_{miner_type}"
    if off_col in final.columns and not final[off_col].isna().all():
        off_ref = final[off_col].dropna().iloc[-1]
        ax.text(0.98, 0.98, f"Off. Base: {off_ref:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#9b59b6", alpha=0.85, edgecolor="none"),
                color="white", fontsize=8, fontweight="bold")

    if is_local_ax:
        plt.tight_layout()
        fname = f"07_privacy_utility_heatmap_{miner_type}"
        savefig(fig, out_dir, fname)

# ---------------------------------------------------------------------------
# Plot 11: Faceted Privacy-Utility Heatmap (L, W, r/schedule)
# ---------------------------------------------------------------------------
def plot_faceted_heatmap(df: pd.DataFrame, log_name: str, out_dir: Path, is_time: bool, miner_type: str = "heur"):
    """Faceted Heatmap per trace event limit L (Heuristic or Inductive).

    All grid cells are guaranteed to be filled: if the last publication for a
    given (config, L) combination has a null / zero metric value, the function
    falls back to the last finite non-zero value in that config's history.
    Cells where the fallback was used are marked with a small red triangle so
    the reader knows the displayed value is not from the actual final publication.
    """
    metric_col = f"f1_filtered_sa_accum_{miner_type}"
    if metric_col not in df.columns:
        return

    sorted_df = df.sort_values("pub_index")

    # ------------------------------------------------------------------
    # For each config find the best "last valid" value:
    #   - If the true last publication has a finite, non-zero value → use it,
    #     no flag needed.
    #   - Otherwise fall back to the last publication in the history that had a
    #     valid value, and set _flag = True so a red marker is drawn.
    # ------------------------------------------------------------------
    def _last_valid_row(group):
        last_row = group.iloc[-1].copy()
        last_val = last_row.get(metric_col, float("nan"))
        last_is_valid = np.isfinite(float(last_val)) and float(last_val) > 0
        if last_is_valid:
            last_row["_flag"] = False
            return last_row
        valid = group[group[metric_col].apply(lambda v: np.isfinite(float(v)) and float(v) > 0)]
        if valid.empty:
            last_row["_flag"] = False  # nothing to fall back to, cell stays NaN
            return last_row
        row = valid.iloc[-1].copy()
        row["_flag"] = True
        return row

    final = sorted_df.groupby("config", group_keys=False).apply(_last_valid_row).reset_index(drop=True)
    if final.empty or "config" not in final.columns:
        return

    if is_time:
        final["y_val"] = final["schedule"]
        y_label = "Schedule"
    else:
        parsed = final["config"].apply(lambda c: pd.Series(_parse_w_r(c), index=["w_label_parsed", "r_val"]))
        final = pd.concat([final, parsed], axis=1)
        final = final.dropna(subset=["w_label", "r_val", "l_label"])
        if final.empty:
            return
        final["y_val"] = final["r_val"]
        y_label = "Publications per window (r)"

    def _sort_l(x):
        try:
            return int(re.search(r"(\d+)", x).group(1))
        except:
            return 0

    unique_L = sorted(final["l_label"].unique(), key=_sort_l)
    if not unique_L:
        return

    if is_time:
        sched_order = ["1d", "5d", "10d"]
        unique_y = [s for s in sched_order if s in final["y_val"].unique()]
        y_ticks = unique_y
    else:
        unique_y = sorted(final["y_val"].unique())
        y_ticks = [f"r={int(v)}" for v in unique_y]

    unique_w = sort_w_labels(final["w_label"].unique())

    fig, axes = plt.subplots(1, len(unique_L), figsize=(5 * len(unique_L), 4), sharey=True)
    if len(unique_L) == 1:
        axes = [axes]

    im = None
    off_col = f"f1_offline_base_{miner_type}"
    off_ref = (
        final[off_col].dropna().iloc[-1]
        if (off_col in final.columns and not final[off_col].isna().all())
        else None
    )

    any_flag = False  # track whether we need the legend

    for i, l_val in enumerate(unique_L):
        ax = axes[i]
        sub = final[final["l_label"] == l_val]

        # Value pivot
        piv = sub.groupby(["w_label", "y_val"])[metric_col].mean().reset_index()
        if piv.empty:
            continue
        piv = piv.pivot(index="y_val", columns="w_label", values=metric_col)
        piv = piv.reindex(index=unique_y, columns=unique_w)

        # Flag pivot (True = fallback was used)
        flag_piv = sub.groupby(["w_label", "y_val"])["_flag"].any().reset_index()
        flag_piv = flag_piv.pivot(index="y_val", columns="w_label", values="_flag")
        flag_piv = flag_piv.reindex(index=unique_y, columns=unique_w).fillna(False)

        im = ax.imshow(piv.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1.0)

        ax.set_xticks(range(len(unique_w)))
        ax.set_xticklabels(unique_w)
        if i == 0:
            ax.set_yticks(range(len(unique_y)))
            ax.set_yticklabels(y_ticks)
            ax.set_ylabel(y_label)

        ax.set_xlabel("Window size (% of log)")
        ax.set_title(f"Max Trace Events: {l_val}")

        for r_idx in range(piv.shape[0]):
            for c_idx in range(piv.shape[1]):
                val = piv.values[r_idx, c_idx]
                is_flagged = bool(flag_piv.values[r_idx, c_idx])
                if not np.isnan(val):
                    ax.text(
                        c_idx, r_idx, f"{val:.2f}",
                        ha="center", va="center",
                        color="white" if val > 0.65 else "black",
                        fontsize=9,
                    )
                    if is_flagged:
                        any_flag = True
                        # Small red downward triangle in the top-right corner of the cell
                        ax.plot(
                            c_idx + 0.38, r_idx - 0.38,
                            marker="v", color="#e53935", ms=6,
                            zorder=5, clip_on=False,
                        )
                else:
                    # No valid model ever produced for this config
                    ax.text(
                        c_idx, r_idx, "no\nmodel",
                        ha="center", va="center",
                        color="#e53935",
                        fontsize=8,
                        fontweight="bold"
                    )

        if off_ref is not None:
            ax.text(
                0.98, 0.98, f"Off. Base: {off_ref:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#9b59b6",
                          alpha=0.85, edgecolor="none"),
                color="white", fontsize=8, fontweight="bold",
            )

    if im is not None:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax,
                     label=f"Final Filtered SA Accum-DFG {miner_type.capitalize()} F1")

    # Add legend for the red flag only if at least one cell used the fallback
    if any_flag:
        from matplotlib.lines import Line2D
        flag_handle = Line2D(
            [0], [0], marker="v", color="w", markerfacecolor="#e53935",
            markersize=7, label="Last valid value (not final pub.)", linestyle="None",
        )
        axes[-1].legend(handles=[flag_handle], loc="lower right", fontsize=7,
                        framealpha=0.8, borderpad=0.4)

    mode_title = "Time-based" if is_time else "Event-based"
    miner_title = "Heuristic Miner" if miner_type == "heur" else "Inductive Miner"
    fig.suptitle(
        f"Faceted Model Quality Heatmap ({miner_title}, {mode_title}) — {log_name}",
        fontweight="bold", y=1.05,
    )

    fname = f"11_faceted_heatmap_{miner_type}"
    savefig(fig, out_dir, fname)

# ---------------------------------------------------------------------------
# Plot 8: Generality Comparison
# ---------------------------------------------------------------------------
def plot_generality_comparison(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "gen_raw_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["gen_raw_accum_ind"], 
                color="#e67e22", label="Accum. Generality (Ind)", marker="o")
    if "gen_raw_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["gen_raw_accum_heur"], 
                color="#d35400", label="Accum. Generality (Heur)", marker="s", linestyle="--")
                
    if "gen_filtered_sa_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["gen_filtered_sa_accum_ind"], 
                color="#2ecc71", label="Filtered SA Accum. Generality (Ind)", marker="o")
    if "gen_filtered_sa_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["gen_filtered_sa_accum_heur"], 
                color="#27ae60", label="Filtered SA Accum. Generality (Heur)", marker="s", linestyle="--")

    off_gi = "gen_offline_base_ind" if "gen_offline_base_ind" in sub.columns else "gen_oracle_ind"
    off_gh = "gen_offline_base_heur" if "gen_offline_base_heur" in sub.columns else "gen_oracle_heur"

    if off_gi in sub.columns and not sub[off_gi].isna().all():
        val = sub[off_gi].dropna().iloc[-1]
        ax.axhline(val, color="#9b59b6", linestyle="-", label="Offline Baseline Generality (Ind)")
    if off_gh in sub.columns and not sub[off_gh].isna().all():
        val = sub[off_gh].dropna().iloc[-1]
        ax.axhline(val, color="#8e44ad", linestyle="--", label="Offline Baseline Generality (Heur)")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("Generality")
    ax.set_title(f"Generality ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '08_generality')

# ---------------------------------------------------------------------------
# Plot 9: Simplicity Comparison
# ---------------------------------------------------------------------------
def plot_simplicity_comparison(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "simp_raw_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["simp_raw_accum_ind"], 
                color="#e67e22", label="Accum. Simplicity (Ind)", marker="x")
    if "simp_raw_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["simp_raw_accum_heur"], 
                color="#d35400", label="Accum. Simplicity (Heur)", marker="+", linestyle="--")
                
    if "simp_filtered_sa_accum_ind" in sub.columns:
        ax.plot(sub["pct_processed"], sub["simp_filtered_sa_accum_ind"], 
                color="#2ecc71", label="Filtered SA Accum. Simplicity (Ind)", marker="x")
    if "simp_filtered_sa_accum_heur" in sub.columns:
        ax.plot(sub["pct_processed"], sub["simp_filtered_sa_accum_heur"], 
                color="#27ae60", label="Filtered SA Accum. Simplicity (Heur)", marker="+", linestyle="--")

    off_si = "simp_offline_base_ind" if "simp_offline_base_ind" in sub.columns else "simp_oracle_ind"
    off_sh = "simp_offline_base_heur" if "simp_offline_base_heur" in sub.columns else "simp_oracle_heur"

    if off_si in sub.columns and not sub[off_si].isna().all():
        val = sub[off_si].dropna().iloc[-1]
        ax.axhline(val, color="#9b59b6", linestyle="-", label="Offline Baseline Simplicity (Ind)")
    if off_sh in sub.columns and not sub[off_sh].isna().all():
        val = sub[off_sh].dropna().iloc[-1]
        ax.axhline(val, color="#8e44ad", linestyle="--", label="Offline Baseline Simplicity (Heur)")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("Simplicity")
    ax.set_title(f"Simplicity ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '09_simplicity')

# ---------------------------------------------------------------------------
# Plot 10: MRE Comparison (Windowed vs Accumulating)
# ---------------------------------------------------------------------------
def plot_mre_comparison(df: pd.DataFrame, log_name: str, out_dir: Path, ax=None, best_cfg: str = None):
    if not best_cfg:
        best_cfg = get_best_config(df)
    if not best_cfg: return
    sub = df[df["config"] == best_cfg].sort_values("pct_processed")
    
    is_local_ax = ax is None
    if is_local_ax:
        fig, ax = plt.subplots(figsize=(8, 5))

    if "mre" in sub.columns:
        ax.plot(sub["pct_processed"], sub["mre"], 
                color="#e74c3c", label="Windowed MRE (Noise Cost)", marker="d")
    if "mre_raw_accum" in sub.columns:
        ax.plot(sub["pct_processed"], sub["mre_raw_accum"], 
                color="#f39c12", label="Accum. MRE", marker="o")

    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("Mean Relative Error (MRE)")
    ax.set_title(f"MRE Comparison ({best_cfg}) — {log_name}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    if is_local_ax:
        plt.tight_layout()
        savefig(fig, out_dir, '10_mre_comparison')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_specific_files(file_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in file_paths:
        path = path.resolve()
        if not path.is_file() or not path.name.endswith(".json"):
            continue
        data = load_json(path)
        is_time = "time_based" in path.parts
        if is_time:
            log_name = path.parent.parent.name
        else:
            log_name = path.parent.name
        rows.extend(extract_rows(data, log_name, is_time=is_time))
    return pd.DataFrame(rows)


def make_grid_plots(df: pd.DataFrame, is_time: bool, out_dir: Path):
    logs = sorted(df["log"].unique())
    if not logs: return
    
    print(f"  → Generating grid comparisons to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plots_to_make = [
        ("01_windowed_vs_baseline", plot_windowed_comparison, {}),
        ("01b_windowed_f1_difference", plot_windowed_f1_difference, {}),
        ("02_accumulating_vs_baseline", plot_accumulating_comparison, {}),
        ("02b_accumulating_f1_difference", plot_accumulating_f1_difference, {}),
        ("02c_accumulating_vs_peer_baseline", plot_accumulating_peer_comparison, {}),
        ("02d_accumulating_peer_f1_difference", plot_accumulating_peer_f1_difference, {}),
        ("03_clean_accumulating_vs_baseline", plot_clean_accumulating_comparison, {}),
        ("04_dfg_f1_scores", plot_dfg_f1_scores, {}),
        ("05_mre", plot_mre, {}),
        ("06_budget_decay", plot_budget_decay, {}),
        ("07_privacy_utility_heatmap_heuristic", plot_heatmap, {"is_time": is_time, "miner_type": "heur"}),
        ("07_privacy_utility_heatmap_inductive", plot_heatmap, {"is_time": is_time, "miner_type": "ind"}),
        ("08_generality", plot_generality_comparison, {}),
        ("09_simplicity", plot_simplicity_comparison, {}),
        ("10_mre_comparison", plot_mre_comparison, {}),
    ]
    
    rows = int(np.ceil(len(logs) / 2))
    cols = min(2, len(logs))
    if rows == 0 or cols == 0: return

    for plot_name, plot_func, kwargs in plots_to_make:
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
        axes_flat = axes.flatten()
        
        for i, log_name in enumerate(logs):
            ax = axes_flat[i]
            r_idx = i // cols
            c_idx = i % cols
            
            df_log = df[df["log"] == log_name]
            if df_log.empty:
                ax.set_visible(False)
                continue
            
            best_cfg = get_best_config(df_log)
            plot_func(df_log, log_name, out_dir, ax=ax, best_cfg=best_cfg, **kwargs)
            
            # Keep log titles with parameter info (W, r/d, L)
            ax.set_title(f"{log_name} ({best_cfg})", fontweight="bold")
            
            # Clean layout: Y-axis scale only on left column
            if c_idx > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
                # Handle twinx if present
                for other_ax in fig.axes:
                    if other_ax != ax and hasattr(other_ax, "get_position") and other_ax.get_position().bounds == ax.get_position().bounds:
                        if c_idx < cols - 1:
                            other_ax.set_ylabel("")
                            other_ax.tick_params(labelright=False)

            # Clean layout: X-axis scale only on bottom row
            if r_idx < rows - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

        for j in range(len(logs), len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Deduplicate legends into single figure legend
        all_handles, all_labels = [], []
        for a in fig.axes:
            handles, labels = a.get_legend_handles_labels()
            for h, label in zip(handles, labels):
                if label and label not in all_labels:
                    all_handles.append(h)
                    all_labels.append(label)
            leg = a.get_legend()
            if leg:
                leg.remove()
                
        if all_handles:
            fig.legend(all_handles, all_labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(len(all_labels), 4), fontsize=9)

        fig.tight_layout()
        savefig(fig, out_dir, plot_name)

def make_plots_for_mode(df_mode: pd.DataFrame, log_name: str, is_time: bool, out_dir: Path):
    print(f"  → {out_dir}")
    if df_mode.empty:
        print("     (no data, skipping)")
        return
    best_cfg = get_best_config(df_mode)
    plot_windowed_comparison(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_windowed_f1_difference(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_accumulating_comparison(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_accumulating_f1_difference(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_accumulating_peer_comparison(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_accumulating_peer_f1_difference(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_clean_accumulating_comparison(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_dfg_f1_scores(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_mre(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_budget_decay(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_heatmap(df_mode, log_name, out_dir, is_time, best_cfg=best_cfg, miner_type="heur")
    plot_heatmap(df_mode, log_name, out_dir, is_time, best_cfg=best_cfg, miner_type="ind")
    plot_faceted_heatmap(df_mode, log_name, out_dir, is_time, miner_type="heur")
    plot_faceted_heatmap(df_mode, log_name, out_dir, is_time, miner_type="ind")
    plot_generality_comparison(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_simplicity_comparison(df_mode, log_name, out_dir, best_cfg=best_cfg)
    plot_mre_comparison(df_mode, log_name, out_dir, best_cfg=best_cfg)

def main():
    parser = argparse.ArgumentParser(description="Visualize DP Streaming DFG results.")
    parser.add_argument("--files", nargs="+", type=Path, help="Specific JSON files to visualize.")
    args = parser.parse_args()

    print("Loading data …")
    if args.files:
        df = load_specific_files(args.files)
    else:
        df = load_all(OUTPUT_ROOT, SKIP_LOGS)
        
    if df.empty:
        print("No valid data found to plot. Exiting.")
        return
        
    print(f"  Loaded {len(df)} publication rows across {df['log'].nunique()} logs.")

    for log_name in sorted(df["log"].unique()):
        print(f"\n[{log_name}]")
        df_log = df[df["log"] == log_name]

        # Event-based
        make_plots_for_mode(
            df_log[~df_log["is_time"]],
            log_name,
            False,
            PLOTS_ROOT / log_name,
        )
        # Time-based
        make_plots_for_mode(
            df_log[df_log["is_time"]],
            log_name,
            True,
            PLOTS_ROOT / f"{log_name}_timebased",
        )

    print("\n[Grid Comparison - Event-based]")
    make_grid_plots(df[~df["is_time"]], is_time=False, out_dir=PLOTS_ROOT / "comparison_grid_event_based")
    
    print("\n[Grid Comparison - Time-based]")
    make_grid_plots(df[df["is_time"]], is_time=True, out_dir=PLOTS_ROOT / "comparison_grid_time_based")

    print("\nDone. Figures written to ./plots/")

if __name__ == "__main__":
    main()
