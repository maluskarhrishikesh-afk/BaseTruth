"""
Training Data Visualisation — BaseTruth ML Pipeline
====================================================

Generates a 3-panel PNG report that answers two questions:

  1. How was the training data split across the 4 forensic classes?
  2. Which forensic signals drove those class boundaries?

Run from the repo root:
    .venv\\Scripts\\python.exe fraud_model\\scripts\\visualize_training.py

Output file:
    fraud_model/training_summary.png

Panels
------
  A) Class Distribution  — sample counts per class (ORIGINAL … TAMPERED-DERIVED)
  B) Global Feature Importance — mean gain across all trees (sorted highest first)
  C) Per-Class SHAP Heatmap   — average |SHAP| per class × feature, showing which
                                 signals matter most for each verdict bucket.
"""

import sys
import os

# Allow imports from the repo src/ tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend so it runs headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

# Silence sklearn warnings about imputer feature names (we know what we're doing)
warnings.filterwarnings("ignore", category=UserWarning)

from basetruth.analysis.ml_scorer import FEATURE_NAMES, ML_VERDICT_LABELS
from basetruth.logger import get_logger

log = get_logger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CSV_PATH   = os.path.join(_ROOT, "fraud_model", "data",   "training_data_image.csv")
_MODEL_PATH = os.path.join(_ROOT, "fraud_model", "models", "ml_scorer_image.pkl")
_OUT_PATH         = os.path.join(_ROOT, "fraud_model", "training_summary.png")
_SCATTER_OUT_PATH = os.path.join(_ROOT, "fraud_model", "scatter_plot.png")
_TREE_OUT_PATH    = os.path.join(_ROOT, "fraud_model", "tree_diagram.png")

# ─── Brand colours — one per verdict class, matching the UI palette ──────────
#     ORIGINAL=green, ORIGINAL-DERIVED=blue, TAMPERED=red, TAMPERED-DERIVED=purple
_CLASS_COLORS = {
    0: "#22c55e",   # ORIGINAL         — green
    1: "#3b82f6",   # ORIGINAL-DERIVED — blue
    2: "#ef4444",   # TAMPERED         — red
    3: "#a855f7",   # TAMPERED-DERIVED — purple
}

# ─── Human-readable labels (same order as ML_VERDICT_LABELS) ─────────────────
_LABELS = [ML_VERDICT_LABELS[i] for i in range(4)]
_COLORS  = [_CLASS_COLORS[i]    for i in range(4)]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load training data and remap raw CSV columns to model feature names
# ─────────────────────────────────────────────────────────────────────────────

def _load_features() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the training CSV and return (feature_matrix, label_array).

    The raw CSV stores engine outputs under names like 'ela_mean', 'dct_comb_ratio',
    etc.  The _remap_raw_csv() helper in ml_scorer converts them to the canonical
    FEATURE_NAMES (19 raw signals) used by the model.
    """
    from basetruth.analysis.ml_scorer import _remap_raw_csv   # noqa: PLC0415

    log.info("Loading training CSV", extra={"path": _CSV_PATH})
    df = pd.read_csv(_CSV_PATH)
    labels = df["label"].values.astype(int)

    df_feat = _remap_raw_csv(df)
    X = df_feat[FEATURE_NAMES].copy().astype(float)

    log.info(
        "Training data loaded",
        extra={
            "samples": len(df),
            "features": X.shape[1],
            "class_counts": dict(zip(range(4), np.bincount(labels, minlength=4).tolist())),
        },
    )
    return X, labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Extract per-class SHAP values from the trained XGBoost booster
# ─────────────────────────────────────────────────────────────────────────────

def _compute_shap(X: pd.DataFrame, pipe) -> np.ndarray:
    """Compute mean |SHAP| per class using the XGBoost built-in tree SHAP.

    For a 4-class multiclass model, booster.predict(pred_contribs=True) returns
    a 3-D array of shape (n_samples, n_classes, n_features + 1).  The last column
    is the baseline (bias) value, which we discard.  We then average absolute SHAP
    across samples to get a (4, n_features) importance matrix.

    The booster may have been fitted on fewer features than the full FEATURE_NAMES
    list if a late-added feature never appeared in training data.  We trim X to the
    booster's expected feature count (n_booster) before building the DMatrix.
    """
    imputer = pipe.named_steps["imputer"]
    model   = pipe.named_steps["model"]
    booster = model.get_booster()

    # How many features the booster was actually trained on (≤ len(FEATURE_NAMES))
    n_booster = booster.num_features()
    # Map f0…f(n-1) positions to the matching FEATURE_NAMES entries
    active_names = FEATURE_NAMES[:n_booster]

    log.debug("SHAP computation", extra={"n_booster_features": n_booster})

    # Impute first, then trim to the booster's feature width
    X_imp = imputer.transform(X)                        # shape: (n, 11)
    X_trim = X_imp[:, :n_booster]                       # drop trailing features booster doesn't know
    dmat = xgb.DMatrix(X_trim, feature_names=active_names)

    # pred_contribs → shape (n_samples, n_classes, n_features + 1)
    raw = booster.predict(dmat, pred_contribs=True)
    log.debug("SHAP raw shape", extra={"shape": list(raw.shape)})

    # Drop the bias column (last) → shape (n, n_classes, n_features)
    shap_vals = raw[:, :, :-1]                          # (n, 4, n_features)

    # Average absolute SHAP across all training samples → (4, n_features)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    return mean_abs, active_names   # active_names is a subset of FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build the 3-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def _draw_pca_scatter(ax: plt.Axes, X: pd.DataFrame, labels: np.ndarray, pipe) -> None:
    """Panel D — 2-D PCA scatter plot coloured by the 4 verdict classes.

    PCA projects the 19 forensic features down to 2 principal components so we
    can plot all training samples on a flat plane.  Each dot is one document;
    the colour tells you the ground-truth label.  Well-separated clusters mean
    the features carry good discriminative signal.

    The imputer is applied first (same as during training) so that missing-value
    rows are handled identically to the real training pipeline.
    """
    # Impute NaNs with the training-time median before PCA (PCA cannot handle NaN)
    imputer = pipe.named_steps["imputer"]
    X_imp = imputer.transform(X)                        # (n, n_features)

    # Two principal components capture the two directions of maximum variance
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_imp)                   # (n, 2)

    var_explained = pca.explained_variance_ratio_ * 100  # percent

    # Draw each class as its own scatter series so the legend is automatic
    for cls in range(4):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=_CLASS_COLORS[cls],
            label=f"{ML_VERDICT_LABELS[cls]}  (n={mask.sum()})",
            s=55,
            alpha=0.80,
            edgecolors="#0f172a",
            linewidths=0.4,
        )

    ax.set_xlabel(f"PC 1  ({var_explained[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC 2  ({var_explained[1]:.1f}% variance)", fontsize=10)
    ax.set_title("D  |  PCA Scatter — 19 Signals → 2 Dimensions",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=9,
              facecolor="#1e293b", edgecolor="#334155")
    ax.grid(color="#334155", linewidth=0.5, alpha=0.4)


def _plot_scatter_standalone(X: pd.DataFrame, labels: np.ndarray, pipe) -> None:
    """Write a standalone scatter PNG (larger, square, easier to share)."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f172a")
    fig.suptitle(
        "BaseTruth — Training Data PCA Scatter Plot",
        color="white", fontsize=16, fontweight="bold", y=0.98,
    )
    _style_axes(ax)
    _draw_pca_scatter(ax, X, labels, pipe)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(_SCATTER_OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Scatter plot saved", extra={"path": _SCATTER_OUT_PATH})


def _plot(X: pd.DataFrame, labels: np.ndarray, pipe) -> None:
    """Render and save the 4-panel training summary figure.

    Layout:
        A) Top-left:   Class distribution bar chart
        B) Top-right:  Global feature importances (XGBoost gain, sorted)
        C) Mid-bottom: Per-class SHAP heatmap (4 rows × n_features columns)
        D) Bot-bottom: PCA scatter plot — all samples coloured by class
    """
    fig = plt.figure(figsize=(18, 18), facecolor="#0f172a")   # dark navy background
    fig.suptitle(
        "BaseTruth — ML Training Summary",
        color="white",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )

    # GridSpec: top row = 2 equal panels; bottom 2 rows = 1 wide panel each
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35,
                          left=0.06, right=0.97, top=0.96, bottom=0.04)
    ax_dist    = fig.add_subplot(gs[0, 0])   # Panel A — class distribution
    ax_fi      = fig.add_subplot(gs[0, 1])   # Panel B — feature importances
    ax_shap    = fig.add_subplot(gs[1, :])   # Panel C — SHAP heatmap (full width)
    ax_scatter = fig.add_subplot(gs[2, :])   # Panel D — PCA scatter (full width)

    _style_axes(ax_dist)
    _style_axes(ax_fi)
    _style_axes(ax_shap)
    _style_axes(ax_scatter)

    _draw_class_distribution(ax_dist, labels)
    _draw_feature_importances(ax_fi, pipe)
    _draw_shap_heatmap(ax_shap, X, labels, pipe)
    _draw_pca_scatter(ax_scatter, X, labels, pipe)

    log.info("Saving figure", extra={"path": _OUT_PATH})
    plt.savefig(_OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Figure saved", extra={"path": _OUT_PATH})


def _style_axes(ax: plt.Axes) -> None:
    """Apply a consistent dark-theme style to a single Axes object.

    Dark background with muted grid lines — matches the BaseTruth web UI palette.
    """
    ax.set_facecolor("#1e293b")              # slate-800 panel background
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")       # slate-700 border


def _draw_class_distribution(ax: plt.Axes, labels: np.ndarray) -> None:
    """Panel A — horizontal bar chart of sample counts per class.

    Displays exact sample counts as text on each bar so the viewer can
    immediately see the class balance / imbalance without reading the axis.
    """
    counts = np.bincount(labels, minlength=4)   # counts[0..3]
    y_pos  = np.arange(4)

    bars = ax.barh(
        y_pos,
        counts,
        color=_COLORS,
        edgecolor="#334155",
        height=0.6,
    )

    # Label each bar with its exact count at the right edge
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(_LABELS, fontsize=9)
    ax.set_xlabel("Sample Count", fontsize=10)
    ax.set_title("A  |  Class Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, max(counts) * 1.25)  # extra room for count labels
    ax.grid(axis="x", color="#334155", linewidth=0.6, alpha=0.5)
    ax.invert_yaxis()   # class 0 at the top matches the legend order


def _draw_feature_importances(ax: plt.Axes, pipe) -> None:
    """Panel B — horizontal bar chart of per-feature gain importance (global).

    Gain importance measures how much each feature reduced the loss when it was
    chosen as a split point, averaged over all trees and all classes.  Higher
    gain = more discriminative signal.

    Features are sorted from most to least important so the most valuable
    signals are immediately visible at the top.
    """
    booster = pipe.named_steps["model"].get_booster()
    n_booster = booster.num_features()
    active_names = FEATURE_NAMES[:n_booster]

    # get_score returns only features that were actually used in at least one split
    raw_scores = booster.get_score(importance_type="gain")

    # Build a complete importance vector (zero for unused features)
    importance_values = []
    for i, name in enumerate(active_names):
        fi_key = f"f{i}"    # XGBoost names features f0, f1, … when no names are set
        importance_values.append(raw_scores.get(fi_key, 0.0))

    # Create a short human-readable label for each feature (strip trailing _score/_ratio etc.)
    display_names = [
        n.replace("_score", "").replace("_ratio", "").replace("_count", "").replace("_", " ")
        for n in active_names
    ]

    # Sort by importance so the biggest bars are at the top
    order = np.argsort(importance_values)   # ascending; we'll invert y-axis
    sorted_vals  = [importance_values[i] for i in order]
    sorted_names = [display_names[i]     for i in order]

    # Colour bars on a green→red gradient to match "safe→risky" intuition
    cmap = LinearSegmentedColormap.from_list("risk", ["#22c55e", "#ef4444"])
    bar_colors = [cmap(v / max(sorted_vals) if max(sorted_vals) > 0 else 0)
                  for v in sorted_vals]

    ax.barh(range(len(sorted_vals)), sorted_vals, color=bar_colors,
            edgecolor="#334155", height=0.7)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Mean Gain", fontsize=10)
    ax.set_title("B  |  Feature Importance  (Gain)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", color="#334155", linewidth=0.6, alpha=0.5)
    ax.invert_yaxis()   # highest importance at the top


def _draw_shap_heatmap(ax: plt.Axes, X: pd.DataFrame, labels: np.ndarray, pipe) -> None:
    """Panel C — heatmap of mean |SHAP| for each class × feature combination.

    Each cell (row=class, col=feature) shows how strongly that forensic signal
    pushed the model toward that class verdict on average across all training
    samples of that class.

    Reading guide:
      - Bright cells = feature matters a lot for that class.
      - Dark cells   = feature barely influenced that class decision.
      - Compare rows to see which signals are class-specific vs. shared.
    """
    mean_shap, active_names = _compute_shap(X, pipe)   # shape (4, n_features)

    # Shorten feature names for horizontal axis labels
    short_names = [
        n.replace("_score", "").replace("_ratio", "").replace("_count", "").replace("_", "\n")
        for n in active_names
    ]

    # Normalise each column (feature) so colours reflect relative importance within
    # each feature — this stops a single dominant feature from washing out the rest.
    col_max = mean_shap.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0        # avoid division by zero for unused features
    normalised = mean_shap / col_max   # values 0..1

    # Dark-themed heatmap: from dark slate to class colour blend
    im = ax.imshow(
        normalised,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("shap", ["#0f172a", "#f59e0b", "#ef4444"]),
        vmin=0,
        vmax=1,
    )

    # Annotate every cell with its rounded mean |SHAP| value
    for row in range(normalised.shape[0]):
        for col in range(normalised.shape[1]):
            val = mean_shap[row, col]
            text_color = "black" if normalised[row, col] > 0.6 else "white"
            ax.text(col, row, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    # Axis labels
    ax.set_xticks(range(len(active_names)))
    ax.set_xticklabels(short_names, fontsize=8, rotation=0)
    ax.set_yticks(range(4))
    ax.set_yticklabels(_LABELS, fontsize=9)

    # Colour-coded row borders matching the class palette — makes rows scannable
    for row_idx, (class_color, label) in enumerate(zip(_COLORS, _LABELS)):
        ax.axhline(row_idx - 0.5, color=class_color, linewidth=1.5, alpha=0.4)
    ax.axhline(3.5, color="#334155", linewidth=1)

    ax.set_title(
        "C  |  Per-Class SHAP  —  Mean |SHAP| (column-normalised)",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Colour bar on the right edge
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Relative Influence", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # Legend patches — one per class
    patches = [
        mpatches.Patch(color=_CLASS_COLORS[i], label=ML_VERDICT_LABELS[i])
        for i in range(4)
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.2,
              labelcolor="white", fontsize=8, facecolor="#1e293b")


# ─────────────────────────────────────────────────────────────────────────────
# Decision-tree diagram (pure matplotlib — no graphviz dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _build_tree_dict(df_tree: pd.DataFrame) -> dict:
    """Parse one tree's rows into a node-keyed dict with left/right child links.

    Each row in the XGBoost tree dataframe represents one node.  The 'Yes' and
    'No' columns hold the child IDs as 'treeIdx-nodeIdx' strings.  We strip the
    tree-index prefix and store integer node IDs for easy lookup.
    """
    nodes: dict = {}
    for _, row in df_tree.iterrows():
        nid = int(row["Node"])
        nodes[nid] = {
            "feature": row["Feature"],     # e.g. "f4" or "Leaf"
            "split":   row["Split"],       # threshold value (NaN for leaves)
            "yes_raw": row["Yes"],         # e.g. "0-1"  (left child)
            "no_raw":  row["No"],          # e.g. "0-2"  (right child)
            "gain":    row["Gain"],        # leaf value or split gain
            "cover":   row["Cover"],       # number of training samples
            "left":    None,              # filled below
            "right":   None,
        }

    # Resolve child node IDs — format is "{tree_idx}-{node_idx}"
    for nid, node in nodes.items():
        if node["feature"] != "Leaf":
            def _parse(raw) -> int | None:
                if not isinstance(raw, str):
                    return None
                parts = raw.split("-")
                return int(parts[-1])  # take the node-index part
            node["left"]  = _parse(node["yes_raw"])
            node["right"] = _parse(node["no_raw"])

    return nodes


def _assign_positions(nodes: dict, nid: int, depth: int, counter: list) -> None:
    """In-order traversal assigns x positions so the tree layout is tidy.

    Left subtree gets smaller x values, parent comes next, then right subtree.
    y is simply -depth so the root is at the top and leaves are at the bottom.
    """
    node = nodes[nid]
    # Recurse left first so left subtree occupies lower x positions
    if node["left"] is not None:
        _assign_positions(nodes, node["left"],  depth + 1, counter)

    # Claim the next x slot for this node
    node["x"] = counter[0]
    node["y"] = -depth
    counter[0] += 1

    # Recurse right
    if node["right"] is not None:
        _assign_positions(nodes, node["right"], depth + 1, counter)


def _draw_single_tree(
    ax: plt.Axes,
    nodes: dict,
    nid: int,
    feat_labels: list[str],
    class_color: str,
    box_w: float,
    box_h: float,
) -> None:
    """Recursively draw all nodes and edges for one decision tree.

    Split nodes get a blue-tinted box; leaf nodes get a class-coloured box.
    Each box shows the split condition (or leaf value) plus the sample coverage.
    Lines connect parent to child; 'Y' (left) and 'N' (right) are labelled.
    box_w and box_h are passed in so they can be tuned to the tree's actual size.
    """
    node = nodes[nid]
    nx_, ny_ = node["x"], node["y"]
    is_leaf = node["feature"] == "Leaf"

    # ── Box colours ─────────────────────────────────────────────────────────
    face_color = "#1e3a5f" if not is_leaf else "#0f2a1f"
    edge_color = class_color if is_leaf else "#60a5fa"

    rect = plt.Rectangle(
        (nx_ - box_w / 2, ny_ - box_h / 2),
        box_w, box_h,
        linewidth=1.0,
        edgecolor=edge_color,
        facecolor=face_color,
        zorder=3,
    )
    ax.add_patch(rect)

    # ── Box text ─────────────────────────────────────────────────────────────
    font_size = max(4.5, min(7.0, box_w * 7))   # scales with box width
    if is_leaf:
        text = f"leaf\n{node['gain']:.4f}\nn={int(node['cover'])}"
        txt_color = class_color
    else:
        feat_idx  = int(node["feature"][1:])
        feat_name = feat_labels[feat_idx] if feat_idx < len(feat_labels) else node["feature"]
        short = (feat_name
                 .replace("_ratio", "_r")
                 .replace("_count", "_n")
                 .replace("_suspicious", "_sus")
                 .replace("_outlier", "_out")
                 .replace("sharpness", "sharp")
                 .replace("_hotspot", "_hot")
                 .replace("anomaly", "anom")
                 .replace("_largest_blob_px", "_blob")
                 .replace("_high_density", "_hd"))
        text = f"{short}\n< {node['split']:.3f}\nn={int(node['cover'])}"
        txt_color = "#e2e8f0"

    ax.text(
        nx_, ny_,
        text,
        ha="center", va="center",
        fontsize=font_size,
        color=txt_color,
        fontweight="bold" if is_leaf else "normal",
        zorder=4,
        linespacing=1.3,
    )

    # ── Edges to children ────────────────────────────────────────────────────
    for child_key, label in (("left", "Y"), ("right", "N")):
        child_id = node[child_key]
        if child_id is None:
            continue
        child = nodes[child_id]
        cx_, cy_ = child["x"], child["y"]

        ax.plot(
            [nx_, cx_],
            [ny_ - box_h / 2, cy_ + box_h / 2],
            color="#475569", linewidth=0.7, zorder=2,
        )
        # Place Y/N label near the parent end of the edge
        lx = nx_ + (cx_ - nx_) * 0.22
        ly = ny_ - box_h / 2 - box_h * 0.15
        ax.text(lx, ly, label, ha="center", va="top",
                fontsize=font_size * 0.8, color="#94a3b8")

        _draw_single_tree(ax, nodes, child_id, feat_labels, class_color, box_w, box_h)


def _plot_tree_diagrams(pipe) -> None:
    """Draw the first-round decision tree for each of the 4 verdict classes.

    XGBoost multiclass (multi:softprob, 4 classes) trains 4 trees per boosting
    round — one per class.  In the booster's internal order, tree index 0 covers
    class 0, tree index 1 covers class 1, etc.  These 'round 0' trees capture the
    most impactful initial splits learned from the full training set and are the
    most readable (later rounds add small refinements on the residuals).

    The figure is 2×2 — one panel per class — with a dark theme matching the
    BaseTruth UI.  Each node box shows:
        • Split nodes: feature name, threshold, and sample coverage
        • Leaf nodes:  raw XGBoost leaf score (positive = nudge toward this class)
    YES (feature < threshold) branches go left; NO branches go right.
    """
    booster = pipe.named_steps["model"].get_booster()
    df_all  = booster.trees_to_dataframe()

    # 4 classes × 300 rounds = 1200 trees; class k at round r is at index r*4 + k
    # We plot round 0: tree indices 0, 1, 2, 3

    fig, axes = plt.subplots(2, 2, figsize=(38, 28), facecolor="#0f172a")
    fig.suptitle(
        "BaseTruth — XGBoost Decision Trees  (Round 0, one tree per class)",
        color="white", fontsize=16, fontweight="bold", y=0.995,
    )

    for class_idx, ax in enumerate(axes.flat):
        _style_axes(ax)

        class_label = ML_VERDICT_LABELS[class_idx]
        class_color = _CLASS_COLORS[class_idx]

        # Filter to the single tree for this class at round 0
        df_tree = df_all[df_all["Tree"] == class_idx].reset_index(drop=True)

        # Build the node structure and assign raw in-order positions
        nodes = _build_tree_dict(df_tree)
        _assign_positions(nodes, 0, 0, [0])

        # Determine raw coordinate ranges so we can set axis limits and box sizes
        all_x = [n["x"] for n in nodes.values()]
        all_y = [n["y"] for n in nodes.values()]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_span = max(x_max - x_min, 1)
        y_span = max(abs(y_min - y_max), 1)   # depth of the tree

        # Box should be slightly narrower than the minimum x gap (= 1 unit)
        # and proportionally tall based on the depth-to-width aspect ratio.
        box_w = 0.82           # x units; leaves are 1 unit apart
        box_h = box_w * (x_span / max(y_span, 1)) * 0.45  # scale to aspect ratio
        box_h = max(0.30, min(box_h, 0.70))  # clamp to readable range

        # Draw the tree
        _draw_single_tree(ax, nodes, 0, FEATURE_NAMES, class_color, box_w, box_h)

        # Add a small margin around the bounding box
        pad_x = max(1.0, x_span * 0.04)
        pad_y = max(0.5, y_span * 0.12)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_title(
            f"Class {class_idx}  |  {class_label}",
            color=class_color, fontsize=13, fontweight="bold", pad=8,
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.993])
    plt.savefig(_TREE_OUT_PATH, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Tree diagram saved", extra={"path": _TREE_OUT_PATH})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load data + model, generate the 3-panel PNG, and print the output path."""
    log.info("Starting training visualisation")

    if not os.path.exists(_CSV_PATH):
        log.error("Training CSV not found", extra={"path": _CSV_PATH})
        sys.exit(1)

    if not os.path.exists(_MODEL_PATH):
        log.error("Trained model not found", extra={"path": _MODEL_PATH})
        sys.exit(1)

    X, labels = _load_features()
    pipe = joblib.load(_MODEL_PATH)
    log.info("Model loaded", extra={"path": _MODEL_PATH})

    _plot(X, labels, pipe)
    print(f"[visualize_training] Saved -> {_OUT_PATH}")

    _plot_scatter_standalone(X, labels, pipe)
    print(f"[visualize_training] Scatter -> {_SCATTER_OUT_PATH}")

    _plot_tree_diagrams(pipe)
    print(f"[visualize_training] Tree diagram -> {_TREE_OUT_PATH}")


if __name__ == "__main__":
    main()
