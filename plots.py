# -*- coding: utf-8 -*-
"""
plots.py
========================
Visualizes the trained ridge regression model from poker_analysis.py.

Generates four image files and a machine-readable results.json:
  1. Equity Surface Heatmaps  — model predictions vs Monte Carlo ground truth
                                for suited, offsuit, and pairs
  2. Residual Error Map       — where the model over/under-estimates
  3. Feature Weight Analysis  — which polynomial features drive predictions

Requirements:
  pip install numpy scikit-learn matplotlib

Run from the project root (same folder as the .npy files):
  python plots.py
"""

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import PolynomialFeatures

# ── 0. Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4a",
    "axes.labelcolor":  "#c8ccd8",
    "xtick.color":      "#c8ccd8",
    "ytick.color":      "#c8ccd8",
    "text.color":       "#c8ccd8",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
})

RANK_LABELS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']

# ── 1. Load artifacts ─────────────────────────────────────────────────────────
print("Loading model and dataset...")
try:
    theta     = np.load('model_theta.npy')
    mu        = np.load('model_mu.npy')
    sigma     = np.load('model_sigma.npy')
    X_raw     = np.load('X_raw.npy')
    y_raw     = np.load('y_raw.npy')
except FileNotFoundError as e:
    raise SystemExit(
        f"\nMissing file: {e}\n"
        "Make sure all .npy files are in the same directory as this script.\n"
        "Run generate_dataset.py and poker_analysis.py first."
    )

print(f"  theta shape : {theta.shape}")
print(f"  X_raw shape : {X_raw.shape}   (169 hands × 4 features)")
print(f"  y_raw shape : {y_raw.shape}\n")

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

def predict(features_2d: np.ndarray) -> np.ndarray:
    """Run the full inference pipeline: expand → scale → bias → dot."""
    X_exp    = poly.fit_transform(features_2d.astype(np.float32))
    X_scaled = (X_exp - mu) / (sigma + 1e-8)
    X_biased = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
    return np.clip(X_biased @ theta, 0.0, 1.0)

# ── 2. Build full prediction grids ───────────────────────────────────────────
# Grids are 13×13 (rank 0–12 on both axes).
# Convention: row = high card rank, col = low card rank.
# Only cells where col <= row are valid hands.

suited_pred  = np.full((13, 13), np.nan)
offsuit_pred = np.full((13, 13), np.nan)
suited_true  = np.full((13, 13), np.nan)
offsuit_true = np.full((13, 13), np.nan)
pair_pred    = np.zeros(13)
pair_true    = np.zeros(13)

# --- Predictions (dense grid) ---
for hi in range(13):
    for lo in range(hi + 1):  # lo <= hi
        if hi == lo:
            # Pair
            feat = np.array([[hi, lo, 0, 1]], dtype=np.float32)
            pair_pred[hi] = predict(feat)[0]
        else:
            # Suited
            feat_s = np.array([[hi, lo, 1, 0]], dtype=np.float32)
            suited_pred[hi, lo] = predict(feat_s)[0]
            # Offsuit
            feat_o = np.array([[hi, lo, 0, 0]], dtype=np.float32)
            offsuit_pred[hi, lo] = predict(feat_o)[0]

# --- Ground truth (from dataset) ---
for row, (feats, eq) in enumerate(zip(X_raw, y_raw)):
    hi, lo, is_s, is_p = int(feats[0]), int(feats[1]), int(feats[2]), int(feats[3])
    if is_p:
        pair_true[hi] = eq
    elif is_s:
        suited_true[hi, lo] = eq
    else:
        offsuit_true[hi, lo] = eq

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# FIGURE 1 — Equity Surface: Model vs Ground Truth
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
print("Rendering Figure 1 — Equity Surface Heatmaps...")
fig1, axes = plt.subplots(2, 2, figsize=(16, 13))
fig1.suptitle(
    "Texas Hold'em Equity Surface\nModel Predictions vs Monte Carlo Estimates",
    fontsize=14, fontweight='bold', y=0.98
)

cmap       = "RdYlGn"
vmin, vmax = 0.30, 0.75

panels = [
    (axes[0, 0], suited_pred,  "Suited — Model",        True),
    (axes[0, 1], suited_true,  "Suited — Monte Carlo",  True),
    (axes[1, 0], offsuit_pred, "Offsuit — Model",       False),
    (axes[1, 1], offsuit_true, "Offsuit — Monte Carlo", False),
]

for ax, grid, title, is_suited in panels:
    # Mask upper triangle (invalid hands) + diagonal (pairs shown separately)
    mask = np.zeros((13, 13), dtype=bool)
    for r in range(13):
        for c in range(13):
            if c >= r:          # upper triangle and diagonal
                mask[r, c] = True

    plot_grid = np.where(mask, np.nan, grid)

    im = ax.imshow(
        plot_grid,
        cmap=cmap, vmin=vmin, vmax=vmax,
        origin='upper', aspect='auto',
        interpolation='nearest'
    )

    # Annotate each cell with % value
    for r in range(13):
        for c in range(r):      # strict lower triangle only
            val = plot_grid[r, c]
            if not np.isnan(val):
                text_color = "black" if 0.38 < val < 0.68 else "white"
                ax.text(c, r, f"{val*100:.0f}", ha='center', va='center',
                        fontsize=6.5, color=text_color, fontweight='bold')

    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(RANK_LABELS, fontsize=8)
    ax.set_yticklabels(RANK_LABELS, fontsize=8)
    ax.set_xlabel("Low Card Rank",  fontsize=9)
    ax.set_ylabel("High Card Rank", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Equity", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# Pair bar chart inset (bottom-right of fig1 overridden)
ax_pairs = axes[1, 1]   # reuse; add pair bars as a small secondary figure instead
fig_pair, ax_pair = plt.subplots(figsize=(7, 3.5))
fig_pair.patch.set_facecolor("#0f1117")
ax_pair.set_facecolor("#1a1d27")

x = np.arange(13)
bar_w = 0.35
bars_pred = ax_pair.bar(x - bar_w/2, pair_pred * 100, bar_w,
                         label='Model', color='#4ea8de', alpha=0.85)
bars_true = ax_pair.bar(x + bar_w/2, pair_true * 100, bar_w,
                         label='Monte Carlo', color='#f7c59f', alpha=0.85)

ax_pair.set_xticks(x)
ax_pair.set_xticklabels(RANK_LABELS)
ax_pair.set_xlabel("Pair Rank",    fontsize=10)
ax_pair.set_ylabel("Equity (%)",   fontsize=10)
ax_pair.set_title("Pocket Pairs — Model vs Monte Carlo", fontsize=11, fontweight='bold')
ax_pair.legend(fontsize=9)
ax_pair.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax_pair.grid(axis='y', alpha=0.4)
ax_pair.set_ylim(0, 100)

for bar in bars_pred:
    ax_pair.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{bar.get_height():.0f}", ha='center', va='bottom',
                 fontsize=6, color='#c8ccd8')
for bar in bars_true:
    ax_pair.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{bar.get_height():.0f}", ha='center', va='bottom',
                 fontsize=6, color='#c8ccd8')

fig_pair.tight_layout()
fig_pair.savefig('plot_pairs.png', dpi=160, bbox_inches='tight',
                  facecolor=fig_pair.get_facecolor())
print("  Saved: plot_pairs.png")

fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig('plot_equity_surface.png', dpi=160, bbox_inches='tight',
              facecolor=fig1.get_facecolor())
print("  Saved: plot_equity_surface.png")

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# FIGURE 2 — Residual Error Map (Prediction − Ground Truth)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
print("Rendering Figure 2 — Residual Error Maps...")
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle(
    "Residual Error Map  (Model − Monte Carlo)\n"
    "Green = model overestimates  |  Red = model underestimates",
    fontsize=13, fontweight='bold'
)

residual_suited  = suited_pred  - suited_true
residual_offsuit = offsuit_pred - offsuit_true

err_panels = [
    (axes2[0], residual_suited,  "Suited Residuals"),
    (axes2[1], residual_offsuit, "Offsuit Residuals"),
]

err_lim = 0.04   # ±4 pp colour scale

for ax, res_grid, title in err_panels:
    mask = np.zeros((13, 13), dtype=bool)
    for r in range(13):
        for c in range(13):
            if c >= r:
                mask[r, c] = True
    plot_res = np.where(mask, np.nan, res_grid)

    im = ax.imshow(
        plot_res,
        cmap='RdYlGn', vmin=-err_lim, vmax=err_lim,
        origin='upper', aspect='auto', interpolation='nearest'
    )

    for r in range(13):
        for c in range(r):
            val = plot_res[r, c]
            if not np.isnan(val):
                sign  = "+" if val >= 0 else ""
                tcolor = "black" if abs(val) < 0.025 else "white"
                ax.text(c, r, f"{sign}{val*100:.1f}", ha='center', va='center',
                        fontsize=6, color=tcolor, fontweight='bold')

    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(RANK_LABELS, fontsize=8)
    ax.set_yticklabels(RANK_LABELS, fontsize=8)
    ax.set_xlabel("Low Card Rank",  fontsize=9)
    ax.set_ylabel("High Card Rank", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Error (pp)", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x*100:+.1f}pp")
    )

fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig('plot_residuals.png', dpi=160, bbox_inches='tight',
              facecolor=fig2.get_facecolor())
print("  Saved: plot_residuals.png")

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# FIGURE 3 — Feature Weight Analysis
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
print("Rendering Figure 3 — Feature Weight Analysis...")

# Reconstruct poly feature names
dummy = np.zeros((1, 4), dtype=np.float32)
poly.fit_transform(dummy)
raw_names = poly.get_feature_names_out(['high_card','low_card','is_suited','is_pair'])

# theta[0] = bias; theta[1:] = one per standardised poly feature
feature_names = ['bias'] + list(raw_names)
weights = np.array(theta)    # shape (15,)

# Sort by absolute weight (exclude bias for ranking, keep for display)
order = np.argsort(np.abs(weights[1:]))[::-1]   # indices into raw_names
sorted_names   = [feature_names[i+1] for i in order]
sorted_weights = weights[1:][order]

# Nicer labels
label_map = {
    'high_card':             'high_card  (H)',
    'low_card':              'low_card  (L)',
    'is_suited':             'is_suited  (S)',
    'is_pair':               'is_pair  (P)',
    'high_card^2':           'H²',
    'high_card low_card':    'H × L',
    'high_card is_suited':   'H × S',
    'high_card is_pair':     'H × P',
    'low_card^2':            'L²',
    'low_card is_suited':    'L × S',
    'low_card is_pair':      'L × P',
    'is_suited^2':           'S²',
    'is_suited is_pair':     'S × P',
    'is_pair^2':             'P²',
}
pretty_names = [label_map.get(n, n) for n in sorted_names]

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle(
    "Learned Model Weights  (standardised features)\n"
    "What the regression says matters for predicting equity",
    fontsize=13, fontweight='bold'
)

# Left panel: bar chart of all 14 feature weights
ax_bar = axes3[0]
colors = ['#4ea8de' if w > 0 else '#e06c75' for w in sorted_weights]
bars   = ax_bar.barh(pretty_names[::-1], sorted_weights[::-1], color=colors[::-1],
                      edgecolor='none', height=0.65)
ax_bar.axvline(0, color='#888', linewidth=1)
ax_bar.margins(x=0.20)
ax_bar.set_xlabel("Standardised Weight", fontsize=10)
ax_bar.set_title("All Feature Weights\n(ranked by magnitude)", fontsize=10, fontweight='bold')
ax_bar.grid(axis='x', alpha=0.4)

for bar, val in zip(bars, sorted_weights[::-1]):
    sign = '+' if val >= 0 else ''
    ax_bar.text(
        val + (0.0005 if val >= 0 else -0.0005),
        bar.get_y() + bar.get_height()/2,
        f"{sign}{val:.4f}",
        va='center', ha='left' if val >= 0 else 'right',
        fontsize=7.5, color='#c8ccd8'
    )

# Right panel: impact decomposition — what moves equity the most across real hands
# Show predicted equity contribution = weight × standardised feature value
# averaged over the full dataset
X_exp    = poly.fit_transform(X_raw)
X_scaled = (X_exp - mu) / (sigma + 1e-8)

contributions = np.abs(X_scaled * weights[1:])   # (169, 14)
mean_impact   = contributions.mean(axis=0)        # (14,)

# Re-sort by mean impact
impact_order   = np.argsort(mean_impact)[::-1]
impact_names   = [label_map.get(raw_names[i], raw_names[i]) for i in impact_order]
impact_vals    = mean_impact[impact_order]

ax_imp = axes3[1]
imp_colors = ['#c678dd' if 'is_suited' in raw_names[i]
              else '#e5c07b' if 'is_pair' in raw_names[i]
              else '#98c379'
              for i in impact_order]

ax_imp.barh(impact_names[::-1], impact_vals[::-1],
             color=imp_colors[::-1], edgecolor='none', height=0.65)
ax_imp.set_xlabel("Mean |weight × feature value| across 169 hands", fontsize=9)
ax_imp.set_title("Average Prediction Impact per Feature\n(which features actually move the needle)",
                  fontsize=10, fontweight='bold')
ax_imp.grid(axis='x', alpha=0.4)

legend_elements = [
    plt.Rectangle((0,0),1,1, color='#98c379', label='Rank features'),
    plt.Rectangle((0,0),1,1, color='#c678dd', label='Suited interactions'),
    plt.Rectangle((0,0),1,1, color='#e5c07b', label='Pair interactions'),
]
ax_imp.legend(handles=legend_elements, fontsize=8, loc='lower right')

fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig('plot_feature_weights.png', dpi=160, bbox_inches='tight',
              facecolor=fig3.get_facecolor())
print("  Saved: plot_feature_weights.png")

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Summary stats
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
all_preds = predict(X_raw)
residuals = (all_preds - y_raw) * 100   # in percentage points
errors = all_preds - y_raw
r_squared = 1 - np.sum(errors ** 2) / np.sum((y_raw - y_raw.mean()) ** 2)

print("\n── Model Summary ─────────────────────────────────────────")
print(f"  MSE (full dataset) : {np.mean((all_preds - y_raw)**2):.6f}")
print(f"  MAE (pp)           : {np.abs(residuals).mean():.2f}pp")
print(f"  Max error (pp)     : {np.abs(residuals).max():.2f}pp")
print(f"  R²                 : {r_squared:.5f}")
print(f"\n── Top 3 features by weight magnitude ───────────────────")
for i in range(3):
    print(f"  {i+1}. {pretty_names[i]:20s}  θ = {sorted_weights[i]:+.4f}")
print(f"\n── Top 3 features by average prediction impact ──────────")
for i in range(3):
    print(f"  {i+1}. {impact_names[i]:20s}  impact = {impact_vals[i]:.4f}")

print("\nDone. Output files:")
print("  plot_equity_surface.png")
print("  plot_pairs.png")
print("  plot_residuals.png")
print("  plot_feature_weights.png")

# Record the exact saved artifacts used in the figures. These are snapshot
# diagnostics, not held-out test scores or a record of the original training run.
hand_labels = [
    RANK_LABELS[int(hi)] + RANK_LABELS[int(lo)]
    + ('' if pair else 's' if suited else 'o')
    for hi, lo, suited, pair in X_raw
]
groups = {
    'pairs': X_raw[:, 3] == 1,
    'suited': X_raw[:, 2] == 1,
    'offsuit': (X_raw[:, 2] == 0) & (X_raw[:, 3] == 0),
}
artifact_names = [
    'X_raw.npy', 'y_raw.npy', 'model_theta.npy', 'model_mu.npy', 'model_sigma.npy'
]
summary = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'evaluation': 'Saved model predictions compared with the saved Monte Carlo labels; not held-out test performance.',
    'environment': {name: version(name) for name in ['numpy', 'scikit-learn', 'matplotlib']},
    'artifact_sha256': {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in artifact_names
    },
    'n_hands': int(len(y_raw)),
    'n_features': int(X_raw.shape[1]),
    'n_polynomial_features': int(len(theta) - 1),
    'mse': float(np.mean(errors ** 2)),
    'mae_pp': float(np.abs(residuals).mean()),
    'rmse_pp': float(np.sqrt(np.mean(errors ** 2)) * 100),
    'max_absolute_error_pp': float(np.abs(residuals).max()),
    'r_squared': float(r_squared),
    'groups': {
        name: {
            'count': int(mask.sum()),
            'mae_pp': float(np.abs(residuals[mask]).mean()),
            'max_absolute_error_pp': float(np.abs(residuals[mask]).max()),
        }
        for name, mask in groups.items()
    },
    'hands': [
        {
            'hand': label,
            'monte_carlo_equity': float(actual),
            'model_equity': float(predicted),
            'residual_pp': float(residual),
        }
        for label, actual, predicted, residual in zip(hand_labels, y_raw, all_preds, residuals)
    ],
    'features': [
        {
            'name': str(name),
            'standardized_weight': float(weight),
            'mean_absolute_contribution': float(impact),
        }
        for name, weight, impact in zip(raw_names, weights[1:], mean_impact)
    ],
}
Path('results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print("  results.json")
plt.close('all')
