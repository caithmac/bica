import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# Use a clean, paper-friendly style
matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # editable text in PDF (no type-3 fonts)
    "ps.fonttype": 42,
})

# ---- Verified numbers (E9) ----
factors = ["Model\nfamily", "Ligand\nrepresentation", "Protein\nrepresentation"]
eta2    = [0.66, 0.31, 0.13]
ci_low  = [0.60, 0.23, 0.06]
ci_high = [0.84, 0.45, 0.40]
pvals   = [r"$p<10^{-4}$", r"$p<10^{-4}$", r"$p=0.14$ (n.s.)"]

# Convert CIs to error-bar offsets (asymmetric)
err_lower = [e - lo for e, lo in zip(eta2, ci_low)]
err_upper = [hi - e for e, hi in zip(eta2, ci_high)]
yerr = np.array([err_lower, err_upper])

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6.5, 3.2))   # wide, fits a two-column figure

x = np.arange(len(factors))
colors = ["#2c7fb8", "#7fcdbb", "#c7e9b4"]   # dark->light: family strongest

bars = ax.bar(x, eta2, yerr=yerr, capsize=5, color=colors,
              edgecolor="black", linewidth=0.8, width=0.6,
              error_kw={"elinewidth": 1.0, "capthick": 1.0})

# Annotate p-values above each error bar
for xi, e, hi, p in zip(x, eta2, ci_high, pvals):
    ax.text(xi, hi + 0.03, p, ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(factors)
ax.set_ylabel(r"Partial $\eta^2$ (effect size)")
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Light horizontal grid only
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig_variance.pdf", bbox_inches="tight")
print("Wrote figures/fig_variance.pdf")
