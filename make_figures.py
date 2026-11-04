import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import os
R = "results"
os.makedirs("figures", exist_ok=True)
plt.rcParams["font.family"] = "Liberation Serif"
INK = "#1b2430"; BLUE = "#dbe7f3"; TEAL = "#d9efec"; AMB = "#f6e7c4"; GR = "#eeeeee"; PUR = "#e6def0"

def architecture():
    fig, ax = plt.subplots(figsize=(3.45, 3.25), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 96); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.0, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fc, ec=INK, lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=INK))

    box(6, 84, 88, 11, "MovieLens ratings, movies, tags\nper-user temporal split (70/10/20) · TF-IDF features", GR)
    arrow(50, 83.5, 50, 79.5)
    gap = 1.6; bw = (100 - 3 * gap - 1.2) / 4
    xs = [0.6 + i * (bw + gap) for i in range(4)]
    ax.text(50, 77, "Explicit-feedback models", ha="center", fontsize=5.5, style="italic", color=INK)
    for x, t in zip(xs, ["Bias\nbaseline", "Item-\nkNN", "ALS-WR", "Content\nTF-IDF"]):
        box(x, 64, bw, 10.5, t, TEAL, fs=5.6)
    ax.text(50, 60.5, "Implicit-feedback models", ha="center", fontsize=5.5, style="italic", color=INK)
    bw3 = (100 - 2 * gap - 1.2) / 3
    xs3 = [0.6 + i * (bw3 + gap) for i in range(3)]
    for x, t in zip(xs3, ["Implicit\nALS", "BPR", "EASE"]):
        box(x, 48.5, bw3, 9.5, t, PUR, fs=5.6)
    arrow(50, 47.5, 50, 43.5)
    box(14, 34, 72, 9, "Hybrid blend (weights tuned on validation)", AMB, fs=5.8)
    arrow(50, 33.5, 50, 30)
    box(14, 21, 72, 8.5, "Popularity-aware re-ranking (λ$_p$)", AMB, fs=5.8)
    arrow(30, 20.5, 30, 17)
    arrow(70, 20.5, 70, 17)
    box(4, 8, 42, 8.5, "Rating prediction", GR, fs=5.8)
    box(54, 8, 42, 8.5, "Top-N list", GR, fs=5.8)
    ax.text(50, 2.5, "Evaluation: RMSE/MAE · P/R/NDCG@10 · coverage", ha="center", va="center", fontsize=5.6, style="italic", color=INK)
    fig.savefig("figures/fig1.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

def ndcg_bars():
    r = pd.read_csv(f"{R}/ranking.csv").sort_values("ndcg")
    colors = []
    for m in r.model:
        colors.append("#c9702d" if m == "Hybrid + implicit" else "#8aa4bd" if m in ("Implicit ALS", "BPR", "EASE") else "#b9b9b9")
    fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)
    ax.barh(r.model, r.ndcg, color=colors, height=0.68, xerr=r.ndcg_std, error_kw=dict(lw=0.6, capsize=1.5))
    for i, v in enumerate(r.ndcg):
        ax.text(v + r.ndcg_std.iloc[i] + 0.003, i, f"{v:.4f}", va="center", fontsize=6)
    ax.set_xlim(0, 0.13)
    ax.set_xlabel("NDCG@10", fontsize=7)
    ax.tick_params(labelsize=6.5, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig("figures/fig2.png", dpi=300)
    plt.close(fig)

def tradeoff():
    r = pd.read_csv(f"{R}/rerank.csv")
    fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)
    styles = {"Hybrid": ("#8a8a8a", "o", "Hybrid (explicit only)"), "Hybrid + implicit": ("#c9702d", "s", "Hybrid + implicit")}
    for v, (c, mk, lab) in styles.items():
        d = r[r.variant == v]
        ax.plot(d.coverage * 100, d.ndcg, color=c, marker=mk, ms=3.2, lw=0.9, label=lab)
        for _, row in d.iterrows():
            if row.lambda_p in (0.0, 2.0, 4.0, 6.0):
                ax.annotate(f"{row.lambda_p:g}", (row.coverage * 100, row.ndcg), textcoords="offset points",
                            xytext=(3, 3), fontsize=5.8, color=c)
    ax.axhline(0.0730, color="#555", lw=0.6, ls=":")
    ax.text(18.5, 0.0745, "popularity ranker", fontsize=5.8, ha="right", color="#444")
    ax.set_xlabel("Catalog coverage (%)", fontsize=7)
    ax.set_ylabel("NDCG@10", fontsize=7)
    ax.tick_params(labelsize=6.5, length=2)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig("figures/fig3.png", dpi=300)
    plt.close(fig)

architecture(); ndcg_bars(); tradeoff(); print("ok")
