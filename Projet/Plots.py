#Importation des librairies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Analysis import moves
from collect import FIG_DIR, MIN_CELL_N, TICKERS
from features import LARGE_MOVE


COLORS = {"drop": "#c0392b", "rise": "#2471a3"}
LABELS = {"drop": "Forte baisse", "rise": "Forte hausse"}
NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon",
    "NVDA": "Nvidia", "JPM": "JPMorgan", "TTE": "TotalEnergies",
    "NFLX": "Netflix", "SBUX": "Starbucks",
}


def _name(ticker):
    """Nom lisible d'une action (retombe sur le ticker s'il est inconnu)."""
    return NAMES.get(ticker, ticker)


def _save(fig, name):
    if name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_group_bar(summary, save_as = None):
    """Médiane du volume relatif : baisses vs hausses, une paire de barres par action."""
    s = summary.reindex(list(TICKERS))
    x, w = np.arange(len(s)), 0.38
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(s)), 5))
    ax.bar(x - w / 2, s["median_relvol_drop"], w, color=COLORS["drop"], label=LABELS["drop"])
    ax.bar(x + w / 2, s["median_relvol_rise"], w, color=COLORS["rise"], label=LABELS["rise"])
    ax.axhline(1, color="grey", ls="--", lw=1, label="Volume habituel (= 1)")
    ax.set_xticks(x, [_name(t) for t in s.index], rotation=30, ha="right")
    ax.set_ylabel("Médiane du volume relatif")
    ax.set_title("Volume relatif les jours de forte baisse vs forte hausse")
    ax.legend()
    _save(fig, save_as)
    return fig


plot_grouped_bar = plot_group_bar  # alias


def plot_boxplots(df, save_as=None):
    """Distribution du volume relatif (échelle log), un panneau par action."""
    m = moves(df)
    n = len(TICKERS)
    ncols = min(n, 5)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.2 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()
    for i, (ax, t) in enumerate(zip(flat, TICKERS)):
        sns.boxplot(data=m[m["ticker"] == t], x="move_type", y="rel_volume",
                    order=["drop", "rise"], hue="move_type", palette=COLORS,
                    legend=False, ax=ax, flierprops={"markersize": 2, "alpha": 0.4})
        ax.set_yscale("log")
        ax.axhline(1, color="grey", ls="--", lw=1)
        ax.set_xticks([0, 1], ["Baisse", "Hausse"])
        ax.set_xlabel("")
        ax.set_ylabel("Volume relatif (échelle log)" if i % ncols == 0 else "")
        ax.set_title(_name(t))
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Distribution du volume relatif selon le sens du mouvement")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_yearly_lines(yearly, save_as=None):
    """Écart de médiane de volume relatif (baisse - hausse) par année."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TICKERS:
        ax.plot(yearly.index, yearly[t], marker="o", label=_name(t))
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(yearly.index)
    ax.set_xlabel("Année")
    ax.set_ylabel("Médiane vol. relatif : baisses - hausses")
    ax.set_title("Évolution de l'écart au fil des années")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, save_as)
    return fig


def plot_heatmap(diff, nmin, save_as=None):
    """Écart baisses - hausses par action et ampleur du mouvement (n = effectif minimal)."""
    labels = diff.copy().astype(object)
    for i in diff.index:
        for j in diff.columns:
            if pd.isna(diff.loc[i, j]) or pd.isna(nmin.loc[i, j]):
                labels.loc[i, j] = "n/a"
                continue
            flag = "*" if nmin.loc[i, j] < MIN_CELL_N else ""
            labels.loc[i, j] = f"{diff.loc[i, j]:+.2f}\n(n={int(nmin.loc[i, j])}){flag}"
    fig, ax = plt.subplots(figsize=(8, max(5, 0.6 * len(diff))))
    sns.heatmap(diff.astype(float), annot=labels.values, fmt="", cmap="RdBu_r", center=0,
                cbar_kws={"label": "Médiane vol. relatif : baisses - hausses"}, ax=ax)
    ax.set_yticklabels([_name(t) for t in diff.index], rotation=0)
    ax.set_xlabel("Ampleur du mouvement de prix")
    ax.set_ylabel("")
    ax.set_title(f"Écart selon l'ampleur (* : n < {MIN_CELL_N} dans un des deux groupes)")
    _save(fig, save_as)
    return fig


def plot_signed_curve(curve, save_as=None):
    """Figure 5 : médiane du log du volume relatif selon le rendement signé du jour.

    Une asymétrie gauche/droite (baisses vs hausses de même ampleur) se lit directement.
    Barres grises : nombre de jours par tranche (axe de droite). Les tranches trop
    peu fournies (voir min_n dans Analysis.signed_return_curve) ne sont pas tracées.
    """
    x = 100 * curve["bin_mid"]
    width = 0.9 * 100 * (curve["bin_right"] - curve["bin_left"]).median()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax_n = ax.twinx()
    ax_n.bar(x, curve["n"], width=width, color="lightgrey", alpha=0.6, zorder=0)
    ax_n.set_ylabel("Nombre de jours par tranche (barres grises)")
    ax_n.grid(False)
    ax.set_zorder(ax_n.get_zorder() + 1)   # courbe devant les barres
    ax.patch.set_visible(False)

    ax.axvspan(x.min() - width, 0, color=COLORS["drop"], alpha=0.05)
    ax.axvspan(0, x.max() + width, color=COLORS["rise"], alpha=0.05)
    ax.fill_between(x, curve["ci_low"], curve["ci_high"], color="grey", alpha=0.35,
                    label="IC bootstrap à 95 %")
    ax.plot(x, curve["median_lrv"], marker="o", color="black", label="Médiane du LRV")
    ax.axhline(0, color="grey", ls="--", lw=1, label="Volume habituel (LRV = 0)")
    ax.axvline(0, color="black", lw=1)
    for sign in (-1, 1):
        ax.axvline(sign * 100 * LARGE_MOVE, color="black", ls=":", lw=1)

    ax.set_xlabel("Rendement journalier (log-rendement, %)")
    ax.set_ylabel("Médiane du log du volume relatif (LRV)")
    ax.set_title("Volume relatif selon le rendement du jour (toutes actions réunies)")
    ax.legend(loc="upper center")
    _save(fig, save_as)
    return fig