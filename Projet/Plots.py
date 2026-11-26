#Importation des librairies
import matplotlib as plt
import numpy as np

from Analysis import moves
from collect import FIG_DIR, MIN_CELL_N, TICKERS


COLORS = {"drop": "#c0392b", "rise": "#2471a3"}
LABELS = {"drop": "Forte baisse", "rise": "Forte hausse"}
NAMES = TICKERS


def _save(fig, name):
    if name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_group_bar(summary, save_as = None):
    s = summary.reindex(list(TICKERS))
    x, w = np.arange(len(s)), 0,38
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x - w / 2, s["median_relvol_drop"], w, color=COLORS["drop"], label=LABELS["drop"])
    ax.bar(x + w / 2, s["median_relvol_rise"], w, color=COLORS["rise"], label=LABELS["rise"])
    ax.axhline(1, color="grey", ls="--", lw=1, label="Volume habituel (= 1)")
    ax.set_xticks(x, NAMES)
    ax.set_ylabel("Médiane du volume relatif")
    ax.set_title("Volume relatif les jours de forte baisse vs forte hausse")
    ax.legend()
    _save(fig, save_as)
    return fig   



def plot_boxplots(df, save_as=None):
    m = moves(df)
    fig, axes = plt.subplots(1, len(TICKERS), figsize=(16, 5), sharey=True)
    for ax, (t, name) in zip(axes, TICKERS.items()):
        sns.boxplot(data=m[m["ticker"] == t], x="move_type", y="rel_volume",
                    order=["drop", "rise"], hue="move_type", palette=COLORS,
                    legend=False, ax=ax, flierprops={"markersize": 2, "alpha": 0.4})
        ax.set_yscale("log")
        ax.axhline(1, color="grey", ls="--", lw=1)
        ax.set_xticks([0, 1], ["Baisse", "Hausse"])
        ax.set_xlabel("")
        ax.set_ylabel("Volume relatif (échelle log)" if ax is axes[0] else "")
        ax.set_title(name)
    fig.suptitle("Distribution du volume relatif selon le sens du mouvement")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_yearly_lines(yearly, save_as=None):
    """Écart de médiane de volume relatif (baisse - hausse) par année."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for t, name in TICKERS.items():
        ax.plot(yearly.index, yearly[t], marker="o", label=name)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(yearly.index)
    ax.set_xlabel("Année")
    ax.set_ylabel("Médiane vol. relatif : baisses - hausses")
    ax.set_title("Évolution de l'écart au fil des années")
    ax.legend()
    _save(fig, save_as)
    return fig


def plot_heatmap(diff, nmin, save_as=None):
    """Écart baisses - hausses par action et ampleur du mouvement (n = effectif minimal)."""
    labels = diff.copy().astype(object)
    for i in diff.index:
        for j in diff.columns:
            flag = "*" if nmin.loc[i, j] < MIN_CELL_N else ""
            labels.loc[i, j] = f"{diff.loc[i, j]:+.2f}\n(n={int(nmin.loc[i, j])}){flag}"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(diff.astype(float), annot=labels.values, fmt="", cmap="RdBu_r", center=0,
                cbar_kws={"label": "Médiane vol. relatif : baisses - hausses"}, ax=ax)
    ax.set_yticklabels(NAMES, rotation=0)
    ax.set_xlabel("Ampleur du mouvement de prix")
    ax.set_ylabel("")
    ax.set_title(f"Écart selon l'ampleur (* : n < {MIN_CELL_N} dans un des deux groupes)")
    _save(fig, save_as)
    return fig
