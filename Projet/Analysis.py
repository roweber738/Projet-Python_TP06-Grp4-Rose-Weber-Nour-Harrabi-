import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from collect import TICKERS
from features import LARGE_MOVE

# Période d'étude (la collecte démarre plus tôt, en 2020, pour la médiane glissante du volume)
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-12-31"


def moves(df: pd.DataFrame) -> pd.DataFrame:
    """Uniquement les jours de gros mouvement (baisse ou hausse)."""
    return df[df["move_type"] != "other"]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par action : effectifs, volumes (bruts et relatifs), écarts."""
    rows = []
    for t in TICKERS:
        g = df[df["ticker"] == t]
        d, r = g[g["large_drop"]], g[g["large_rise"]]
        med_d, med_r = d["rel_volume"].median(), r["rel_volume"].median()
        raw_d, raw_r = d["volume"].median(), r["volume"].median()
        rows.append({
            "ticker": t,
            "n_drop": len(d), "n_rise": len(r),
            "mean_vol_drop": d["volume"].mean(), "median_vol_drop": raw_d,
            "mean_vol_rise": r["volume"].mean(), "median_vol_rise": raw_r,
            "median_relvol_drop": med_d, "median_relvol_rise": med_r,
            "diff_relvol": med_d - med_r,
            "diff_relvol_pct": 100 * (med_d / med_r - 1),
            "diff_rawvol_pct": 100 * (raw_d / raw_r - 1),
        })
    return pd.DataFrame(rows).set_index("ticker")


def yearly_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Écart (médiane rel. volume baisse - hausse) par année (lignes) et action (colonnes)."""
    p = moves(df).pivot_table(index=["year", "ticker"], columns="move_type",
                              values="rel_volume", aggfunc="median")
    return (p["drop"] - p["rise"]).unstack("ticker")[list(TICKERS)]


def heatmap_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Écart par action x tranche de mouvement, et effectif minimal (drop/rise) par cellule."""
    m = moves(df).dropna(subset=["magnitude_bin"])
    idx = ["ticker", "magnitude_bin"]
    med = m.pivot_table(index=idx, columns="move_type", values="rel_volume",
                        aggfunc="median", observed=True)
    cnt = m.pivot_table(index=idx, columns="move_type", values="rel_volume",
                        aggfunc="count", observed=True)
    diff = (med["drop"] - med["rise"]).unstack("magnitude_bin").reindex(list(TICKERS))
    nmin = cnt.min(axis=1).unstack("magnitude_bin").reindex(list(TICKERS))
    return diff, nmin


def mann_whitney_table(df: pd.DataFrame) -> pd.DataFrame:
    """Test de Mann-Whitney (bilatéral) : volume relatif baisses vs hausses.

    prob_superiority = P(volume rel. d'un jour de baisse > celui d'un jour de hausse).
    0.5 = aucune différence ; > 0.5 = volumes plus élevés les jours de baisse.
    """
    rows = []
    for t in TICKERS:
        g = df[df["ticker"] == t]
        d = g.loc[g["large_drop"], "rel_volume"]
        r = g.loc[g["large_rise"], "rel_volume"]
        u, p = mannwhitneyu(d, r, alternative="two-sided")
        rows.append({"ticker": t, "n_drop": len(d), "n_rise": len(r),
                     "U": u, "p_value": p, "prob_superiority": u / (len(d) * len(r))})
    return pd.DataFrame(rows).set_index("ticker")


def magnitude_bins(step: float = 0.01) -> tuple[list, list]:
    """Tranches d'ampleur des gros mouvements, construites à partir du seuil LARGE_MOVE.

    Avec LARGE_MOVE = 0.02 et step = 0.01 : [2 %, 3 %[, [3 %, 4 %[, >= 4 %.
    """
    edges = [LARGE_MOVE, LARGE_MOVE + step, LARGE_MOVE + 2 * step, np.inf]
    pct = [round(100 * e, 2) for e in edges[:-1]]
    labels = [f"{pct[0]:g}–{pct[1]:g} %", f"{pct[1]:g}–{pct[2]:g} %", f"> {pct[2]:g} %"]
    return edges, labels


def prepare_analysis_data(features_df: pd.DataFrame,
                          start: str = ANALYSIS_START,
                          end: str = ANALYSIS_END) -> pd.DataFrame:
    """Prépare la base d'analyse à partir du résultat de features.add_features().

    - retire les jours sans volume relatif (60 premiers jours de chaque action) ;
    - restreint à la période d'étude ;
    - ajoute year, abs_return, move_type ("drop" / "rise" / "other") et magnitude_bin.
    Les colonnes de features.py (log_return, rel_volume, large_drop, large_rise) ne sont pas modifiées.
    """
    df = features_df.dropna(subset=["log_return", "rel_volume"]).copy()
    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]

    df["year"] = df["datetime"].dt.year
    df["abs_return"] = df["log_return"].abs()
    df["move_type"] = np.select([df["large_drop"], df["large_rise"]],
                                ["drop", "rise"], default="other")

    edges, labels = magnitude_bins()
    df["magnitude_bin"] = pd.cut(df["abs_return"].where(df["move_type"] != "other"),
                                 bins=edges, labels=labels, right=False)
    return df.reset_index(drop=True)


def signed_return_curve(df: pd.DataFrame, edges=None, min_n: int = 20,
                        n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Médiane du log du volume relatif (LRV) par tranche de rendement signé, toutes actions réunies.

    Utilise TOUS les jours (pas de seuil de gros mouvement) : permet de voir directement
    l'asymétrie entre les rendements négatifs et positifs.
    - edges : bornes des tranches de log_return (par défaut -5 % à +5 %, pas de 0,5 %) ;
    - min_n : en dessous de cet effectif, la tranche n'est pas estimée (NaN) ;
    - IC à 95 % de la médiane par bootstrap (n_boot rééchantillonnages, graine fixée).
    Les jours hors de [-5 %, +5 %[ ne sont pas inclus.
    """
    if edges is None:
        edges = np.round(np.linspace(-0.05, 0.05, 21), 4)
    d = pd.DataFrame({"log_return": df["log_return"], "lrv": np.log(df["rel_volume"])})
    d["ret_bin"] = pd.cut(d["log_return"], bins=edges, right=False)

    rng = np.random.default_rng(seed)
    rows = []
    for interval, g in d.groupby("ret_bin", observed=False):
        x = g["lrv"].to_numpy()
        row = {"bin_left": interval.left, "bin_right": interval.right,
               "bin_mid": (interval.left + interval.right) / 2, "n": len(x),
               "median_lrv": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        if len(x) >= min_n:
            boots = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
            row.update(median_lrv=np.median(x),
                       ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)