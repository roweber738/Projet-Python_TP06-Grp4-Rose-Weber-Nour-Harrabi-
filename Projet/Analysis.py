import pandas as pd
from scipy.stats import mannwhitneyu

from collect import TICKERS


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
