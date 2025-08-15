"""Pipeline for building NFL rookie projections and boom/bust tiers using nflverse data."""
import ssl
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass

DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
COMBINE_URL = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"
SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"

def load_data(season: int):
    """Download rookies, historical data, and preseason snap counts."""
    ctx = ssl._create_unverified_context()
    draft = pd.read_csv(urlopen(DRAFT_URL, context=ctx))
    combine = pd.read_csv(urlopen(COMBINE_URL, context=ctx))
    try:
        snaps = pd.read_csv(urlopen(SNAP_COUNTS_URL.format(season=season), context=ctx))
    except Exception:
        snaps = pd.DataFrame(columns=["player", "offense_snaps", "game_type"])
    rookies = draft[draft["season"] == season].copy()
    historical = draft[(draft["season"] >= 2000) & (draft["season"] < season)].copy()
    rookies = rookies.merge(combine[["pfr_id", "forty"]], left_on="pfr_player_id", right_on="pfr_id", how="left")
    historical = historical.merge(combine[["pfr_id", "forty"]], left_on="pfr_player_id", right_on="pfr_id", how="left")
    preseason = snaps[snaps.get("game_type") == "PRE"]
    preseason_usage = preseason.groupby("player", as_index=False)["offense_snaps"].sum()
    preseason_usage.rename(columns={"player": "pfr_player_name", "offense_snaps": "preseason_snaps"}, inplace=True)
    return rookies, historical, preseason_usage

def build_comps(rookies: pd.DataFrame, historical: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Find statistical comps for rookies using nearest neighbors."""
    features = ["rush_yards", "rush_tds", "forty"]
    scaler = StandardScaler()
    historical_scaled = scaler.fit_transform(historical[features].fillna(0))
    rookies_scaled = scaler.transform(rookies[features].fillna(0))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(historical_scaled)
    _, indices = nbrs.kneighbors(rookies_scaled)
    comps = []
    for i, idxs in enumerate(indices):
        comp_stats = historical.iloc[idxs]
        comps.append({
            "pfr_player_name": rookies.iloc[i]["pfr_player_name"],
            "position": rookies.iloc[i]["position"],
            "comp_av": comp_stats["dr_av"].mean(),
            "comps": comp_stats["pfr_player_name"].tolist(),
        })
    return pd.DataFrame(comps)

def transition_rates(historical: pd.DataFrame) -> pd.DataFrame:
    """Compute transition rates from college stats to rookie AV by position."""
    tmp = historical.copy()
    tmp["av_per_yard"] = tmp["dr_av"] / tmp["rush_yards"].replace(0, np.nan)
    tmp["av_per_td"] = tmp["dr_av"] / tmp["rush_tds"].replace(0, np.nan)
    rates = tmp.groupby("position").agg(
        transition_yards=("av_per_yard", "mean"),
        transition_tds=("av_per_td", "mean"),
    ).reset_index()
    return rates

def incorporate_transition(rookies: pd.DataFrame, comps: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    merged = rookies.merge(comps, on=["pfr_player_name", "position"]).merge(rates, on="position", how="left")
    merged["transition_av_pred"] = merged["rush_yards"] * merged["transition_yards"].fillna(0)
    merged["pred_av"] = (merged["comp_av"] + merged["transition_av_pred"]) / 2
    return merged

def incorporate_preseason(merged: pd.DataFrame, preseason: pd.DataFrame) -> pd.DataFrame:
    merged = merged.merge(preseason, on="pfr_player_name", how="left")
    merged["preseason_snaps"] = merged["preseason_snaps"].fillna(0)
    max_snaps = merged["preseason_snaps"].max()
    merged["usage_factor"] = merged["preseason_snaps"] / max_snaps if max_snaps > 0 else 0
    merged["boom_metric"] = merged["pred_av"] * (1 + merged["usage_factor"])
    return merged

def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    q33, q66 = df["boom_metric"].quantile([0.33, 0.66])
    def tier(val: float) -> str:
        if val >= q66:
            return "Boom"
        if val >= q33:
            return "Neutral"
        return "Bust"
    df["tier"] = df["boom_metric"].apply(tier)
    return df


@dataclass
class RookieProjectionPipeline:
    """Convenience wrapper for running the rookie projection workflow.

    Parameters
    ----------
    season: int
        Draft season for which to build projections.
    k: int, default 3
        Number of nearest neighbors to use when building comps.
    """

    season: int
    k: int = 3

    def run(self, save_csv: bool = False) -> pd.DataFrame:
        """Execute the full pipeline and return the resulting DataFrame.

        Parameters
        ----------
        save_csv: bool, default False
            Whether to persist the results to ``rookie_boom_bust_<season>.csv``.
        """

        rookies, historical, preseason = load_data(self.season)
        comps = build_comps(rookies, historical, k=self.k)
        rates = transition_rates(historical)
        merged = incorporate_transition(rookies, comps, rates)
        enriched = incorporate_preseason(merged, preseason)
        final = assign_tiers(enriched)
        if save_csv:
            final.to_csv(f"rookie_boom_bust_{self.season}.csv", index=False)
        return final

def main() -> None:
    pipeline = RookieProjectionPipeline(season=2024)
    final = pipeline.run(save_csv=True)
    print(
        final[
            [
                "pfr_player_name",
                "position",
                "comps",
                "pred_av",
                "boom_metric",
                "tier",
            ]
        ]
    )

if __name__ == "__main__":
    main()
