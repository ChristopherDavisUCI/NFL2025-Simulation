
"""
Fast NFL-style standings & tiebreakers with caching.
Updated from original version by ChatGPT

Public API
----------
- make_ind(df_scores) -> pd.DataFrame
- Standings(df_scores: pd.DataFrame, team_to_division: dict[str, str])

`df_scores` schema (expected columns):
  - home_team, away_team : str
  - home_score, away_score : numeric
  - div_game, conf_game : bool
  - schedule_playoff : optional bool (rows with True are excluded)

`team_to_division`:
  mapping team -> division string such as "AFC East", "NFC North", etc.
  Conference is inferred as the first 3 characters (e.g., "AFC", "NFC").

Outputs
-------
Standings(...).standings : DataFrame indexed by Team with columns
  - Wins, Losses, Ties, WLT
  - Points_scored, Points_allowed
  - Division, Conference
  - Division_rank

Standings(...).div_ranks : dict[division] -> list of teams ranked within division
Standings(...).playoffs : dict[conf] -> list of 7 teams (4 division winners + 3 wild cards)
Standings(...).best_reg_record : team with better regular-season record among two conference #1 seeds

Notes
-----
This module emphasizes speed by:
  * building a long-form table once via vectorized ops (make_ind)
  * computing all heavy groupbys/pivots exactly once in _Caches
  * reusing cached Series/matrices in all tiebreakers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import os

from name_helper import get_abbr


# -----------------------------
# Long-form individual game table
# -----------------------------

def make_ind(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Vectorized long-form: 2 rows per game (home, away).

    Excludes rows where schedule_playoff == True if present.
    Adds:
      - Team, Opponent
      - Points_scored, Points_allowed
      - Outcome ("Win"/"Tie"/"Loss")
      - Outcome_points (Win=1.0, Tie=0.5, Loss=0.0)
    """
    df = df_scores.copy()
    if "schedule_playoff" in df.columns:
        df = df.loc[~df["schedule_playoff"]].copy()

    needed = ["home_team","away_team","div_game","conf_game","home_score","away_score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"df_scores is missing required columns: {missing}")

    # build home/away "views"
    home = df.loc[:, ["home_team","away_team","div_game","conf_game","home_score","away_score"]].copy()
    home.columns = ["Team","Opponent","div_game","conf_game","Points_scored","Points_allowed"]

    away = df.loc[:, ["away_team","home_team","div_game","conf_game","away_score","home_score"]].copy()
    away.columns = ["Team","Opponent","div_game","conf_game","Points_scored","Points_allowed"]

    df_ind = pd.concat([home, away], ignore_index=True)

    # normalize team codes early (e.g., LAR->LA) to keep everything consistent
    df_ind['Team'] = df_ind['Team'].map(get_abbr)
    df_ind['Opponent'] = df_ind['Opponent'].map(get_abbr)

    # numeric outcome & label without value_counts()
    ps = df_ind["Points_scored"].to_numpy()
    pa = df_ind["Points_allowed"].to_numpy()
    p = (ps > pa).astype(float)
    ties = (ps == pa)
    p[ties] = 0.5
    df_ind["Outcome_points"] = p

    outcome = np.empty(len(df_ind), dtype=object)
    outcome[p == 1.0] = "Win"
    outcome[p == 0.5] = "Tie"
    outcome[p == 0.0] = "Loss"
    df_ind["Outcome"] = outcome

    # normalize types
    for c in ["div_game","conf_game"]:
        if c in df_ind.columns:
            df_ind[c] = df_ind[c].astype(bool)

    return df_ind


# -----------------------------
# Internal Caches
# -----------------------------

@dataclass
class _Caches:
    df_ind: pd.DataFrame
    standings_index: pd.Index
    team_to_div: Dict[str, str]

    def __post_init__(self):
        di = self.df_ind

        # Masks
        self._div_mask  = di["div_game"].to_numpy() if "div_game" in di.columns else np.zeros(len(di), dtype=bool)
        self._conf_mask = di["conf_game"].to_numpy() if "conf_game" in di.columns else np.zeros(len(di), dtype=bool)

        # Opponent sets by team
        self.oppsets: Dict[str,set] = (di.groupby("Team")["Opponent"]
                                         .agg(lambda s: set(s.tolist()))
                                         .to_dict())

        # Set of teams defeated (strength of victory)
        self.victory_sets: Dict[str,set] = (
            di.loc[di["Outcome_points"] == 1.0]
              .groupby("Team")["Opponent"]
              .agg(lambda s: set(s.tolist()))
              .to_dict()
        )

        # Mean outcome points head-to-head matrix
        self.h2h = di.pivot_table(index="Team", columns="Opponent",
                                  values="Outcome_points", aggfunc="mean")

        # Overall WLT (mean outcome points)
        self.wlt_all = di.groupby("Team")["Outcome_points"].mean()

        # Division and conference WLT
        if self._div_mask.any():
            self.wlt_div  = di.loc[self._div_mask].groupby("Team")["Outcome_points"].mean()
        else:
            self.wlt_div  = pd.Series(dtype=float)

        if self._conf_mask.any():
            self.wlt_conf = di.loc[self._conf_mask].groupby("Team")["Outcome_points"].mean()
        else:
            self.wlt_conf = pd.Series(dtype=float)

        # Points totals overall & conference
        sums = di.groupby("Team")[["Points_scored","Points_allowed"]].sum()
        self.ps = sums["Points_scored"]
        self.pa = sums["Points_allowed"]

        if self._conf_mask.any():
            conf_sums = di.loc[self._conf_mask].groupby("Team")[["Points_scored","Points_allowed"]].sum()
            self.ps_conf = conf_sums["Points_scored"]
            self.pa_conf = conf_sums["Points_allowed"]
        else:
            self.ps_conf = pd.Series(dtype=float)
            self.pa_conf = pd.Series(dtype=float)

        # Ranks (lower = better). We'll store negative combos when needed.
        self.rank_ps = (-self.ps).rank(method="min")
        self.rank_pa = ( self.pa).rank(method="min")

        # Per-conference ranks
        self.team_to_conf: Dict[str,str] = {t: (self.team_to_div[t][:3] if t in self.team_to_div else "")
                                            for t in self.standings_index}
        self.rank_ps_conf: Dict[str, pd.Series] = {}
        self.rank_pa_conf: Dict[str, pd.Series] = {}

        for conf in {"AFC","NFC"}:
            idx = [t for t in self.standings_index if self.team_to_conf.get(t,"") == conf]
            if idx:
                self.rank_ps_conf[conf] = (-self.ps.reindex(idx)).rank(method="min")
                self.rank_pa_conf[conf] = ( self.pa.reindex(idx)).rank(method="min")
            else:
                self.rank_ps_conf[conf] = pd.Series(dtype=float)
                self.rank_pa_conf[conf] = pd.Series(dtype=float)


# -----------------------------
# Small utilities
# -----------------------------

def _get_conf(team: str, team_to_div: Dict[str,str]) -> str:
    d = team_to_div.get(team, "")
    return d[:3] if d else ""

def analyze_dict(metric: Dict[str, float], teams: Optional[Sequence[str]] = None) -> Optional[str]:
    """Pick the single team with the maximum metric. Return None if still tied/NaN-only.

    metric: mapping team->score (higher is better)
    teams: optional subset; if None, uses metric.keys()
    """
    if teams is None:
        teams = list(metric.keys())
    # limit to teams, drop NaNs
    s = pd.Series(metric, dtype=float).reindex(teams)
    s = s.dropna()
    if s.empty:
        return None
    maxv = s.max()
    winners = s.index[s == maxv].tolist()
    return winners[0] if len(winners) == 1 else None

def get_common(teams: Sequence[str], caches: _Caches) -> set:
    sets = [caches.oppsets.get(t, set()) for t in teams]
    return set.intersection(*sets) if sets else set()

def wlt_vs_subset(team: str, opps: set, caches: _Caches) -> float:
    if not opps:
        return np.nan
    row = caches.h2h.loc[team, list(opps)] if team in caches.h2h.index else None
    if row is None:
        return np.nan
    return float(row.mean(skipna=True))

def strength_of_set(opps: set, caches: _Caches) -> float:
    if not opps:
        return np.nan
    return float(caches.wlt_all.reindex(list(opps)).mean())

def point_diff_over_subset(team: str, opps: set, df_ind: pd.DataFrame) -> float:
    if not opps:
        return np.nan
    sub = df_ind[(df_ind["Team"] == team) & (df_ind["Opponent"].isin(list(opps)))]
    if sub.empty:
        return np.nan
    sums = sub[["Points_scored","Points_allowed"]].sum()
    return float(sums["Points_scored"] - sums["Points_allowed"])


# -----------------------------
# Tiebreakers (Division)
# -----------------------------

def tb_div_h2h(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    if len(teams) == 2:
        a,b = teams
        m = {a: float(caches.h2h.loc[a, b]) if (a in caches.h2h.index and b in caches.h2h.columns) else np.nan,
             b: float(caches.h2h.loc[b, a]) if (b in caches.h2h.index and a in caches.h2h.columns) else np.nan}
        return analyze_dict(m, teams)
    # multi-team: average of H2H round robin (mean of row vs others)
    m = {}
    for t in teams:
        vs = [u for u in teams if u != t and u in caches.h2h.columns and t in caches.h2h.index]
        if not vs:
            m[t] = np.nan
        else:
            m[t] = float(caches.h2h.loc[t, vs].mean(skipna=True))
    return analyze_dict(m, teams)

def tb_div_divWLT(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    m = {t: float(caches.wlt_div.get(t, np.nan)) for t in teams}
    return analyze_dict(m, teams)

def tb_div_common(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    common = get_common(teams, caches)
    m = {t: wlt_vs_subset(t, common, caches) for t in teams}
    return analyze_dict(m, teams)

def tb_div_confWLT(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    m = {t: float(caches.wlt_conf.get(t, np.nan)) for t in teams}
    return analyze_dict(m, teams)

def tb_div_strength_victory(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    m = {t: strength_of_set(caches.victory_sets.get(t, set()), caches) for t in teams}
    return analyze_dict(m, teams)

def tb_div_strength_schedule(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    m = {t: strength_of_set(caches.oppsets.get(t, set()), caches) for t in teams}
    return analyze_dict(m, teams)

def tb_div_points_rank_conf(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    # lower rank number is better; use negative for "higher is better"
    conf = caches.team_to_conf.get(teams[0], "")
    ps_r = caches.rank_ps_conf.get(conf, pd.Series(dtype=float))
    pa_r = caches.rank_pa_conf.get(conf, pd.Series(dtype=float))
    m = {}
    for t in teams:
        if t in ps_r.index and t in pa_r.index:
            m[t] = -float(ps_r[t]) - float(pa_r[t])
        else:
            m[t] = np.nan
    return analyze_dict(m, teams)

def tb_div_points_rank_all(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    m = {t: -float(caches.rank_ps.get(t, np.nan)) - float(caches.rank_pa.get(t, np.nan)) for t in teams}
    return analyze_dict(m, teams)

def tb_div_point_diff_common(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    common = get_common(teams, caches)
    m = {t: point_diff_over_subset(t, common, caches.df_ind) for t in teams}
    return analyze_dict(m, teams)

def tb_div_point_diff_all(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    m = {t: float(caches.ps.get(t, np.nan) - caches.pa.get(t, np.nan)) for t in teams}
    return analyze_dict(m, teams)

DIV_TB_CHAIN = [
    tb_div_h2h,
    tb_div_divWLT,
    tb_div_common,
    tb_div_confWLT,
    tb_div_strength_victory,
    tb_div_strength_schedule,
    tb_div_points_rank_conf,
    tb_div_points_rank_all,
    tb_div_point_diff_common,
    tb_div_point_diff_all,
]


# -----------------------------
# Tiebreakers (Conference / Wildcard)
# -----------------------------

def tb_conf_h2h(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    # average round-robin within tied set
    return tb_div_h2h(teams, caches)

def tb_conf_confWLT(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return tb_div_confWLT(teams, caches)

def tb_conf_common4(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    # only if each pair has at least 4 common opponents; approximate via intersection size
    common = get_common(teams, caches)
    if len(common) < 4:
        return None
    m = {t: wlt_vs_subset(t, common, caches) for t in teams}
    return analyze_dict(m, teams)

def tb_conf_strength_victory(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return tb_div_strength_victory(teams, caches)

def tb_conf_strength_schedule(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return tb_div_strength_schedule(teams, caches)

def tb_conf_points_rank_conf(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return tb_div_points_rank_conf(teams, caches)

def tb_conf_points_rank_all(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return tb_div_points_rank_all(teams, caches)

def tb_conf_point_diff_common(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return tb_div_point_diff_common(teams, caches)

def tb_conf_point_diff_all(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return tb_div_point_diff_all(teams, caches)

CONF_TB_CHAIN = [
    tb_conf_h2h,
    tb_conf_confWLT,
    tb_conf_common4,
    tb_conf_strength_victory,
    tb_conf_strength_schedule,
    tb_conf_points_rank_conf,
    tb_conf_points_rank_all,
    tb_conf_point_diff_common,
    tb_conf_point_diff_all,
]


# -----------------------------
# Division winners, ranks, playoffs
# -----------------------------

def _best_by_chain(teams: Sequence[str], chain: Sequence, caches: _Caches) -> Optional[str]:
    for fn in chain:
        pick = fn(teams, caches)
        if pick is not None:
            return pick
    return None

def _group_by_division(teams: Iterable[str], team_to_div: Dict[str,str]) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {}
    for t in teams:
        d = team_to_div.get(t, "")
        g.setdefault(d, []).append(t)
    return g

def _group_by_conference(teams: Iterable[str], team_to_div: Dict[str,str]) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {}
    for t in teams:
        c = _get_conf(t, team_to_div)
        g.setdefault(c, []).append(t)
    return g

def _rank_group(teams: List[str], base_metric: pd.Series, chain: Sequence, caches: _Caches) -> List[str]:
    # sort by base metric desc, then break equal chunks by chain
    df = pd.DataFrame({"team": teams, "base": base_metric.reindex(teams)}).sort_values("base", ascending=False)
    ranked: List[str] = []
    i = 0
    while i < len(df):
        val = df.iloc[i]["base"]
        # slice of ties (same base)
        j = i
        tied = [df.iloc[j]["team"]]
        j += 1
        while j < len(df) and (
            (pd.isna(val) and pd.isna(df.iloc[j]["base"])) 
            or (df.iloc[j]["base"] == val)
        ):
            tied.append(df.iloc[j]["team"])
            j += 1


        if len(tied) == 1:
            ranked.append(tied[0])
        else:
            # iterative selection among tied using chain (like stable ranking)
            pool = tied[:]
            while pool:
                pick = _best_by_chain(pool, chain, caches)
                if pick is None:
                    # stable alphabetical fallback to keep deterministic
                    pool.sort()
                    ranked.extend(pool)
                    pool = []
                else:
                    ranked.append(pick)
                    pool.remove(pick)
        i = j
    return ranked

def get_div_winners(df_ind: pd.DataFrame, standings: pd.DataFrame, team_to_div: Dict[str,str], caches: _Caches) -> Dict[str, List[str]]:
    winners: Dict[str, List[str]] = {}
    by_div = _group_by_division(standings.index, team_to_div)
    for div, teams in by_div.items():
        base = standings["WLT"]
        ranked = _rank_group(list(teams), base, DIV_TB_CHAIN, caches)
        winners[div] = ranked[:1]  # top 1 is the winner
    return winners

def rank_within_divs(df_ind: pd.DataFrame, standings: pd.DataFrame, team_to_div: Dict[str,str], caches: _Caches) -> Dict[str, List[str]]:
    ranks: Dict[str, List[str]] = {}
    by_div = _group_by_division(standings.index, team_to_div)
    for div, teams in by_div.items():
        base = standings["WLT"]
        ranks[div] = _rank_group(list(teams), base, DIV_TB_CHAIN, caches)
    return ranks

def rank_div_winners(div_winners: Dict[str, List[str]], standings: pd.DataFrame, team_to_div: Dict[str,str], caches: _Caches) -> Dict[str, List[str]]:
    # For each conference, take 4 division winners and rank them
    by_conf: Dict[str, List[str]] = {}
    for div, lst in div_winners.items():
        if not lst:
            continue
        t = lst[0]
        by_conf.setdefault(_get_conf(t, team_to_div), []).append(t)

    seeds: Dict[str, List[str]] = {}
    base = standings["WLT"]
    for conf, teams in by_conf.items():
        seeds[conf] = _rank_group(list(teams), base, CONF_TB_CHAIN, caches)
    return seeds

def break_tie_conf(teams: Sequence[str], caches: _Caches) -> Optional[str]:
    return _best_by_chain(list(teams), CONF_TB_CHAIN, caches)

def get_best_record(teams: Sequence[str], standings: pd.DataFrame, caches: _Caches) -> Optional[str]:
    base = standings['WLT'].reindex(list(teams))
    # If both defined and not equal, pick max
    if base.notna().all() and base.iloc[0] != base.iloc[1]:
        return str(base.idxmax())
    # Fallback chain that works cross-conference
    chain = [
        tb_div_points_rank_all,
        tb_div_point_diff_all,
        tb_div_strength_schedule,
        tb_div_strength_victory,
    ]
    pick = _best_by_chain(list(teams), chain, caches)
    if pick is not None:
        return pick
    # Final deterministic fallback
    return sorted([t for t in teams if isinstance(t, str)])[:1][0] if any(isinstance(t, str) for t in teams) else None


# -----------------------------
# Top-level class
# -----------------------------

class Standings:
    def __init__(self, df_scores: pd.DataFrame, team_to_division: dict[str, str] | None = None):
        import os
        if team_to_division is None:
            div_path = os.path.join("data", "divisions.csv")
            if not os.path.exists(div_path):
                raise FileNotFoundError(
                    "team_to_division not provided and 'data/divisions.csv' not found."
                )
            div_df = pd.read_csv(div_path)
            if div_df.shape[1] >= 2:
                team_to_division = dict(zip(div_df.iloc[:, 0], div_df.iloc[:, 1]))
            else:
                team_to_division = div_df.squeeze().to_dict()

        # --- Normalize observed team codes using get_abbr ---
        observed = pd.unique(df_scores[["home_team", "away_team"]].values.ravel())
        normalized = {t: get_abbr(t) for t in observed}

        # Expand team_to_division so it includes normalized keys
        expanded = dict(team_to_division)
        for raw, norm in normalized.items():
            if raw not in expanded and norm in team_to_division:
                expanded[raw] = team_to_division[norm]

        # Final validation
        missing = sorted(t for t in observed if t not in expanded)
        if missing:
            raise ValueError(
                "Unmapped team codes in standings computation: "
                + ", ".join(missing)
                + ". Update 'data/divisions.csv' or extend 'team_to_division'."
            )

        self.team_to_div = expanded

        # --- everything else stays the same ---
        df_ind = make_ind(df_scores)

        agg = (df_ind.groupby("Team")
               .agg(Wins=("Outcome", lambda s: (s == "Win").sum()),
                    Losses=("Outcome", lambda s: (s == "Loss").sum()),
                    Ties=("Outcome", lambda s: (s == "Tie").sum()),
                    Points_scored=("Points_scored","sum"),
                    Points_allowed=("Points_allowed","sum")))

        totals = (agg["Wins"] + agg["Losses"] + agg["Ties"]).replace(0, np.nan)
        agg["WLT"] = (agg["Wins"] + 0.5*agg["Ties"]) / totals

        agg["Division"]   = [self.team_to_div.get(t, "") for t in agg.index]
        agg["Conference"] = [d[:3] if d else "" for d in agg["Division"]]

        self.standings = agg.sort_index()

        caches = _Caches(df_ind=df_ind, standings_index=self.standings.index, team_to_div=self.team_to_div)

        div_ranks = rank_within_divs(df_ind, self.standings, self.team_to_div, caches)
        self.div_ranks = div_ranks

        div_rank_map = {}
        for div, order in div_ranks.items():
            for i, t in enumerate(order, start=1):
                div_rank_map[t] = i
        self.standings["Division_rank"] = pd.Series(div_rank_map)

        winners_by_div = get_div_winners(df_ind, self.standings, self.team_to_div, caches)
        seeded = rank_div_winners(winners_by_div, self.standings, self.team_to_div, caches)

        by_conf_divlists = _group_by_conference(self.standings.index, self.team_to_div)

        wild_cards: Dict[str, List[str]] = {}
        for conf, conf_teams in by_conf_divlists.items():
            already = set(seeded.get(conf, []))
            pool = [t for t in conf_teams if t not in already]
            base = self.standings["WLT"]
            conf_ranked = _rank_group(pool, base, CONF_TB_CHAIN, caches)
            wild_cards[conf] = conf_ranked[:3]

        self.playoffs = {conf: seeded.get(conf, []) + wild_cards.get(conf, [])
                         for conf in by_conf_divlists.keys()}

        top_afc = seeded.get("AFC", [None])[0] if seeded.get("AFC") else None
        top_nfc = seeded.get("NFC", [None])[0] if seeded.get("NFC") else None
        if top_afc and top_nfc:
            self.best_reg_record = get_best_record([top_afc, top_nfc], self.standings, caches)
        else:
            self.best_reg_record = None
