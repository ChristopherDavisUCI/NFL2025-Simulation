import pandas as pd
import numpy as np
from numpy.random import default_rng

df = pd.read_csv("schedules/schedule24.csv")
df_stored = df.copy()
pr_default = pd.read_csv("data/pr.csv", index_col="Team").squeeze()
pr_custom = pd.Series()
teams = sorted(list(set(df["home_team"])))
rng = default_rng()

def simulate_reg_season(pr = pr_default, df_stored=df_stored):
    try:
        df["pr_home_Off"] = df.home_team.map(lambda s: pr_custom[s+"_Off"])
        df["pr_home_Def"] = df.home_team.map(lambda s: pr_custom[s+"_Def"])
        df["pr_away_Off"] = df.away_team.map(lambda s: pr_custom[s+"_Off"])
        df["pr_away_Def"] = df.away_team.map(lambda s: pr_custom[s+"_Def"])
        df["mean_home"] = df["pr_home_Off"]-df["pr_away_Def"]+pr_custom["mean_score"]+pr["HFA"]/2
        df["mean_away"] = df["pr_away_Off"]-df["pr_home_Def"]+pr_custom["mean_score"]-pr["HFA"]/2
    except:
        make_pr_custom(pr)
        df["pr_home_Off"] = df["home_team"].map(lambda s: pr_custom[s+"_Off"])
        df["pr_home_Def"] = df["home_team"].map(lambda s: pr_custom[s+"_Def"])
        df["pr_away_Off"] = df["away_team"].map(lambda s: pr_custom[s+"_Off"])
        df["pr_away_Def"] = df["away_team"].map(lambda s: pr_custom[s+"_Def"])
        df["mean_home"] = df["pr_home_Off"]-df["pr_away_Def"]+pr_custom["mean_score"]+pr["HFA"]/2
        df["mean_away"] = df["pr_away_Off"]-df["pr_home_Def"]+pr_custom["mean_score"]-pr["HFA"]/2
    scores = ["home_score","away_score"]
    df.loc[:,scores] = rng.normal(df[["mean_home","mean_away"]], 10)
    df.loc[:,scores] = df.loc[:,scores].round()
    df[scores] = df[scores].mask(df[scores] < 0, 0)
    # Games already played, put back in
    played_boolean = df_stored["home_score"].notna()
    adjust_ties(df)
    df.loc[played_boolean, scores] = df_stored.loc[played_boolean, scores]
    # We don't want to use playoff results in computing the standings
    return df[df["game_type"] == "REG"]

def make_pr_custom(pr):
    global pr_custom
    for s in teams:
        if s in pr.keys():
            pr_custom[s+"_Off"] = pr[s]/2
            pr_custom[s+"_Def"] = pr[s]/2
        else:
            pr_custom[s+"_Off"] = pr_default[s+"_Off"]
            pr_custom[s+"_Def"] = pr_default[s+"_Def"]
    for k in ["HFA", "mean_score"]:
        if k in pr.keys():
            pr_custom[k] = pr[k]
        else:
            pr_custom[k] = pr_default[k]
    return pr_custom

def adjust_ties(df):
    tied_games = df.loc[df.home_score == df.away_score,["home_score","away_score"]].copy()
    x = rng.normal(size=len(tied_games))
    tied_games.iloc[np.where((x > -1.6) & (x < 0))[0]] += (0,3)
    tied_games.iloc[np.where(x >= 0)[0]] += (3,0)
    df.loc[tied_games.index,["home_score","away_score"]] = tied_games
    return None