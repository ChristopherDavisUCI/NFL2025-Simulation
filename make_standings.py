import numpy as np
import pandas as pd
from math import isclose

div_series = pd.read_csv("data/divisions.csv", index_col=0).squeeze()
div_series.name = None
teams = sorted(list(div_series.index))
tol = .0001



reverse_dict = {'Win':'Loss','Loss':'Win', 'Tie':'Tie'}

def get_outcome(row):
    s = row["Points_scored"]
    a = row["Points_allowed"]
    if s > a:
        return "Win"
    elif s < a:
        return "Loss"
    elif s == a:
        return "Tie"

def get_WLT(games):
    s = games.Outcome.value_counts()
    try:
        return (s.get("Win",0) + 0.5*s.get("Tie",0))/len(games)
    except:
        if len(games) == 0:
            return 0
        else:
            raise

def get_strength(teams, df_ind):
    return get_WLT(df_ind[df_ind["Team"].isin(teams)])

def get_victories(team, df_ind):
    return set(df_ind["Opponent"][(df_ind["Team"] == team) & (df_ind["Outcome"] == "Win")])

def get_opps(team, df_ind):
    return set(df_ind["Opponent"][df_ind["Team"] == team])

def get_common(teams, df_ind):
    opps = []
    for t in teams:
        opps.append(get_opps(t, df_ind))
    return set.intersection(*opps)

# This seems to assume teams are either all in the same division
# or all in different divisions.
def analyze_dict(d, df_ind, df_standings):
    outs = sorted(d.values(),reverse=True)
    if isclose(outs[0],outs[-1],abs_tol=tol):
        return None
    top_teams = [k for k in d.keys() if isclose(d[k],outs[0],abs_tol = tol)]
    if len(top_teams) == 1:
        return top_teams[0]
    if div_series[top_teams[0]] == div_series[top_teams[1]]:
        return break_tie_div(top_teams,df_ind,df_standings)
    else:
        return break_tie_conf(top_teams,df_ind,df_standings)

break_tie_fns = {}

# Are there any numerical precision problems here?
def fd21(teams,df_ind,df_standings):
    wlt_dict = {teams[0]:get_WLT(df_ind[(df_ind["Team"] == teams[0]) & (df_ind["Opponent"] == teams[1])])}
    wlt_dict[teams[1]] = get_WLT(df_ind[(df_ind["Team"] == teams[1]) & (df_ind["Opponent"] == teams[0])])
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("div",2,1)] = fd21

def fd22(teams,df_ind,df_standings):
    div_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.div_game)]) for t in teams}
    return analyze_dict(div_dict, df_ind, df_standings)

break_tie_fns[("div",2,2)] = fd22

def fd23(teams,df_ind,df_standings):
    common = get_common(teams,df_ind)
    common_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.Opponent.isin(common))]) for t in teams}
    return analyze_dict(common_dict, df_ind, df_standings)

break_tie_fns[("div",2,3)] = fd23

def fd24(teams,df_ind,df_standings):
    conf_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.conf_game)]) for t in teams}
    return analyze_dict(conf_dict, df_ind, df_standings)

break_tie_fns[("div",2,4)] = fd24

def fd25(teams,df_ind,df_standings):
    wlt_dict = {t: get_strength(get_victories(t,df_ind),df_ind) for t in teams}
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("div",2,5)] = fd25
break_tie_fns[("div",3,5)] = fd25

def fd26(teams,df_ind,df_standings):
    wlt_dict = {t: get_strength(get_opps(t,df_ind),df_ind) for t in teams}
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("div",2,6)] = fd26
break_tie_fns[("div",3,6)] = fd26

# The negatives are because we want "lower rank" to correspond to higher numbers
# Maybe would be better to have a reverse flag in the dictionary
def fd27(teams,df_ind,df_standings):
    conf = div_series[teams[0]][:3]
    ps_series = (-df_standings.query("Conference==@conf").Points_scored).rank(method="min")
    pa_series = df_standings.query("Conference==@conf").Points_allowed.rank(method="min")
    rank_dict = {t: -ps_series[t] - pa_series[t] for t in teams}
    return analyze_dict(rank_dict, df_ind, df_standings)

break_tie_fns[("div",2,7)] = fd27
break_tie_fns[("div",3,7)] = fd27

# The negatives are because we want "lower rank" to correspond to higher numbers
def fd28(teams,df_ind,df_standings):
    ps_series = (-df_standings.Points_scored).rank(method="min")
    pa_series = df_standings.Points_allowed.rank(method="min")
    rank_dict = {t: -ps_series[t] - pa_series[t] for t in teams}
    return analyze_dict(rank_dict, df_ind, df_standings)

break_tie_fns[("div",2,8)] = fd28
break_tie_fns[("div",3,8)] = fd28

def fd29(teams,df_ind,df_standings):
    common = get_common(teams,df_ind)
    pt_dict = {}
    for t in teams:
        df_common = df_ind[["Points_scored","Points_allowed"]][(df_ind.Team == t) & (df_ind.Opponent.isin(common))].sum()
        pt_dict[t] = df_common["Points_scored"]-df_common["Points_allowed"]
    return analyze_dict(pt_dict, df_ind, df_standings)

break_tie_fns[("div",2,9)] = fd29
break_tie_fns[("div",3,9)] = fd29

# Would it be better to use df_standings for this?
def fd210(teams,df_ind,df_standings):
    pt_dict = {}
    for t in teams:
        df_common = df_ind[["Points_scored","Points_allowed"]][df_ind.Team == t].sum()
        pt_dict[t] = df_common["Points_scored"]-df_common["Points_allowed"]
    return analyze_dict(pt_dict, df_ind, df_standings)

break_tie_fns[("div",2,10)] = fd210
break_tie_fns[("div",3,10)] = fd210

# How does the coin toss work for three teams?
def fd212(teams,df_ind,df_standings):
    return np.random.choice(teams)

break_tie_fns[("div",2,12)] = fd212
break_tie_fns[("div",3,12)] = fd212

def fd31(teams,df_ind,df_standings):
    df = df_ind[df_ind["Team"].isin(teams) & df_ind["Opponent"].isin(teams)]
    wlt_dict = {t: get_WLT(df[df.Team == t]) for t in teams}
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("div",3,1)] = fd31

def fd32(teams,df_ind,df_standings):
    wlt_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.div_game)]) for t in teams}
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("div",3,2)] = fd32

def fd33(teams,df_ind,df_standings):
    opps = []
    for t in teams:
        opps.append(get_opps(t,df_ind))
    common = set.intersection(*opps)
    wlt_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.Opponent.isin(common))]) for t in teams}
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("div",3,3)] = fd33

def fd34(teams,df_ind,df_standings):
    wlt_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.conf_game)]) for t in teams}
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("div",3,4)] = fd34

def fc21(teams,df_ind,df_standings):
    df_head = df_ind[(df_ind["Team"] == teams[0]) & (df_ind["Opponent"] == teams[1])]
    if len(df_head) == 0:
        return None
    wlt_dict = {teams[0]:get_WLT(df_ind[(df_ind["Team"] == teams[0]) & (df_ind["Opponent"] == teams[1])])}
    wlt_dict[teams[1]] = get_WLT(df_ind[(df_ind["Team"] == teams[1]) & (df_ind["Opponent"] == teams[0])])
    return analyze_dict(wlt_dict, df_ind, df_standings)

break_tie_fns[("conf",2,1)] = fc21

def fc23(teams,df_ind,df_standings):
    common = get_common(teams,df_ind)
    if len(common) < 4:
        return None
    common_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.Opponent.isin(common))]) for t in teams}
    return analyze_dict(common_dict, df_ind, df_standings)

break_tie_fns[("conf",2,3)] = fc23

def fc28(teams,df_ind,df_standings):
    pt_dict = {}
    for t in teams:
        df_conf = df_ind[["Points_scored","Points_allowed"]][(df_ind.Team == t) & (df_ind.conf_game)].sum()
        pt_dict[t] = df_conf["Points_scored"]-df_conf["Points_allowed"]
    return analyze_dict(pt_dict, df_ind, df_standings)

break_tie_fns[("conf",2,8)] = fc28

break_tie_fns[("conf",2,2)] = break_tie_fns[("div",2,4)]
break_tie_fns[("conf",2,4)] = break_tie_fns[("div",2,5)]
break_tie_fns[("conf",2,5)] = break_tie_fns[("div",2,6)]
break_tie_fns[("conf",2,6)] = break_tie_fns[("div",2,7)]
break_tie_fns[("conf",2,7)] = break_tie_fns[("div",2,8)]
break_tie_fns[("conf",2,9)] = break_tie_fns[("div",2,10)]
break_tie_fns[("conf",2,11)] = break_tie_fns[("div",2,12)]

def find_sweep(teams, df_ind, df_standings):
    for t in teams:
        others = [x for x in teams if x != t]
        outcomes = []
        temp_df = df_ind[(df_ind["Team"] == t) & (df_ind["Opponent"].isin(others))]
        outcomes += list(temp_df["Outcome"])
        if len(outcomes) >= len(others): # make sure teams played
            if set(outcomes) == {'Win'}:
                return t
            elif set(outcomes) == {'Loss'}:
                return break_tie_conf(others,df_ind,df_standings)
    return None

break_tie_fns[("conf",3,2)] = find_sweep
break_tie_fns[("conf",3,3)] = break_tie_fns[("div",3,4)]
break_tie_fns[("conf",3,4)] = break_tie_fns[("conf",2,3)]
break_tie_fns[("conf",3,5)] = break_tie_fns[("div",2,5)]
break_tie_fns[("conf",3,6)] = break_tie_fns[("div",2,6)]
break_tie_fns[("conf",3,7)] = break_tie_fns[("div",2,7)]
break_tie_fns[("conf",3,8)] = break_tie_fns[("div",2,8)]
break_tie_fns[("conf",3,9)] = break_tie_fns[("conf",2,8)]
break_tie_fns[("conf",3,10)] = break_tie_fns[("div",2,10)]
break_tie_fns[("conf",3,12)] = break_tie_fns[("div",2,12)]

def break_tie_div(teams,df_ind,df_standings):
    scenario = 2 if len(teams) == 2 else 3
    rules = sorted([c for (a,b,c) in break_tie_fns.keys() if a == 'div' and b == scenario])
    for c in rules:
        t = break_tie_fns[("div",scenario,c)](teams,df_ind,df_standings)
        if t is not None:
            return t

def break_tie_conf(teams,df_ind,df_standings):
    scenario = 2 if len(teams) == 2 else 3
    rules = sorted([c for (a,b,c) in break_tie_fns.keys() if a == 'conf' and b == scenario])
    for c in rules:
        t = break_tie_fns[("conf",scenario,c)](teams,df_ind,df_standings)
        #print(("conf",scenario,c))
        if t is not None:
            return t

# Any numerical precision problems here?
def get_div_winners(df_ind,df_standings):
    winner_dict = {}
    divs = sorted(list(set(df_standings.Division)))
    for div in divs:
        df = df_standings[df_standings["Division"] == div].sort_values("WLT",ascending=False).copy()
        t = analyze_dict(dict(zip(df.index,df.WLT)),df_ind,df_standings)
        if t is None:
            t = break_tie_div(list(df.index),df_ind,df_standings)
        winner_dict[div] = t
    return winner_dict

def rank_div_winners(dw,df_ind,df_standings):
    playoffs = {}
    for conf in ["AFC","NFC"]:
        playoffs[conf] = []
        teams = [dw[x] for x in dw.keys() if x[:3] == conf]
        while len(playoffs[conf]) < 3:
            df = df_standings[df_standings["Team"].isin(teams)].sort_values("WLT",ascending=False).copy()
            t = analyze_dict(dict(df.loc[teams,"WLT"]), df_ind, df_standings)
            if t is None:
                t = break_tie_conf(teams,df_ind,df_standings)
            playoffs[conf].append(t)
            try:
                teams.remove(t)
            except:
                pass
        playoffs[conf].append(teams[0])
    return playoffs

def rank_within_divs(dw,df_ind,df_standings):
    dr = {}
    for div in dw.keys():
        dr[div] = []
        df = df_standings[(df_standings["Division"] == div) & ~(df_standings["Team"] == dw[div])].sort_values("WLT",ascending=False).copy()
        teams = list(df.index)
        while len(dr[div]) < 2:
            t = analyze_dict(dict(df.loc[teams,"WLT"]), df_ind, df_standings)
            if t is None:
                t = break_tie_div(teams,df_ind,df_standings)
            dr[div].append(t)
            teams.remove(t)
        dr[div].append(teams[0])
        dr[div] = [dw[div]]+dr[div]
    return dr


# Should probably rewrite this to be more like break_tie_conf
def get_best_record(teams, df_ind, df_standings):
    df = df_standings[df_standings["Team"].isin(teams)].sort_values("WLT",ascending=False).copy()
    t = analyze_dict(dict(df.loc[teams,"WLT"]), df_ind, df_standings)
    if t:
        return t
    t = find_sweep(teams, df_ind, df_standings)
    if t:
        return t
    common = get_common(teams,df_ind)
    df_sub_list = [df_ind[(df_ind.Team == t) & (df_ind.Opponent.isin(common))] for t in teams]
    if min([len(df_sub) for df_sub in df_sub_list]) >= 4:
        common_dict = {t: get_WLT(df_ind[(df_ind.Team == t) & (df_ind.Opponent.isin(common))]) for t in teams}
        t = analyze_dict(common_dict, df_ind, df_standings)
        if t:
            return t
    wlt_dict = {t: get_strength(get_victories(t,df_ind),df_ind) for t in teams}
    t = analyze_dict(wlt_dict, df_ind, df_standings)
    if t:
        return t
    # Not finished, we return a random choice if not yet found
    return np.random.choice(teams)
    


def make_ind(df):
    df_ind = pd.DataFrame(index=range(2*len(df)),columns=["Team","Opponent","Points_scored","Points_allowed","Outcome","div_game","conf_game"])
    cols = list(df_ind.columns)
    inds = [cols.index(col) for col in ["Team","Opponent","div_game","conf_game"]]
    df_ind.iloc[::2,inds] = df[["home_team","away_team","div_game","conf_game"]]
    df_ind.iloc[1::2,inds] = df[["away_team","home_team","div_game","conf_game"]]
    scores = df[["home_score","away_score"]].apply(tuple,axis=1).to_list()
    df_ind.loc[::2,["Points_scored","Points_allowed"]] = scores
    df_ind.loc[1::2,["Points_allowed","Points_scored"]] = scores
    df_ind.Outcome = df_ind.apply(get_outcome,axis=1)
    return df_ind

class Standings:
    df_standings = pd.DataFrame(index=sorted(teams,key=lambda t: div_series[t]),columns=["Team","Wins","Losses","Ties","Points_scored","Points_allowed","WLT","Division","Conference"])
    df_standings.Team = df_standings.index
    df_standings.Division = df_standings.index.map(lambda t: div_series[t])
    df_standings.Conference = df_standings.index.map(lambda t: div_series[t][:3])
    standings = df_standings.copy()
    
    def __init__(self,df_scores):
        if "schedule_playoff" in df_scores.columns:
            df_scores = df_scores.loc[~df_scores["schedule_playoff"]].copy()
        df_ind = make_ind(df_scores)
        for a,b in df_ind.groupby("Team"):
            res = b.Outcome.value_counts()
            self.standings.loc[a,["Wins","Losses","Ties","Points_scored","Points_allowed"]] = [res.get("Win",0), res.get("Loss",0), res.get("Tie",0),
                b["Points_scored"].sum(),b["Points_allowed"].sum()]
        self.standings["WLT"] = (self.standings["Wins"]+0.5*self.standings["Ties"])/(self.standings["Wins"]+self.standings["Ties"]+self.standings["Losses"])
        dw_unranked = get_div_winners(df_ind,self.standings)
        dw = rank_div_winners(dw_unranked, df_ind, self.standings)
        self.div_ranks = rank_within_divs(dw_unranked,df_ind,self.standings)
        self.standings["Division_rank"] = self.standings.apply(lambda x: self.div_ranks[x["Division"]].index(x.name)+1, axis=1)
        self.standings = self.standings.sort_values(["Division", "Division_rank"])
        wild_cards = {}
        div_eligible = {k:self.div_ranks[k][1:] for k in self.div_ranks.keys()}
        for conf in ["AFC","NFC"]:
            wild_cards[conf] = []
            while len(wild_cards[conf]) < 3:
                top_teams = [div_eligible[x][0] for x in div_eligible.keys() if x[:3] == conf]
                t = analyze_dict(dict(self.standings.loc[top_teams,"WLT"]), df_ind, self.standings)
                if t is None:
                    t = break_tie_conf(top_teams, df_ind, self.standings)
                wild_cards[conf].append(t)
                try:
                    div_eligible[div_series[t]].remove(t)
                except:
                    pass
        self.playoffs = {conf:dw[conf] + wild_cards[conf] for conf in ["AFC","NFC"]}
        self.best_reg_record = get_best_record([dw["AFC"][0], dw["NFC"][0]], df_ind, self.standings)
