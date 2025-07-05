import pandas as pd
import numpy as np
from numpy.random import default_rng

df = pd.read_csv("schedules/schedule25.csv")
pr_default = pd.read_csv("data/pr.csv", index_col="Team").squeeze()
pr_custom = pd.Series()
teams = sorted(list(set(df["home_team"])))
rng = default_rng()

div_series = pd.read_csv("data/divisions.csv", index_col=0).squeeze()
conf_teams = {}
for conf in ["AFC","NFC"]:
    conf_teams[conf] = [t for t in teams if div_series[t][:3]==conf]

# Key is number of teams left, we count wild-card as 6 because of how it's implemented
name_dct = {
    6: "Wild card",
    4: "Divisional",
    2: "Conference championship"
}

stages = ["Miss playoffs"] + list(name_dct.values()) + ["Lose in Super Bowl", "Win Super Bowl"]

# TODO: Unify this with the season simulation function
# I didn't pass pr in case we want to use other things eventually
def simulate_game(match_dct):
    '''The input should be of the form {"Home": team, "Home_pr": number, "Away": team, "Away_pr": number, "HFA": number}'''
    score_dct = {}
    place_dct = {"Home": "Home_pr", "Away": "Away_pr"}
    for place, place_pr in place_dct.items():
        score_dct[place] = rng.normal(match_dct[place_pr], 10)
        if place == "Home":
            score_dct[place] += match_dct["HFA"]

    if score_dct["Home"] >= score_dct["Away"]:
        return match_dct["Home"]
    else:
        return match_dct["Away"]

def simulate_round(pr, round_seeds, stage_of_elim):
    # Special case because not everyone plays
    if len(round_seeds) == 7:
        # The 1-seed always advances from the wild-card round
        output_dct = {1: round_seeds[1]}
        wc_results, stage_of_elim = simulate_round(pr, {k: v for k, v in round_seeds.items() if k >= 2}, stage_of_elim)
        output_dct.update(wc_results)
        return output_dct, stage_of_elim
    
    output_dct = {}
    seeds = round_seeds.copy()
    name = name_dct[len(round_seeds)]
    while len(seeds) > 0:
        match_dct = {"HFA": pr["HFA"]}
        home_seed = min(seeds)
        home_team = seeds.pop(home_seed)
        match_dct["Home"] = home_team
        match_dct["Home_pr"] = pr[home_team]
        away_seed = max(seeds)
        away_team = seeds.pop(away_seed)
        match_dct["Away"] = away_team
        match_dct["Away_pr"] = pr[away_team]
        
        winner = simulate_game(match_dct)
        if winner == home_team:
            output_dct[home_seed] = round_seeds[home_seed]
            stage_of_elim[round_seeds[away_seed]] = name
        else:
            output_dct[away_seed] = round_seeds[away_seed]
            stage_of_elim[round_seeds[home_seed]] = name

    return output_dct, stage_of_elim
    

def simulate_conference(pr, conf_seeds, stage_of_elim):
    bye_team = conf_seeds[1]
    round_seeds = conf_seeds
    while len(round_seeds) > 1:
        if len(round_seeds) == 2:
            # conf_champ_matchup has the form (bye_team, [conf champ 1, conf champ 2])
            # the conference championship teams are alphabetized
            conf_champ_matchup = (bye_team, tuple(sorted(list(round_seeds.values()))))
        round_seeds, stage_of_elim = simulate_round(pr, round_seeds, stage_of_elim)

    return list(round_seeds.values())[0], stage_of_elim, conf_champ_matchup


def simulate_playoffs(pr, seeds):
    seeds = seeds.copy()
    # Are the seeds missing ranks?
    if isinstance(seeds["AFC"], list):
        for conf, teams in seeds.items():
            seeds[conf] = dict(zip(range(1, 8), teams))
    
    conf_champs = []
    conf_champ_matchups = []
    stage_of_elim = {}
    for conf, conf_seeds in seeds.items():
        stage_of_elim.update({team: "Miss playoffs" for team in conf_teams[conf] if team not in conf_seeds.values()})
        conf_champ, stage_of_elim, conf_champ_matchup = simulate_conference(pr, conf_seeds, stage_of_elim)
        conf_champs.append(conf_champ)
        conf_champ_matchups.append(conf_champ_matchup)

    # We assume no home-field advantage in the Super Bowl
    match_dct = {
                    "Home": conf_champs[0],
                    "Home_pr": pr[conf_champs[0]],
                    "Away": conf_champs[1],
                    "Away_pr": pr[conf_champs[1]],
                    "HFA": 0
        }
    champion = simulate_game(match_dct)
    stage_of_elim[champion] = "Win Super Bowl"
    sb_loser = next(team for team in conf_champs if team != champion)
    stage_of_elim[sb_loser] = "Lose in Super Bowl"
    # We want to know the exact super bowl matchup, listing NFC first
    matchup = f"{champion} vs {sb_loser}" if champion in conf_teams["NFC"] else f"{sb_loser} vs {champion}"
    return stage_of_elim, matchup, conf_champ_matchups
