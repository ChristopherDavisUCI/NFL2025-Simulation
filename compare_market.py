import pandas as pd
import numpy as np
from odds_helper import kelly, odds_to_prob
import streamlit as st


def win_div(raw_data):
    '''Returns a Series indicating the probability of winning the division for each team.'''
    df = raw_data.set_index("Team", drop=True)
    df = df[df["Seed"] == 4].copy()
    return df["Equal_better"] 


def make_playoffs(raw_data):
    '''Returns a Series indicating the probability of making the playoffs for each team.'''
    df = raw_data.set_index("Team", drop=True)
    # Just want each team one time.  Could just as well say 7.
    df = df[df["Seed"] == 1].copy()
    return df["Make_playoffs"]


def get_prob(row, prob_dct):
    if row["raw_market"] == "division":
        ser = prob_dct["div"]
        return ser[row["team"]]
    elif (row["raw_market"] == "make playoffs") and (row["result"] == "Yes"):
        ser = prob_dct["mp"]
        return ser[row["team"]]
    elif (row["raw_market"] == "make playoffs") and (row["result"] == "No"):
        ser = prob_dct["mp"]
        return 1-ser[row["team"]]
    elif row["raw_market"] == "conference":
        ser = prob_dct["conf"]
        return ser[row["team"]]
    elif row["raw_market"] == "super bowl":
        ser = prob_dct["sb"]
        return ser[row["team"]]
    elif row["raw_market"] == "most wins":
        ser = prob_dct["most wins"]
        return ser[row["team"]]
    elif row["raw_market"] == "last undefeated":
        ser = prob_dct["undefeated"]
        return ser.get(row["team"], 0)
    elif row["raw_market"] == "last winless":
        ser = prob_dct["winless"]
        return ser.get(row["team"], 0)
    elif row["raw_market"] == "exact matchup":
        # this isn't very robust, because we are assuming NFC team is first
        # and that they are written exactly the same
        return prob_dct["matchup"].get(row["team"], 0)
    elif (row["raw_market"] == "stage") and (row["result"] in prob_dct):
        ser = prob_dct[row["result"]]
        return ser.get(row["team"], 0)

def name_market(row):
    if row["raw_market"] == "division":
        return "Win division"
    elif (row["raw_market"] == "make playoffs") and (row["result"] == "Yes"):
        return "Make playoffs - Yes"
    elif (row["raw_market"] == "make playoffs") and (row["result"] == "No"):
        return "Make playoffs - No"
    elif row["raw_market"] == "conference":
        return "Conference Champion"
    elif row["raw_market"] == "super bowl":
        return "Super Bowl Champion"
    elif row["raw_market"] == "most wins":
        return "Best Record"
    elif row["raw_market"] == "last undefeated":
        return "Last undefeated team"
    elif row["raw_market"] == "last winless":
        return "Last winless team"
    elif row["raw_market"] == "exact matchup":
        return "Exact Super bowl matchup"
    if row["raw_market"] == "stage":
        return f"Stage of Elim - {row['result']}"
    

def display_plus(s):
    if s[0] == "-":
        return s
    else:
        return "+"+s


def make_stage_series(champ_data, stage):
    '''Returns a Series indicating the probability of each team being eliminated in that stage.'''
    df = champ_data.set_index("Team", drop=True)
    df = df[df["Stage"] == stage].copy()
    return df["Proportion"] 


def matchup_prob(matchup_list):
    n = len(matchup_list)
    matchup_dct = {}
    for m in matchup_list:
        if m in matchup_dct.keys():
            matchup_dct[m] += 1/n
        else:
            matchup_dct[m] = 1/n
    return pd.Series(matchup_dct)

# raw_data and champ_data are about our predicted numbers
# Columns for raw_data are:
# Seed, Team, Proportion, Make_playoffs, Equal_better
# Odds, Odds_Make_playoffs, Odds_Equal_better
# Columns for champ_data are:
# Stage, Team, Proportion, Odds
# pivot_all has Team as index and columns like "Best Record", "Last undefeated", "Last winless"
# entries in pivot_all are probabilities
# matchup_list is a list of the different super bowl exact matchups, listed with NFC team first
def compare_market(raw_data, champ_data, pivot_all, matchup_list):
    market = pd.read_csv("data/markets.csv")
    market = market[(market["odds"].notna()) & (market["team"].notna())].copy()
    market.rename({"market": "raw_market"}, axis=1, inplace=True)
    ser_div = win_div(raw_data)
    ser_mp = make_playoffs(raw_data)
    # Keys are stage of elimination
    # Values are a Series with index teams, values probabilities of being eliminated in that stage
    stage_dct = {}
    for stage in champ_data["Stage"].unique():
        stage_dct[stage] = make_stage_series(champ_data, stage)
    
    ser_sb = stage_dct["Win Super Bowl"]
    ser_lose = stage_dct["Lose in Super Bowl"]
    # Winning the Conf = Win SB or Lose SB
    ser_conf = ser_sb + ser_lose
    ser_matchup = matchup_prob(matchup_list)
    prob_dct = {
        "div": ser_div,
        "mp": ser_mp,
        "sb": ser_sb,
        "conf": ser_conf,
        "most wins": pivot_all["Best Record"],
        "undefeated": pivot_all["Last undefeated"],
        "winless": pivot_all["Last winless"],
        "matchup": ser_matchup
    }
    prob_dct.update(stage_dct)
    market["prob"] = market.apply(lambda row: get_prob(row, prob_dct), axis=1)
    market["market"] = market.apply(name_market, axis=1)
    market["kelly"] = market.apply(lambda row: kelly(row["prob"], row["odds"]), axis=1)
    rec = market[market["kelly"] > 0].sort_values("kelly", ascending=False)
    rec["odds"] = rec["odds"].astype(str).map(display_plus)
    return rec[["team", "market", "odds", "prob", "site", "kelly"]].reset_index(drop=True)
