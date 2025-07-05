import pandas as pd

import sim_playoffs
from sim_playoffs import simulate_playoffs
from sim_season import simulate_reg_season
from make_standings import Standings, make_ind, get_div_winners
df = pd.read_csv("schedules/schedule23.csv")

pr_default = pd.read_csv("data/pr.csv", index_col="Team").squeeze()

div_series = pd.read_csv("data/divisions.csv", index_col=0).squeeze()
teams = div_series.index
conf_teams = {}
for conf in ["AFC","NFC"]:
    conf_teams[conf] = [t for t in teams if div_series[t][:3]==conf]

df = simulate_reg_season(pr_default)
df_ind = make_ind(df)
stand = Standings(df)
standings = stand.standings

while standings.loc["DAL", "Wins"] != standings.loc["PHI", "Wins"]:
    df = simulate_reg_season(pr_default)
    df_ind = make_ind(df)
    stand = Standings(df)
    standings = stand.standings

get_div_winners(df_ind, stand.standings)