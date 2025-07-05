import pandas as pd
import numpy as np

def outcome_dct(row):
    if row["home_score"] > row["away_score"]:
        winner = row["home_team"]
        loser = row["away_team"]
    elif row["away_score"] > row["home_score"]:
        winner = row["away_team"]
        loser = row["home_team"]
    else:
        # Tie game, no winner or loser
        return {}
    
    return {"winner": winner, "loser": loser}


def get_results(df):
    df = df.copy()
    df[["winner", "loser"]] = pd.DataFrame(list(df.apply(outcome_dct, axis=1)))
    return df


def get_streaks(df):
    df = get_results(df)

    streak_dct = {team: 0 for team in set(df["away_team"])}

    for team in streak_dct.keys():
        arr = np.nonzero(df[(df[["winner", "loser"]] == team).any(axis=1)]["winner"] == team)[0]
        if len(arr) == 0:
            streak_dct[team] = 0
            continue
        
        arr = arr + 1 # Just so they are week numbers rather than index numbers
        idx = np.nonzero(np.diff(arr) != 1)[0]
        start_idx = np.concatenate(([0], idx + 1)) # I don't really understand, Bing chat
        end_idx = np.concatenate((idx, [len(arr) - 1]))
        run_lengths = end_idx - start_idx + 1
        
        streak_dct[team] = run_lengths.max()

    return streak_dct


def get_last(df):
    df = get_results(df)
    
    # The RHS is just a quick way to get all teams
    winless = set(df["away_team"])
    undefeated = set(df["away_team"])
    
    for i in range(1,19):
        df_week = df[df["week"] == i]
        
        winners = set(df_week["winner"])
        temp = winless.difference(winners)
        if len(temp) == 0:
            last_winless = winless
            break
        else:
            winless = temp

        if i == 18:
            last_winless = winless
            

        
    for i in range(1,19):
        df_week = df[df["week"] == i]
        
        losers = set(df_week["loser"])
        temp = undefeated.difference(losers)
        if len(temp) == 0:
            last_undefeated = undefeated
            break
        else:
            undefeated = temp

        if i == 18:
            last_undefeated = undefeated
            
    return {"last_undefeated": tuple(last_undefeated), "last_winless": tuple(last_winless)}

