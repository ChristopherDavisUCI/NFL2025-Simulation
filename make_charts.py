import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()

df = pd.read_csv("schedules/schedule25.csv")
div_series = pd.read_csv("data/divisions.csv", index_col=0).squeeze()
teams = sorted(list(set(df["home_team"])))
conf_teams = {}
for conf in ["AFC","NFC"]:
    conf_teams[conf] = [t for t in teams if div_series[t][:3]==conf]

def prob_to_odds(p):
    if p < .000001:
        return "NA"
    if p > .999999:
        return "NA"
    if p > 0.5:
        x = 100*p/(p-1)
        return f"{x:.0f}"
    elif p <= 0.5:
        x = 100*(1-p)/p
        return f"+{x:.0f}"

def make_playoff_charts(total_dict):

    reps = sum(total_dict["AFC"][1].values())

    odds_dict = {"Proportion":"Odds", "Make_playoffs": "Odds_Make_playoffs", "Equal_better":"Odds_Equal_better"}

    # Trying to match the Tableau10 colors in the order I want.
    #color_dict = dict(zip(["purple","yellow","green","teal","red","orange","blue"],["#b07aa1","#EDC948","#59A14F","#76B7B2","#E15759","#F28E2B","#4E79A7"]))
    color_dict = {"blue": "#5778a4", "orange": "#e49444", "red": "#d1615d", "teal": "#85b6b2", "green": "#6a9f58", "yellow": "#e7ca60", "purple": "#a87c9f",}

    chart_dict = {}

    # The following will hold both conference DataFrames, which we will then concatenate.
    source_list = []

    for conf in ["AFC","NFC"]:

        playoff_dicts = total_dict[conf]
        
        source = pd.DataFrame([(k,t,playoff_dicts[k][t]/reps) for k in playoff_dicts.keys() for t in conf_teams[conf]],
            columns = ["Seed","Team","Proportion"])
        
        for a,b in source.groupby("Team"):
            source.loc[source["Team"] == a, "Make_playoffs"] = b.Proportion.sum()
            source.loc[source["Team"] == a,"Equal_better"] = b.Proportion.cumsum()

        for c in odds_dict.keys():
            source[odds_dict[c]] = source[c].map(prob_to_odds)

        source_list.append(source.copy())

        # Sort first on probability of making playoffs
        # then on 1 seed, then on 1 seed + 2 seed probability, etc
        ordering = sorted(conf_teams[conf],key=lambda t: (sum([playoff_dicts[i][t] for i in playoff_dicts.keys()]), playoff_dicts[1][t], playoff_dicts[1][t]+playoff_dicts[2][t]), reverse=True)
        ordering_seeds = list(range(7,0,-1))


        c = alt.Chart(source).mark_bar().encode(
            x=alt.X('Team',sort = ordering),
            y=alt.Y('Proportion',scale=alt.Scale(domain=[0,1]),sort=ordering_seeds),
            tooltip = [alt.Tooltip('Team'),alt.Tooltip('Seed', format=".0f"), 
                        #alt.Tooltip('Proportion', format=".3f"),
                        alt.Tooltip("prob_odds:N",title="Proportion"),
                        alt.Tooltip("equal_odds:N",title="Equal or better"),
                        alt.Tooltip("playoffs_odds:N",title="Make playoffs")],
            color=alt.Color('Seed:N', scale=alt.Scale(domain=[7,6,5,4,3,2,1],range=[color_dict[color] for color in ["purple","yellow","green","teal","red","orange","blue"]])),
            order=alt.Order(
                    'Seed:N',
                    sort='ascending'
            )).transform_calculate(
                prob_odds="format(datum.Proportion, ',.3f')+' (' +datum.Odds+')'",
                equal_odds="format(datum.Equal_better, ',.3f')+' (' +datum.Odds_Equal_better+')'",
                playoffs_odds="format(datum.Make_playoffs, ',.3f')+' (' +datum.Odds_Make_playoffs+')'"
            ).properties(
                title=f"{conf} playoff seedings",
                width=300,
                height=350,
            )
        
        chart_dict[conf] = c
        
    playoff_charts = alt.hconcat(*chart_dict.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    raw_data = pd.concat(source_list, axis=0)

    return (playoff_charts, raw_data)


def make_conf_champ_charts(matchup_dct):
    chart_dct = {}

    for conf in ["AFC", "NFC"]:
        prob_dct = matchup_dct[conf]
        # To get a symmetric DataFrame at the end
        for bye_team, dct in prob_dct.items():
            temp_dct = {}
            # Reversing the tuples
            for k,v in dct.items():
                temp_dct[k[::-1]] = v
            dct.update(temp_dct)
        df = pd.DataFrame(prob_dct).reset_index().rename({"level_0": "team1", "level_1": "team2"}, axis=1)
        df = df.melt(id_vars=["team1", "team2"], var_name="bye team", value_name="prob")
        df = df[df["prob"] > 0].copy()
        base = alt.Chart(df).encode(
            alt.X("team1").title(None),
            alt.Y("team2").title(None),
        ).properties(
            width=240
        )
        text = base.mark_text().encode(text="prob")
        rect = base.mark_rect().encode(
            alt.Color("prob", scale=alt.Scale(domain=[0,1]))
        )
        chart = (rect+text).facet(
            row="bye team"
        ).resolve_axis(
            x="independent",
            y="independent"
        ).properties(
            title=conf
        )

        chart_dct[conf] = chart
    
    return chart_dct


def make_win_charts(win_dict):
    odds_dict2 = {
                    "Proportion":"Odds",
                    "Equal_higher":"Odds_Equal_higher",
                    "Push_no_bet":"Odds_Push_no_bet"
                }

    reps = sum(win_dict["ARI"].values())

    win_charts = {}
    for conf in ["AFC","NFC"]:
        source = pd.DataFrame([(w,t,win_dict[t][w]/reps) for t in conf_teams[conf] for w in range(18)],
                    columns = ["Wins","Team","Proportion"])
        
        for a,b in source.groupby("Team"):
            loss_series = b.Proportion.cumsum()
            eq_higher = 1 - loss_series + b.Proportion
            higher = eq_higher - b.Proportion
            push_no_bet = higher/(1-b.Proportion)
            source.loc[source["Team"] == a, "Equal_higher"] = eq_higher
            source.loc[source["Team"] == a, "Push_no_bet"] = push_no_bet

        for c in odds_dict2.keys():
            source[odds_dict2[c]] = source[c].map(prob_to_odds)
        
        ordering = sorted(conf_teams[conf],
            key = lambda t: sum([a*b for a,b in win_dict[t].items()])/reps, reverse = True)

        ordering_wins = list(range(18,-1,-1))

        c = alt.Chart(source).mark_bar().encode(
                y=alt.Y('Team',sort = ordering),
                x=alt.X('Proportion',scale=alt.Scale(domain=[0,1])),
                tooltip = [alt.Tooltip("Team"),
                    alt.Tooltip('Wins', format=".0f"),
                    alt.Tooltip('prob_odds:N',title="Proportion"),
                    alt.Tooltip('equal_odds:N',title="Equal or higher"),
                    alt.Tooltip('push_odds:N',title="Push no bet")],
                color=alt.Color('Wins:N', scale=alt.Scale(scheme='tableau20'),
                            sort = ordering_wins),
                order=alt.Order(
                    'Wins:N',
                    sort='descending'
                )
            ).transform_calculate(
                prob_odds="format(datum.Proportion, ',.3f')+' (' +datum.Odds+')'",
                equal_odds="format(datum.Equal_higher, ',.3f')+' (' +datum.Odds_Equal_higher+')'",
                push_odds="format(datum.Push_no_bet, ',.3f')+' (' +datum.Odds_Push_no_bet+')'"
            ).properties(
                title=f"{conf} win totals",
                width=300,
                height=350,
            )

        overlay = pd.DataFrame({'Proportion': [0.5]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=.6).encode(x='Proportion:Q')

        win_charts[conf] = c + vline

    win_totals = alt.hconcat(*win_charts.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    return win_totals


def custom_sort(rd,div):
    div_teams = list(div_series[div_series==div].index)
    return sorted(div_teams,key = lambda t: rd[t][1], reverse=True)


def make_div_charts(rd):

    reps = sum(rd["ARI"].values())

    source = pd.DataFrame([(t,j,rd[t][j]/reps,div_series[t],0) for t in teams for j in range(1,5)],
            columns = ["Team","Rank","Proportion","Division","Odds"])

    source["Odds"] = source["Proportion"].map(prob_to_odds)

    output_dict = {}

    for div in sorted(list(set(div_series))):

        output_dict[div] = alt.Chart(source.query("Division==@div")).mark_bar().encode(
            x = alt.X("Rank:O",title="Place"),
            y = alt.Y("Proportion", scale=alt.Scale(domain=(0,1))),
            color = alt.Color("Proportion",scale=alt.Scale(scheme="lighttealblue", domain=(0,1))),
            column = alt.Column("Team:N", sort=custom_sort(rd,div)),
            tooltip = ["Team","Division","Rank",alt.Tooltip('Proportion', format=".3f"),alt.Tooltip('Odds:N')]
        ).properties(
            title = div,
            width = 100,
            height = 250
        )
    return alt.vconcat(*output_dict.values()).resolve_scale(
            color='independent'
        ).configure_concat(
            spacing=50
        )


def make_last_charts(last_list):
    last_winless_dct = {t: 0 for t in teams}
    last_undefeated_dct = {t: 0 for t in teams}

    for dct in last_list:
        und = dct["last_undefeated"]
        for t in und:
            last_undefeated_dct[t] += 1/len(und)
            
        winless = dct["last_winless"]
        for t in winless:
            last_winless_dct[t] += 1/len(winless)

    winless_ser = pd.Series(last_winless_dct)
    winless_ser = winless_ser/winless_ser.sum()
    winless_ser.name = "Proportion"
    df_winless = winless_ser.reset_index()
    df_winless["odds"] = df_winless["Proportion"].map(prob_to_odds)
    df_winless = df_winless.rename({"index": "Team"}, axis=1)

    undefeated_ser = pd.Series(last_undefeated_dct)
    undefeated_ser = undefeated_ser/undefeated_ser.sum()
    undefeated_ser.name = "Proportion"
    df_undefeated = undefeated_ser.reset_index()
    df_undefeated["odds"] = df_undefeated["Proportion"].map(prob_to_odds)
    df_undefeated = df_undefeated.rename({"index": "Team"}, axis=1)

    c1 = alt.Chart(df_undefeated, width=alt.Step(40)).mark_bar().encode(
        x = alt.X("Team", sort=alt.EncodingSortField("Proportion", order="descending")),
        y = "Proportion",
        color = alt.Color("Proportion",scale=alt.Scale(scheme="lighttealblue")),
        tooltip = ["Team", "Proportion", "odds"]
    ).properties(
        title="Last undefeated"
    )

    c2 = c1.mark_text(dy=-10).encode(
        color=alt.value("black"),
        text="odds"
    )

    undefeated_chart = c1+c2

    c1 = alt.Chart(df_winless, width=alt.Step(40)).mark_bar().encode(
        x = alt.X("Team", sort=alt.EncodingSortField("Proportion", order="descending")),
        y = "Proportion",
        color = alt.Color("Proportion",scale=alt.Scale(scheme="lighttealblue")),
        tooltip = ["Team", "Proportion", "odds"]
    ).properties(
        title="Last winless"
    )

    c2 = c1.mark_text(dy=-10).encode(
        color=alt.value("black"),
        text="odds"
    )

    winless_chart = c1+c2

    output_chart = alt.vconcat(undefeated_chart, winless_chart).configure_scale(
        bandPaddingInner=0.5
    )

    return output_chart, {"undefeated": df_undefeated, "winless": df_winless}


def make_streak_charts(streak_dict):
    odds_dict2 = {
                "Proportion":"Odds",
                "Equal_higher":"Odds_Equal_higher",
                "Push_no_bet":"Odds_Push_no_bet"
            }

    reps = sum(streak_dict["ARI"].values())

    streak_charts = {}
    for conf in ["AFC","NFC"]:
        source = pd.DataFrame([(w,t,streak_dict[t][w]/reps) for t in conf_teams[conf] for w in range(18)],
                    columns = ["Max win streak", "Team", "Proportion"])
        
        for a,b in source.groupby("Team"):
            loss_series = b.Proportion.cumsum()
            eq_higher = 1 - loss_series + b.Proportion
            higher = eq_higher - b.Proportion
            push_no_bet = higher/(1-b.Proportion)
            source.loc[source["Team"] == a, "Equal_higher"] = eq_higher
            source.loc[source["Team"] == a, "Push_no_bet"] = push_no_bet

        for c in odds_dict2.keys():
            source[odds_dict2[c]] = source[c].map(prob_to_odds)
        
        ordering = sorted(conf_teams[conf],
            key = lambda t: sum([a*b for a,b in streak_dict[t].items()])/reps, reverse = True)

        ordering_streaks = list(range(18,-1,-1))

        c = alt.Chart(source).mark_bar().encode(
                y=alt.Y('Team',sort = ordering),
                x=alt.X('Proportion',scale=alt.Scale(domain=[0,1])),
                tooltip = [alt.Tooltip("Team"),
                    alt.Tooltip('Max win streak', format=".0f"),
                    alt.Tooltip('prob_odds:N',title="Proportion"),
                    alt.Tooltip('equal_odds:N',title="Equal or higher"),
                    alt.Tooltip('push_odds:N',title="Push no bet")],
                color=alt.Color('Max win streak:N', scale=alt.Scale(scheme='tableau20'),
                            sort = ordering_streaks),
                order=alt.Order(
                    'Max win streak:N',
                    sort='descending'
                )
            ).transform_calculate(
                prob_odds="format(datum.Proportion, ',.3f')+' (' +datum.Odds+')'",
                equal_odds="format(datum.Equal_higher, ',.3f')+' (' +datum.Odds_Equal_higher+')'",
                push_odds="format(datum.Push_no_bet, ',.3f')+' (' +datum.Odds_Push_no_bet+')'"
            ).properties(
                title=f"{conf} longest win streaks",
                width=300,
                height=350,
            )

        overlay = pd.DataFrame({'Proportion': [0.5]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=.6).encode(x='Proportion:Q')

        streak_charts[conf] = c + vline

    streak_concat = alt.hconcat(*streak_charts.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    return streak_concat


def make_stage_charts(stage_dict):
    odds_dict2 = {"Proportion":"Odds"}

    reps = sum(stage_dict["ARI"].values())
    stages = stage_dict["ARI"].keys()

    source_list = []

    stage_charts = {}
    for conf in ["AFC","NFC"]:
        source = pd.DataFrame([(stage,t,stage_dict[t][stage]/reps) for t in conf_teams[conf] for stage in stages],
                    columns = ["Stage","Team","Proportion"])


        for c in odds_dict2.keys():
            source[odds_dict2[c]] = source[c].map(prob_to_odds)

        c = alt.Chart(source).mark_bar().encode(
                y=alt.Y('Team'),
                x=alt.X('Proportion',scale=alt.Scale(domain=[0,1])),
                tooltip = [alt.Tooltip("Team"),
                    alt.Tooltip('Stage'),
                    alt.Tooltip('prob_odds:N',title="Proportion")],
                color=alt.Color('Stage:N', scale=alt.Scale(scheme='tableau20'),
                            sort = None),
                order=alt.Order(
                    'Team:N',
                    sort='ascending'
                )
            ).transform_calculate(
                prob_odds="format(datum.Proportion, ',.3f')+' (' +datum.Odds+')'"
            ).properties(
                title=f"{conf} Stage of Elimination",
                width=300,
                height=350,
            )

        overlay = pd.DataFrame({'Proportion': [0.5]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=.6).encode(x='Proportion:Q')

        stage_charts[conf] = c + vline

        source_list.append(source)

    stage_totals = alt.hconcat(*stage_charts.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    raw_data = pd.concat(source_list, axis=0)

    return (stage_totals, raw_data)


def make_conference_chart(stage_dict):

    reps = sum(stage_dict["ARI"].values())

    teams = stage_dict.keys()

    sb_dict = {t: (stage_dict[t]["Lose in Super Bowl"]+stage_dict[t]["Win Super Bowl"])/reps for t in teams}

    conf_charts = {}
    for conf in ["AFC","NFC"]:
        source = pd.DataFrame([(t,sb_dict[t]) for t in conf_teams[conf]], columns = ["Team","Proportion"])

        source["Odds"] = source["Proportion"].map(prob_to_odds)

        c1 = alt.Chart(source, width=alt.Step(40)).mark_bar().encode(
            x = alt.X("Team", sort=alt.EncodingSortField("Proportion", order="descending")),
            y = "Proportion",
            color = alt.Color("Proportion", scale=alt.Scale(scheme="lighttealblue")),
            tooltip = ["Team", "Proportion", "Odds"]
        ).properties(
                title=f"{conf} Champion"
        )

        c2 = c1.mark_text(dy=-10).encode(
            color=alt.value("black"),
            text="Odds"
        )

        conf_chart = c1+c2
        conf_charts[conf] = conf_chart

    conf_totals = alt.vconcat(*conf_charts.values()).resolve_scale(
        color='independent'
    ).properties(
        title=f"Based on {reps} simulations:"
    )

    return conf_totals


def make_superbowl_chart(stage_dict):

    reps = sum(stage_dict["ARI"].values())

    teams = stage_dict.keys()

    sb_dict = {t: stage_dict[t]["Win Super Bowl"]/reps for t in teams}

    source = pd.DataFrame([(t,sb_dict[t]) for t in teams], columns = ["Team","Proportion"])

    source["Odds"] = source["Proportion"].map(prob_to_odds)


    c1 = alt.Chart(source, width=alt.Step(40)).mark_bar().encode(
        x = alt.X("Team", sort=alt.EncodingSortField("Proportion", order="descending")),
        y = "Proportion",
        color = alt.Color("Proportion", scale=alt.Scale(scheme="lighttealblue")),
        tooltip = ["Team", "Proportion", "Odds"]
    ).properties(
        title="Super Bowl Winner"
    )

    c2 = c1.mark_text(dy=-10).encode(
        color=alt.value("black"),
        text="Odds"
    )

    superbowl_chart = c1+c2

    return superbowl_chart


def best_record_chart(best_list):
    best_dct = {t: 0 for t in teams}

    for t in best_list:
        best_dct[t] += 1

    best_ser = pd.Series(best_dct)
    best_ser = best_ser/best_ser.sum()
    best_ser.name = "Proportion"
    df_best = best_ser.reset_index()
    df_best["odds"] = df_best["Proportion"].map(prob_to_odds)
    df_best = df_best.rename({"index": "Team"}, axis=1)

    c1 = alt.Chart(df_best, width=alt.Step(40)).mark_bar().encode(
        x = alt.X("Team", sort=alt.EncodingSortField("Proportion", order="descending")),
        y = "Proportion",
        color = alt.Color("Proportion",scale=alt.Scale(scheme="lighttealblue")),
        tooltip = ["Team", "Proportion", "odds"]
    ).properties(
        title="Best Regular Season Record"
    )

    c2 = c1.mark_text(dy=-10).encode(
        color=alt.value("black"),
        text="odds"
    )

    output_chart = (c1+c2).configure_scale(
        bandPaddingInner=0.5
    )

    return output_chart, df_best
