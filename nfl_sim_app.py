import streamlit as st
from numpy.random import default_rng
from collections import Counter
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
from sim_season import simulate_reg_season, make_pr_custom
from sim_playoffs import stages, simulate_playoffs
from make_standings import Standings
from make_charts import (
                            make_playoff_charts,
                            make_conf_champ_charts,
                            make_win_charts, 
                            make_div_charts,
                            make_last_charts,
                            make_streak_charts,
                            make_stage_charts,
                            make_conference_chart,
                            make_superbowl_chart,
                            best_record_chart
                        )
from compare_market import compare_market
from conference_matchups import get_conf_matchup_probs
from itertools import permutations
from last_teams import get_last, get_streaks
from name_helper import get_abbr
from odds_helper import prob_to_odds
import time

st.set_page_config(layout="wide", page_title="2024 Simulator")

st.title('2025 NFL Season Simulator')
# New season: update schedule below and in the `sim_season.py` file
# New season: update every schedule24.csv to schedule25.csv

pr_default = pd.read_csv("data/pr.csv", index_col="Team").squeeze()
div_series = pd.read_csv("data/divisions.csv", index_col=0).squeeze()

# The next portion doesn't seem to work the first time it's run
# due to imports above
download_results = False
if download_results:
    df_schedule = pd.read_csv("https://raw.githubusercontent.com/nflverse/nfldata/refs/heads/master/data/games.csv")
    df_schedule = df_schedule.query("season==2025")
    df_schedule["conf_game"] = df_schedule.apply(
        lambda row: div_series[row["home_team"]][:3] == div_series[row["away_team"]][:3], axis=1
        )
    df_schedule.to_csv("schedules/schedule25.csv", index=False)
else:
    df_schedule = pd.read_csv("schedules/schedule25.csv")

# last_played = df_schedule[df_schedule["home_score"].notna()].iloc[-1]
last_played = "Nothing"
teams = div_series.index
conf_teams = {}
for conf in ["AFC","NFC"]:
    conf_teams[conf] = [t for t in teams if div_series[t][:3]==conf]
rng = default_rng()

div_dict = {}

s = div_series
for a,b in s.groupby(s):
    div_dict[a] = list(b.index)

def reps_changed():
    st.session_state["rc"] = True

def comb_changed():
    if "pc" in st.session_state:
        st.session_state["rc"] = True


max_reps = 1600

reps = st.sidebar.number_input('Number of simulations to run:',value=200, min_value = 1, max_value = max_reps, help = f'''The maximum allowed value is {max_reps}.''',on_change=reps_changed)

def make_radio(row):
    home = row['home_team']
    away = row['away_team']
    label = f"Week {row['week']}: {away} @ {home}"
    return st.radio(label=label, options=["Blank", away, home, "Tie"])

with st.sidebar:

    st.write("Set your power ratings.")

    with st.expander("Set ratings from a csv file"):

        pr_download = pr_default.drop("mean_score")

        st.download_button(
            "Download a template", 
            data = pr_download.to_csv().encode('utf-8'),
            file_name = "power_ratings.csv",
            mime="text/csv"
            )

        pr_file = st.file_uploader("Accepts a csv file", type=["csv"])

        if pr_file:
            df_upload = pd.read_csv(pr_file)
            for name in ["Team", "team", "TEAM"]:
                if name in df_upload.columns:
                    df_upload = df_upload.set_index("Team", drop=True)
                    found = True
                    break
            if not found:
                df_upload = df_upload.set_index(df_upload.columns[0], drop=True)
            if "PR" in df_upload.columns:
                pr_upload = df_upload["PR"]
            elif "pr" in df_upload.columns:
                pr_upload = df_upload["pr"]
            else:
                pr_upload = df_upload.iloc[:, 0]
            pr_upload2 = pd.Series()
            for k in pr_upload.index:
                if not isinstance(k, str):
                    continue
                if k.upper().strip() == "HFA":
                    pr_upload2["HFA"] = pr_upload[k]
                abbr = get_abbr(k)
                if abbr:
                    pr_upload2[abbr] = pr_upload[k]
            
            # Replace the uploaded ratings with the cleaned version
            pr_upload = pr_upload2

    with st.expander("Fix specific game outcomes:"):
        forced_outcomes = {}
        df_unplayed = df_schedule[df_schedule["home_score"].isna()]
        for ind, row in df_unplayed.iterrows():
            forced_outcomes[ind] = make_radio(row)

pr = {"mean_score": float(pr_default["mean_score"])}

with st.sidebar:

    for t in teams:
        try:
            value = pr_upload[t]
        except (NameError, KeyError): 
            value = pr_default[t]
        pr[t] = st.slider(
            f'{t} power rating',
            -15.0, 15.0, float(value))
        
    st.subheader("Extra parameter:")
    try:
        value = pr_upload["HFA"]
    except (NameError, KeyError): 
        value = pr_default["HFA"]
    pr["HFA"] = st.slider(
        'Home field advantage',
        0.0, 10.0, float(value))
    
    pr_complete = make_pr_custom(pr)


with st.sidebar:
    series_download = pd.Series(pr, name="PR")
    series_download.index.name = "Side"


df_pr = pd.DataFrame({"Overall": {t:pr_complete[t+"_Off"] + pr_complete[t+"_Def"] for t in teams},
        "Offensive":{t:pr_complete[t+"_Off"] for t in teams}, "Defensive":{t:pr_complete[t+"_Def"] for t in teams}}
        )


st.markdown('''Based on your power ratings, we use Python to run many simulations of the 2023 NFL season, and then estimate answers to questions like:
* How likely is Dallas to get the no. 1 seed?  To win its division? To make the playoffs?
* How likely are the Steelers to win exactly 11 games?  To win 11 or more games?
* How likely are the Patriots to finish 3rd in the AFC East?''')

st.markdown('''In each simulation, a random outcome is generated for all 272 regular season + 13 postseason games.  The outcomes are based on the power ratings; you should customize these power ratings on the left.
(**Warning**.  Even if you set the absolute perfect power ratings, our simulation is too simple to provide true probabilities.  For example, our simulation assumes the power ratings stay constant throughout the season.)


Click the button below to run the simulation.''')

button_cols1, button_cols2 = st.columns((1,5))

sim_button = button_cols1.button("Run simulations")

with button_cols2:
    time_holder = st.empty()

# "rc" stands for "repetitions changed"
# does it really make sense to run in that case?
if sim_button or ("rc" in st.session_state):
    try:
        del st.session_state["rc"]
    except:
        pass
    df_forced = df_schedule.copy()
    for k,v in forced_outcomes.items():
        if v == "Blank":
            continue
        row = df_forced.loc[k]
        if v == row["away_team"]:
            df_forced.loc[k, ["away_score", "home_score"]] = [24, 17]
        elif v == row["home_team"]:
            df_forced.loc[k, ["away_score", "home_score"]] = [17, 24]
        elif v == "Tie":
            df_forced.loc[k, ["away_score", "home_score"]] = [17, 17]
        else:
            raise ValueError("The specified value could not be found.")
    st.header("Simulation results")
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder0.text(f"Running {reps} simulations of the 2024 NFL season")
    bar = placeholder1.progress(0.0)
    st.write("")

    playoff_dict = {}
    
    for conf in ["AFC","NFC"]:
        playoff_dict[conf] = {i:{t:0 for t in sorted(conf_teams[conf])} for i in range(1,8)}

    win_dict = {t:{i:0 for i in range(18)} for t in teams}

    # Longest win streak for each team
    streak_dict = {t:{i:0 for i in range(18)} for t in teams}

    rank_dict = {div:{} for div in div_dict.keys()}
    rank_dict1 = {t:{} for t in teams}

    # List of terms like {'last_undefeated': ('KC',), 'last_winless': ('TB',)}
    last_list = []

    # Stage of elimination for each team
    stage_dict = {t:{k: 0 for k in stages} for t in teams}

    # List of best regular season record teams
    best_record_list = []

    # List of super bowl exact matchups, listing NFC first
    matchup_list = []

    # List of conference championship matchups
    afc_matchup_list = []
    nfc_matchup_list = []

    for div in rank_dict.keys():
       for team_sort in permutations(div_dict[div]):
           rank_dict[div][team_sort] = 0

    for t in teams:
        for i in range(1,5):
            rank_dict1[t][i] = 0

    # How often do different playoff scenarios occur?
    # A dictionary whose values are lists of tuples
    # tuples like ("SF", "PHI", ...)
    playoff_full = {"AFC": [], "NFC": []}

    start = time.time()

    for i in range(reps):
        df = simulate_reg_season(pr, df_forced)
        streaks = get_streaks(df)
        last_list.append(get_last(df))
        stand = Standings(df)

        p = stand.playoffs
        stage_of_elim, exact_matchup, conf_champ_matchups = simulate_playoffs(pr, p)
        for conf in ["AFC", "NFC"]:
            playoff_full[conf].append(tuple(p[conf]))
            for j,t in enumerate(p[conf]):
                playoff_dict[conf][j+1][t] += 1
        for t in teams:
            team_outcome = stand.standings.loc[t]
            win_dict[t][team_outcome["Wins"]] += 1
            rank_dict1[t][team_outcome["Division_rank"]] += 1
            streak_dict[t][streaks[t]] += 1
            stage_dict[t][stage_of_elim[t]] += 1

        best_record_list.append(stand.best_reg_record)
        matchup_list.append(exact_matchup)
        
        # TODO: Make this more robust, assumes AFC is always first
        afc_matchup_list.append(conf_champ_matchups[0])
        nfc_matchup_list.append(conf_champ_matchups[1])
        
        for d in rank_dict.keys():
           rank_dict[d][tuple(stand.div_ranks[d])] += 1
        
        bar.progress((i+1)/reps)

    # The following is a dictionary describing, for each conference and each possible bye team, how likely various 
    # conference championships are.
    matchup_dct = get_conf_matchup_probs(
        {
            "AFC": afc_matchup_list,
            "NFC": nfc_matchup_list
        }
    )

    for d in rank_dict.keys():
       rank_dict[d] = {i:j/reps for i,j in rank_dict[d].items()}

    st.session_state["rd"] = rank_dict

    end = time.time()
    
    time_holder.markdown(f'''{reps} simulations of the 2024 NFL season took {end - start:.1f} seconds.''')
    # Last updated game: Week {last_played['week']}: {last_played['away_team']} {int(last_played['away_score'])} - {last_played['home_team']} {int(last_played['home_score'])}''')


    playoff_charts, raw_data = make_playoff_charts(playoff_dict)

    conf_champ_charts = make_conf_champ_charts(matchup_dct)

    pivot_seed = raw_data.pivot(index="Team", columns="Seed")['Proportion']
    pivot_seed.columns = [f"Seed {c}" for c in pivot_seed.columns]

    win_charts = make_win_charts(win_dict)

    div_charts = make_div_charts(rank_dict1)

    last_charts, last_dct = make_last_charts(last_list)
    last_dct_dfs = {}
    for k, df_temp in last_dct.items():
        df_temp = df_temp.set_index("Team", drop=True)[["Proportion"]]
        df_temp.rename({"Proportion": f"Last {k}"}, axis=1, inplace=True)
        last_dct_dfs[k] = df_temp

    streak_charts = make_streak_charts(streak_dict)

    stage_charts, champ_data = make_stage_charts(stage_dict)

    pivot_stage = champ_data.pivot(index="Team", columns="Stage")['Proportion']
    pivot_stage.columns = [f"Stage of Elim: {c}" for c in pivot_stage.columns]

    conference_chart = make_conference_chart(stage_dict)

    superbowl_chart = make_superbowl_chart(stage_dict)

    best_chart, df_best = best_record_chart(best_record_list)

    df_best = df_best.set_index("Team", drop=True)[["Proportion"]]
    df_best.rename({"Proportion": "Best Record"}, axis=1, inplace=True)

    pivot_all = pd.concat(
        (
            pivot_seed,
            pivot_stage,
            df_best,
            *last_dct_dfs.values()
        ), axis=1)

    st.session_state['pc'] = playoff_charts
    st.session_state['ccc'] = conf_champ_charts
    st.session_state['wc'] = win_charts
    st.session_state['dc'] = div_charts
    st.session_state['lc'] = last_charts
    st.session_state['streak_charts'] = streak_charts
    st.session_state['stage_charts'] = stage_charts
    st.session_state['conference_chart'] = conference_chart
    st.session_state['superbowl_chart'] = superbowl_chart
    st.session_state['best_chart'] = best_chart
    st.session_state['raw_data'] = compare_market(raw_data, champ_data, pivot_all, matchup_list)
    st.session_state['full_standings'] = playoff_full
    st.session_state['pivot'] = pivot_all


# Input is a tuple like ("SF", "PHI", ...)
# Matchups are 2-7, 3-6, 4-5
# In slots 1-6, 2-5, 3-4
def stand_to_matchups(tup):
    output_str = f"1: {tup[0]}"
    for i in range(2,5):
        j = 9-i #opponent
        output_str += f". {i}-{j}: {tup[i-1]}-{tup[j-1]}"
    output_str += "."
    return output_str


# Input should be a list of tuples, the tuples like
# ("SF", "PHI", ...)
def process_standings(list_of_tuples):
    tot = len(list_of_tuples)
    count = Counter(list_of_tuples)
    props = [(k,v/tot) for k,v in count.items()]
    sorted_props = sorted(props, key=lambda x: x[1], reverse=True)
    matchup_props = []
    for tup,prop in sorted_props:
        matchup_props.append((stand_to_matchups(tup), prop))
    return matchup_props


def make_ranking(df,col):
    return (-df[col]).rank()

def make_sample():
    st.header("Sample images:")
    if "pc" not in st.session_state:
        st.write('(To replace the sample images with real images, press the "Run simulations" button above.)')
    c_image, c_text = st.columns(2)
    with c_image:
        st.image("images/pc_holder.png")
    with c_text:
        st.subheader("How to interpret the playoff seeding image.")
        st.markdown('''The NFC playoff seeding image shows the probability of different teams getting different playoff seeds, according to our simulations.\n\nFor example:
* Philadelphia has about an 80% chance of making the playoffs, while Arizona has less than a 10% chance.  This corresponds to the total heights of the bars in the chart.
* Philadelphia is much more likely to get a 1 seed (dark blue bar) than Minnesota (about 20% vs 5%).
* Philadelphia is also much less likely to get a 4 seed (light blue bar) than a 1 seed (about 8% vs 20%). 
* For more precise numbers, place your mouse over a bar in one of the real images (not the sample images).
The displayed text in our sample image shows that, according to our simulations:
* Dallas has an 11.3% chance of getting a 6 seed;
* Dallas has a 68.5% chance of making the playoffs.    
* The fair odds corresponding to these probabilities are +785 and -217, respectively.''')
    c_image, c_text = st.columns(2)
    with c_image:
        st.image("images/wc_holder.png")
    with c_text:
        st.subheader("How to interpret the win total image.")
        st.markdown('''The NFC win total image shows the probability, according to our simulations, of different teams having a specific number of wins at the end of the 2023 regular season.
The thin black line represents the median win total for each team.\n\nFor example:
* The thin black vertical line passes through the median win total for each team.  The median win total for San Francisco is 10, the median win total for Chicago is 8, while the median win total for Arizona is 6.
* It looks like Washington has about a 40% chance of winning 8 or more games (the region to the left of the red bar).
* Dallas has a 13.7% chance of winning exactly 11 games, and a 31.5% chance of winning at least 11 games.

**Warning**.  These numbers may look precise because of the decimal places, but they're coming from imperfect simulations.''')
    c_image, c_text = st.columns(2)
    with c_image:
        st.image("images/dc_holder.png")
    with c_text:
        st.subheader("How to interpret the division ranks image.")
        st.markdown('''The division ranks image shows the probability of different teams finishing in specific ranks in their divisions.
\n\nFor example, according to our simulations:
* Dallas has a 32.9% chance of coming in second in the NFC East, and Washington has about a 50% chance of finishing 4th in the NFC East.
* The teams are sorted in terms of how likely they are to win their division.  If all you care about is how likely is the team to win its division, then it's probably more convenient to use the playoff seedings image.''')

if 'pc' in st.session_state:
    
    try:
        placeholder0.write(st.session_state['pc'])
        placeholder1.write(st.session_state['wc'])
    except:
        st.header("Simulation results")
        st.write(st.session_state['pc'])
        st.write(st.session_state['wc'])
    st.altair_chart(st.session_state["conference_chart"])
    st.altair_chart(st.session_state["superbowl_chart"])
    st.altair_chart(st.session_state["stage_charts"])
    # st.altair_chart(st.session_state["best_chart"])
    # df_temp = pd.read_csv("data/markets.csv")
    # st.write(f'''The following were last updated on {df_temp.loc[0, 'date']}.  Needless to say, do not take the kelly staking sizes literally!!  (Along with errors and imprecisions in our app, also keep in mind how long the stake will be sitting unused.)
    # By default, they are sorted by odds, but click the kelly column to sort by that.''')
    # Once we have betting odds to compare, uncomment the next line
    # st.dataframe(st.session_state['raw_data'])
    # st.altair_chart(st.session_state['lc'])
    # st.altair_chart(st.session_state["streak_charts"])

    st.subheader("Download as a csv")
    st.download_button(
            "Download the results", 
            data = st.session_state["pivot"].to_csv().encode('utf-8'),
            file_name = f"sim_results_{reps:.0f}_reps.csv",
            mime="text/csv"
        )
    
else:
    make_sample()

df_rankings = pd.DataFrame({col:make_ranking(df_pr,col) for col in df_pr.columns})
    
st.subheader("Further options")
radio_dict = {
    "Conference": "Probabilities of various Conference Championship matchups.",
    "Matchups": "Probabilities of playoff standings 1-7.",
    "Division": "Probabilities for different division ranks.",
    "Rankings": "See the power rankings 1-32 based on the current values.",
    "Sample": "The sample images and explanations.",
    "Details": "More details about the process.",
    "Follow": "Possible follow-ups.",
    "Contact": "Contact"
}

info_choice = st.radio(
    'Options for more information',
    radio_dict.keys(),
    key="opt_radio",
    format_func=lambda k: radio_dict[k],
    index=0
    )

if info_choice == "Matchups":
    try:
        st.subheader("Probabilities of playoff matchups")
        fs = st.session_state["full_standings"]
        n = 50 # how many matchups to display
        for conf in ["AFC", "NFC"]:
            rows = process_standings(fs[conf][:n])
            m = min(n, len(rows))
            st.write(f"{m} most likely {conf} matchups:")
            output_str = ''''''
            for match, prop in rows:
                output_str += f'''Probability {prop:.3f}: {match}  
'''
            st.markdown(output_str)
    except KeyError:
        st.write('No data yet. Press the "Run simulations" button above.')
elif info_choice == "Conference":
    try:
        conf_champ_charts = st.session_state["ccc"]
        st.altair_chart(
            alt.vconcat(conf_champ_charts["AFC"], conf_champ_charts["NFC"])
            )
    except KeyError:
        st.write('No data yet. Press the "Run simulations" button above.')
elif info_choice == "Rankings":
    st.subheader("Power rankings 1-32 based on current values")
    st.dataframe(df_rankings[["Overall"]].astype(int).sort_values("Overall"),height=1000)
elif info_choice == "Contact":
    st.markdown('')
    st.markdown('''Source code for this app: [NFL2023-Simulation](https://github.com/ChristopherDavisUCI/NFL2023-Simulation)<br>
Twitter profile: [@CDavis_UCI](https://twitter.com/cdavis_uci)<br>
Please contact me with any questions or suggestions!''',unsafe_allow_html=True)
elif info_choice == "Details":
    st.markdown('''* Warning! I'm not an expert on any of this material (not on Python, not on the NFL, not on random processes, not on simulating sports outcomes).  I'm trying to learn more.  I originally made this app before the 2021 season.
* In the simulation, for each game, we compute the expected score for both teams using the power ratings.  (See the examples in the left-hand panel.) 
We then use a normal distribution with the expected score as the mean, and with 10 as the standard deviation.  In the current implementation, only the matchups are relevant, 
not the week in which the matchup occurs.
* You can download the source code from [Github](https://github.com/ChristopherDavisUCI/NFL2023-Simulation).
* This website was made using [Streamlit](https://streamlit.io/).
* The plots were made using [Altair](https://altair-viz.github.io/). 
In the plots from your simulation (not the placeholder images), put your mouse over the bars to get more data.
* The default power ratings were computed based on the season-long lines and totals from [nflverse](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) on June 25, 2023.
* No promises that my code is accurate.  (The hardest/most tedious part was implementing the tie-breaking procedures to determine playoff seedings. 
I believe all tie-breakers are incorporated except for the tie-breakers involving touchdowns.)
* Please report any bugs!
    ''')
elif info_choice == "Division":
    if "dc" in st.session_state:
        st.subheader("Division ranks")
        st.write(f"Based on {reps:.0f} simulations:")
        st.write(st.session_state["dc"])
    else:
        st.write('No data yet.  Press the "Run simulations" button above.')
    
elif info_choice == "Sample":
    make_sample()
elif info_choice == "Follow":
    
    st.subheader("Follow-up ideas")

    st.markdown('''* Adapt the code so that teams coming off of a bye week have slightly boosted power ratings.
    * Given an over-under win total for a specific team, estimate what the fair odds should be.  (Warning.  If the over-under win total is 9, for example,
the fair odds for "over 9" does not correspond directly to the probability of the team winning 10 or more games, because pushes need to be treated differently from losses.)
* Extend the simulations to include the playoffs.  Create charts showing which teams win the super bowl,
reach the super bowl, and reach the conference championship games most often.
* One weakness of our simulation is that a team has the same power rating throughout the entire season.  In fact, the order in which games are played has no impact on our simulation.
Adapt the code so that the power ratings evolve over the course of the season.
* I didn't think much about dealing with ties.  I wrote some ad hoc code that gets rid of most ties, 
with the home team slightly more likely to win in overtime than the road team.  (Without this ad hoc code, there were as many as ten ties per season.)  Come up with a more sophisticated solution.''')

st.markdown('___')
st.markdown("Disclaimer: This app is for Educational and Entertainment purposes only.  Do not take the resulting values literally.  The app has no connection to any organizations nor to any individuals other than the author.")
