from collections import Counter

def matchup_probs(matchup_list):
    # matchup_list looks like [(bye_team, (matchup team 1, matchup team 2)), ...]
    bye_set = {tup[0] for tup in matchup_list}
    # We want to know what the conference championship teams are, for a fixed bye
    # The keys of matchup_dct are the teams with a bye
    matchup_dct = {team: {} for team in bye_set}
    for bye, teams in matchup_list:
        bye_dct = matchup_dct[bye]
        if teams in bye_dct:
            bye_dct[teams] += 1
        else:
            bye_dct[teams] = 1

    for bye, match_dct in matchup_dct.items():
        total_matches = sum(match_dct.values())
        for k,v in match_dct.items():
            match_dct[k] = round(v/total_matches, 2)

    bye_counter = Counter(tup[0] for tup in matchup_list)
    # converting the proportion to a string because it will be used as a dictionary key
    prop_dct = {k:round(v/len(matchup_list),2) for k,v in bye_counter.items()}
    
    # Replacing the bye team like "BUF" with "BUF, 0.21", where the number represents its probability of the bye
    output_dct = {}
    for k,v in matchup_dct.items():
        output_dct[f"{k}, {prop_dct[k]}"] = matchup_dct[k]

    return output_dct

def get_conf_matchup_probs(conf_dct):
    output_dct = {}
    for conf, matchup_list in conf_dct.items():
        output_dct[conf] = matchup_probs(matchup_list)
    
    return output_dct