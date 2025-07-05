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
    

def odds_to_prob(x):
    x = float(x)
    if x < 0:
        y = -x
        return y/(100+y)
    else:
        return 100/(100+x)
    

def kelly(p, odds):
    '''p is the probability, odds is for example 200 for 2/1'''
    implied_prob = odds_to_prob(odds)
    if odds > 0:
        b = odds/100
    elif odds < 0:
        b = 100/(-odds)
    return p - (1-p)/(b)