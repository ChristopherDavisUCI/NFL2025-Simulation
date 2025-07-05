import pandas as pd

df_names = pd.read_csv("data/NameReplacements.csv")
df_names.set_index("Long", drop=True, inplace=True)
series_names = df_names["Short"]

def get_abbr(long):
    long = long.upper()
    if long == "LAS":
        return "LV"
    if long == "LAR":
        return "LA"
    if long in series_names.values:
        return long
    if long in series_names.index:
        return series_names[long]
    temp = long.split()[-1]
    if temp in series_names.index:
        return series_names[temp]
    temp = long.split("-")[-1]
    if temp in series_names.index:
        return series_names[temp]