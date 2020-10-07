# coding: utf-8

"""
Common plot configurations such as labels, colors, etc.
"""

from dhi.util import DotDict


# branching ratios
br_hww_hbb = 0.0264

# NLO -> NNLO k-factor
k_factor = 1.115

# campaign labels
campaign_labels = DotDict({
    "2016": "2016 (13 TeV)",
    "2017": "2017 (13 TeV)",
    "2018": "2018 (13 TeV)",
    "FullRun2": "2016+2017+2018 (13 TeV)",
})

# poi defaults (value, range, points, taken from physics model) and labels
poi_data = DotDict({
    "r": DotDict(range=(0., 10.), label="r"),
    "r_gghh": DotDict(range=(0., 10.), label="r_{gghh}"),
    "r_qqhh": DotDict(range=(0., 10.), label="r_{qqhh}"),
    "kl": DotDict(range=(-30., 30.), label=r"\kappa_{\lambda}"),
    "kt": DotDict(range=(-10., 10.), label=r"\kappa_{t}"),
    "C2V": DotDict(range=(-10., 10.), label="C_{2V}"),
    "CV": DotDict(range=(-10., 10.), label="C_{V}"),
})
# add "$" embedded labels
for poi, data in poi_data.items():
    data["label_math"] = "${}$".format(data.label)
