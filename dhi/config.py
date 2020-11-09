# coding: utf-8

"""
Constants such as cross sections and branchings, and common configs such as labels and colors.
"""

from dhi.util import DotDict


# branching ratios, m_H = 125 GeV
# https://pdg.lbl.gov/2020/listings/rpp2020-list-w-boson.pdf
# https://pdg.lbl.gov/2020/listings/rpp2020-list-z-boson.pdf
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR?rev=22
br_w = DotDict(
    qq=0.6741,
    lv=1.0 - 0.6741,
)
br_z = DotDict(
    qq=0.69911,
    ll=3 * 0.03365,
)
br_h = DotDict(
    bb=0.5824,
    ww=0.2137,
    tautau=0.06272,
    zz=0.02619,
    gg=0.002270,
)
br_ww = DotDict(
    qqqq=br_w.qq ** 2.0,
    llvv=br_w.lv ** 2.0,
    qqlv=2.0 * br_w.qq * br_w.lv,
)
br_zz = DotDict(
    qqqq=br_z.qq ** 2.0,
    llll=br_z.ll ** 2.0,
    qqll=2.0 * br_z.qq * br_z.ll,
)
br_hh = DotDict(
    bbbb=br_h.bb ** 2.0,
    bbvv=2.0 * br_h.bb * (br_h.ww + br_h.zz),
    bbww=2.0 * br_h.bb * br_h.ww,
    bbwwqqlv=2.0 * br_h.bb * br_h.ww * br_ww.qqlv,
    bbwwllvv=2.0 * br_h.bb * br_h.ww * br_ww.llvv,
    bbzz=2.0 * br_h.bb * br_h.zz,
    bbzzqqll=2.0 * br_h.bb * br_h.zz * br_zz.qqll,
    bbzzllll=2.0 * br_h.bb * br_h.zz * br_zz.llll,
    bbtautau=2.0 * br_h.bb * br_h.tautau,
    bbgg=2.0 * br_h.bb * br_h.gg,
)

# HH branching names (TODO: find prettier abbreviations)
br_hh_names = DotDict(
    bbbb=r"bbbb",
    bbvv=r"bbVV",
    bbww=r"bbWW",
    bbwwqqlv=r"bbWW$_{qql\nu}$",
    bbwwllvv=r"bbWW$_{2l2\nu}$",
    bbzz=r"HH $\rightarrow bbZZ$",
    bbzzqqll=r"bbZZ$_{qqll}$",
    bbzzllll=r"bbZZ$_{4l}$",
    bbtautau=r"bb\tau\tau$",
    bbgg=r"bb\gamma\gamma$",
)

# campaign labels
campaign_labels = DotDict({
    "2016": "2016 (13 TeV)",
    "2017": "2017 (13 TeV)",
    "2018": "2018 (13 TeV)",
    "run2": "Run 2 (13 TeV)",
})

# poi defaults (value, range, points, taken from physics model) and labels
poi_data = DotDict({
    "r": DotDict(range=(0.0, 10.0), label="r"),
    "r_gghh": DotDict(range=(0.0, 10.0), label="r_{gghh}"),
    "r_qqhh": DotDict(range=(0.0, 10.0), label="r_{qqhh}"),
    "kl": DotDict(range=(-30.0, 30.0), label=r"\kappa_{\lambda}"),
    "kt": DotDict(range=(-10.0, 10.0), label=r"\kappa_{t}"),
    "C2V": DotDict(range=(-10.0, 10.0), label="C_{2V}"),
    "CV": DotDict(range=(-10.0, 10.0), label="C_{V}"),
})
# add "$" embedded labels
for poi, data in poi_data.items():
    data["label_math"] = "${}$".format(data.label)

# nuisance parameters labels
nuisance_labels = {}

# colors
colors = DotDict({
    "root": DotDict({
        "black": 1,
        "red": 628,
        "green": 418,
        "yellow": 798,
        "red_cream": 46,
        "blue_cream": 38,
    })
})

# cumulative, inverse chi2 values in a mapping "n_dof -> n_sigma -> level"
# for the geometrical determination of errors of nll curves
chi2_levels = {
    1: {1: 1.000, 2: 4.000},
    2: {1: 2.296, 2: 6.180},
}
