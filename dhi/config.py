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
    lv=1. - 0.6741,
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
    qqqq=br_w.qq**2.,
    llvv=br_w.lv**2.,
    qqlv=2. * br_w.qq * br_w.lv,
)
br_zz = DotDict(
    qqqq=br_z.qq**2.,
    llll=br_z.ll**2.,
    qqll=2. * br_z.qq * br_z.ll,
)
br_hh = DotDict(
    bbbb=br_h.bb**2.,
    bbww=2. * br_h.bb * br_h.ww,
    bbwwqqlv=2. * br_h.bb * br_h.ww * br_ww.qqlv,
    bbwwllvv=2. * br_h.bb * br_h.ww * br_ww.llvv,
    bbzz=2. * br_h.bb * br_h.zz,
    bbzzqqll=2. * br_h.bb * br_h.zz * br_zz.qqll,
    bbzzllll=2. * br_h.bb * br_h.zz * br_zz.llll,
    bbtautau=2. * br_h.bb * br_h.tautau,
    bbgg=2. * br_h.bb * br_h.gg,
)

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
