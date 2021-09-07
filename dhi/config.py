# coding: utf-8

"""
Constants such as cross sections and branchings, and common configs such as labels and colors.
"""

from dhi.util import DotDict, ROOTColorGetter


# branching ratios at m_H = 125 GeV, using the decay mode naming scheme suggested by HIG and HComb
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
    tt=0.06272,
    zz=0.02619,
    gg=0.002270,
)
br_ww = DotDict(
    qqqq=br_w.qq ** 2.0,
    lvlv=br_w.lv ** 2.0,
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
    bbwwlvlv=2.0 * br_h.bb * br_h.ww * br_ww.lvlv,
    bbzz=2.0 * br_h.bb * br_h.zz,
    bbzzqqll=2.0 * br_h.bb * br_h.zz * br_zz.qqll,
    bbzzllll=2.0 * br_h.bb * br_h.zz * br_zz.llll,
    bbtt=2.0 * br_h.bb * br_h.tt,
    bbgg=2.0 * br_h.bb * br_h.gg,
    ttww=2.0 * br_h.tt * br_h.ww,
    ttzz=2.0 * br_h.tt * br_h.zz,
    tttt=br_h.tt ** 2.0,
    wwww=br_h.ww ** 2.0,
    zzzz=br_h.zz ** 2.0,
    wwzz=2.0 * br_h.ww * br_h.zz,
    wwgg=2.0 * br_h.ww * br_h.gg,
)
# aliases
br_hh["bbbb_boosted"] = br_hh.bbbb
br_hh["bbbb_boosted_ggf"] = br_hh.bbbb
br_hh["bbbb_boosted_vbf"] = br_hh.bbbb
br_hh["bbbb_boosted"] = br_hh.bbbb
br_hh["bbwwdl"] = br_hh.bbwwlvlv
br_hh["bbwwllvv"] = br_hh.bbwwlvlv
br_hh["bbwwsl"] = br_hh.bbwwqqlv
br_hh["bbzz4l"] = br_hh.bbzzllll

# HH branching names (TODO: find prettier abbreviations)
br_hh_names = DotDict(
    bbbb=r"4b",
    bbbb_low=r"4b, low $m_{HH}$",
    bbbb_boosted=r"4b, high $m_{HH}$",
    bbbb_boosted_ggf=r"4b #scale[0.75]{high $m_{HH}$, ggF}",
    bbbb_boosted_vbf=r"4b #scale[0.75]{high $m_{HH}$, VBF}",
    bbvv=r"bbVV",
    bbww=r"bbWW",
    bbwwqqlv=r"bbWW, qql$\nu$",
    bbwwlvlv=r"bbWW, 2l2$\nu$",
    bbzz=r"bbZZ",
    bbzzqqll=r"bbZZ, qqll",
    bbzzllll=r"bbZZ",
    bbtt=r"bb$\tau\tau$",
    bbgg=r"bb$\gamma\gamma$",
    ttww=r"WW$\tau\tau",
    ttzz=r"ZZ$\tau\tau",
    tttt=r"$\tau\tau\tau\tau$",
    wwww=r"WWWW",
    zzzz=r"ZZZZ",
    wwzz=r"WWZZ",
    wwgg=r"WW$\gamma\gamma$",
    multilepton="Multilepton",
)
# aliases
br_hh_names["bbwwdl"] = br_hh_names.bbwwlvlv
br_hh_names["bbwwllvv"] = br_hh_names.bbwwlvlv
br_hh_names["bbwwsl"] = br_hh_names.bbwwqqlv
br_hh_names["bbzz4l"] = br_hh_names.bbzzllll

# campaign labels, extended by combinations with HH branching names, and names itself
campaign_labels = DotDict({
    "2016": "2016 (13 TeV)",
    "2017": "2017 (13 TeV)",
    "2018": "2018 (13 TeV)",
    "run2": "Run 2 (13 TeV)",
})
for c, c_label in list(campaign_labels.items()):
    for b, b_label in br_hh_names.items():
        campaign_labels["{}_{}".format(b, c)] = "{}, {}".format(b_label, c_label)
campaign_labels.update(br_hh_names)

# poi defaults (value, range, points, taken from physics model) and labels
# note: C2V and CV are not following kappa notation and are upper case to be consistent to the model
poi_data = DotDict(
    r=DotDict(range=(-20.0, 20.0), label="r", sm_value=1.0),
    r_gghh=DotDict(range=(-20.0, 20.0), label="r_{gghh}", sm_value=1.0),
    r_qqhh=DotDict(range=(-20.0, 20.0), label="r_{qqhh}", sm_value=1.0),
    r_vhh=DotDict(range=(-20.0, 20.0), label="r_{vhh}", sm_value=1.0),
    kl=DotDict(range=(-30.0, 30.0), label=r"\kappa_{\lambda}", sm_value=1.0),
    kt=DotDict(range=(-10.0, 10.0), label=r"\kappa_{t}", sm_value=1.0),
    C2V=DotDict(range=(-10.0, 10.0), label=r"\kappa_{2V}", sm_value=1.0),
    CV=DotDict(range=(-10.0, 10.0), label=r"\kappa_{V}", sm_value=1.0),
    C2=DotDict(range=(-2.0, 3.0), label=r"c_{2}", sm_value=0.0),
    CG=DotDict(range=(-2.0, 2.0), label=r"c_{g}", sm_value=0.0),
    C2G=DotDict(range=(-2.0, 2.0), label=r"c_{2g}", sm_value=0.0),
)
# add "$" embedded labels
for poi, data in poi_data.items():
    data["label_math"] = "${}$".format(data.label)

# colors
colors = DotDict(
    root=ROOTColorGetter(
        black=1,
        white=10,
        white_trans_30=(10, 0.3),
        white_trans_70=(10, 0.7),
        grey=921,
        light_grey=920,
        light_grey_trans_50=(920, 0.5),
        dark_grey=13,
        dark_grey_trans_70=(13, 0.7),
        dark_grey_trans_50=(13, 0.5),
        dark_grey_trans_30=(13, 0.3),
        red=628,
        red_trans_50=(628, 0.5),
        blue=214,
        green=418,
        light_green=413,
        yellow=798,
        pink=222,
        cyan=7,
        orange=797,
        red_cream=46,
        blue_cream=38,
        blue_signal=(67, 118, 201),
        blue_signal_trans=(67, 118, 201, 0.5),
    ),
)

# color sequence for plots with multiple elements
color_sequence = ["blue", "red", "green", "grey", "pink", "cyan", "orange", "light_green", "yellow"]
color_sequence += 10 * ["grey"]

# marker sequence for plots with multiple elements
marker_sequence = [20, 21, 22, 23, 24, 25, 26, 32, 27, 33, 28, 34, 29, 30]
marker_sequence += 10 * [20]

# cumulative, inverse chi2 values in a mapping "n_dof -> n_sigma -> level"
# for the geometrical determination of errors of nll curves
# (computed with "sp.stats.chi2.ppf(g, n_dof)" with g being the gaussian intervals)
chi2_levels = {
    1: {1: 1.000, 2: 4.000},
    2: {1: 2.296, 2: 6.180},
    3: {1: 3.527, 2: 8.025},
}

# postfix after "CMS" labels in plots, shwon when the --paper flag is not used
cms_postfix = "Preliminary"
