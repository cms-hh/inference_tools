# coding: utf-8

"""
Constants such as cross sections and branchings, and common configs such as labels and colors.
"""

import os

import scipy as sp
import scipy.stats

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

# HH branching names
br_hh_names = DotDict(
    bbbb=r"bb bb",
    bbbb_low=r"bb bb, #scale[0.75]{low $m_{HH}$}",
    bbbb_boosted=r"bb bb, #scale[0.75]{high $m_{HH}$}",
    bbbb_boosted_ggf=r"bb bb #scale[0.75]{high $m_{HH}$, ggF}",
    bbbb_boosted_vbf=r"bb bb #scale[0.75]{high $m_{HH}$, VBF}",
    bbbb_vhh=r"bb bb, #scale[0.75]{VHH}",
    bbbb_all=r"bb bb",
    bbvv=r"bb VV",
    bbww=r"bb WW",
    bbwwqqlv=r"bb WW, qql$\nu$",
    bbwwlvlv=r"bb WW, 2l2$\nu$",
    bbzz=r"bb ZZ",
    bbzzqqll=r"bb ZZ, qqll",
    bbzzllll=r"bb ZZ, 4l",
    bbtt=r"bb $\tau\tau$",
    bbgg=r"bb $\gamma\gamma$",
    ttww=r"WW $\tau\tau",
    ttzz=r"ZZ $\tau\tau",
    tttt=r"$\tau\tau$ $\tau\tau$",
    wwww=r"WW WW",
    zzzz=r"ZZ ZZ",
    wwzz=r"WW ZZ",
    wwgg=r"WW $\gamma\gamma$",
    ggtt=r"$\gamma\gamma$ $\tau\tau$",
    multilepton="Multilepton",
)
# aliases
br_hh_names["bbwwdl"] = br_hh_names.bbwwlvlv
br_hh_names["bbwwllvv"] = br_hh_names.bbwwlvlv
br_hh_names["bbwwsl"] = br_hh_names.bbwwqqlv
br_hh_names["bbzz4l"] = br_hh_names.bbzzllll

# HH references
hh_references = DotDict(
    bbbb=r"Not yet",
    bbbb_low=r"Phys. Rev. Lett. 129 (2022) 081802",
    bbbb_boosted=r"Phys. Rev. Lett. 131 (2023) 041803",
    bbbb_boosted_ggf=r"CMS-PAS-B2G-21-001",
    bbbb_boosted_vbf=r"CMS-PAS-B2G-21-001",
    bbbb_all=r"Not yet",
    bbvv=r"Not yet",
    bbww=r"CMS-PAS-HIG-21-005",
    bbwwqqlv=r"Not yet",
    bbwwlvlv=r"Not yet",
    bbzz=r"Not yet",
    bbzzqqll=r"Not yet",
    bbzzllll=r"JHEP 06 (2023) 130",
    bbtt=r"Phys. Lett. B 842 (2023) 137531",
    bbgg=r"JHEP 03 (2021) 257",
    ttww=r"Not yet",
    ttzz=r"Not yet",
    tttt=r"Not yet",
    wwww=r"Not yet",
    zzzz=r"Not yet",
    wwzz=r"Not yet",
    wwgg=r"CMS-PAS-HIG-21-014",
    ggtt=r"CMS-PAS-HIG-22-012",
    multilepton="CMS-PAS-HIG-21-002",
)
hh_references["bbzz4l"] = hh_references.bbzzllll

# campaign labels, extended by combinations with HH branching names
# lumi values from https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=163
campaign_labels = DotDict({
    "2016": "36.3 fb^{-1} (2016, 13 TeV)",
    "2017": "41.5 fb^{-1} (2017, 13 TeV)",
    "2018": "59.8 fb^{-1} (2018, 13 TeV)",
    "run2": "138 fb^{-1} (13 TeV)",
})
for c, c_label in list(campaign_labels.items()):
    for b, b_label in br_hh_names.items():
        campaign_labels["{}_{}".format(b, c)] = "{}, {}".format(b_label, c_label)
campaign_labels.update(br_hh_names)

# poi defaults (value, range, points, taken from physics model) and labels
# note: C2V and CV are not following kappa notation and are upper case to be consistent to the model
poi_data = DotDict(
    r=DotDict(range=(-20.0, 20.0), label=r"$r$", sm_value=1.0),
    r_gghh=DotDict(range=(-20.0, 20.0), label=r"$r_{gghh}$", sm_value=1.0),
    r_qqhh=DotDict(range=(-20.0, 20.0), label=r"$r_{qqhh}$", sm_value=1.0),
    r_vhh=DotDict(range=(-20.0, 20.0), label=r"$r_{vhh}$", sm_value=1.0),
    kl=DotDict(range=(-30.0, 30.0), label=r"$\kappa_{\lambda}$", sm_value=1.0),
    kt=DotDict(range=(-10.0, 10.0), label=r"$\kappa_{t}$", sm_value=1.0),
    C2V=DotDict(range=(-10.0, 10.0), label=r"$\kappa_{2V}$", sm_value=1.0),
    CV=DotDict(range=(-10.0, 10.0), label=r"$\kappa_{V}$", sm_value=1.0),
    C2=DotDict(range=(-2.0, 3.0), label=r"$C_{2}$", sm_value=0.0),
    CG=DotDict(range=(-2.0, 2.0), label=r"$C_{g}$", sm_value=0.0),
    C2G=DotDict(range=(-2.0, 2.0), label=r"$C_{2g}$", sm_value=0.0),
    mhh=DotDict(range=(0.0, 3000.0), label=r"$m_{HH}$", unit="GeV"),
    A=DotDict(range=(0.0, 6.0), label=r"$\alpha$", sm_value=0.0),
    CA=DotDict(range=(0.0, 1.0), label=r"$|cos(\alpha)|$", sm_value=1.0),
    LA=DotDict(range=(-10.0, 10.0), label=r"$\lambda_{\alpha}$", sm_value=0.0),
    LE=DotDict(range=(-10.0, 10.0), label=r"$\lambda_{eff}=\lambda_{\alpha}-tan(\alpha)m_{2}/\nu$", sm_value=0.0),
    M2=DotDict(range=(0.0, 3000.0), label=r"$m_{2}$", sm_value=0.0, unit="GeV"),
    B=DotDict(range=(0.0, 6.0), label=r"$\beta$", sm_value=0.1),
    MHE=DotDict(range=(100.0, 3000.0), label=r"$m_{H}$", sm_value=125.01, unit="GeV"),
    MHP=DotDict(range=(0.0, 3000.0), label=r"$m_{H+}$", sm_value=0.0, unit="GeV"),
    MA=DotDict(range=(0.0, 3000.0), label=r"$m_{A}$", sm_value=0.0, unit="GeV"),
    Z6=DotDict(range=(-5.0, 5.0), label=r"$Z_{6}$", sm_value=0.0),
    TB=DotDict(range=(0.0, 1000.0), label=r"$tan(\beta)$", sm_value=1000.0),
    CBA=DotDict(range=(0.0, 1.0), label=r"$cos(\beta-\alpha)$", sm_value=0.0),
    LQ=DotDict(range=(-10.0, 10.0), label=r"$\lambda_{Q}$", sm_value=0.0),
    MQ=DotDict(range=(0.0, 3000.0), label=r"$m_{Q}$", sm_value=1000.0, unit="GeV"),
    XI=DotDict(range=(0.0, 1.0), label=r"$\xi$", sm_value=0.0),
    kl_EFT=DotDict(range=(-30.0, 30.0), label=r"$\kappa_{\lambda}$", sm_value=1.0),
    kt_EFT=DotDict(range=(-10.0, 10.0), label=r"$\kappa_{t}$", sm_value=1.0),
    C2_EFT=DotDict(range=(-2.0, 3.0), label=r"$C_{2}$", sm_value=0.0),
    cosbma=DotDict(range=(0.0, 10.0), label=r"$cos(\beta-\alpha)$", sm_value=0.0),
    tanbeta=DotDict(range=(0.0, 10.0), label=r"$tan(\beta)$", sm_value=10.0),
)

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
        bright_red=632,
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
        purple=881,
        brazil_yellow=800,  # kOrange
        brazil_green=417,  # kGreen + 1
    ),
)

# color sequence for plots with multiple elements
color_sequence = [
    "blue", "green", "orange", "cyan", "red_cream", "grey", "pink", "light_green", "yellow",
]
color_sequence += 10 * ["grey"]

# marker sequence for plots with multiple elements
marker_sequence = [20, 21, 22, 23, 24, 25, 26, 32, 27, 33, 28, 34, 29, 30]
marker_sequence += 10 * [20]

# colors per entry in br_hh_names for deterministic channel colors
br_hh_colors = DotDict(
    root=DotDict(
        bbbb="blue",
        bbbb_boosted="pink",
        bbbb_boosted_ggf="pink",
        bbbb_boosted_vbf="cyan",
        bbww="yellow",
        bbzz="orange",
        bbtt="red",
        bbgg="green",
        wwgg="light_green",
        multilepton="purple",
        Combined="grey",
    ),
)
# aliases
br_hh_colors.root["Combination"] = br_hh_colors.root.Combined
br_hh_colors.root["bbzz4l"] = br_hh_colors.root.bbzz
br_hh_colors.root["bbbb_low"] = br_hh_colors.root.bbbb

# cumulative, inverse chi2 values in a mapping "n_dof -> n_sigma -> level"
# for the geometrical determination of errors of nll curves
# (computed with "sp.stats.chi2.ppf(g, n_dof)" with g being the gaussian intervals)
# chi2_levels = {
#     1: {1: 1.000, 2: 4.000},
#     2: {1: 2.296, 2: 6.180},
#     ...
# }
get_gaus_interval = lambda sigma: 2 * sp.stats.norm.cdf(sigma) - 1.0
get_chi2_level = lambda sigma, ndof: sp.stats.chi2.ppf(get_gaus_interval(sigma), ndof)
get_chi2_level_from_cl = lambda cl, ndof: sp.stats.chi2.ppf(cl, ndof)
chi2_levels = {
    ndof: {sigma: get_chi2_level(sigma, ndof) for sigma in range(1, 8 + 1)}
    for ndof in range(1, 3 + 1)
}

# default postfix after "CMS" labels in plots
cms_postfix = os.getenv("DHI_CMS_POSTFIX", "Work in progress")

# shorthands for EFT benchmark labels and groups
bm_labels = DotDict()
# JHEP04
for bm in ["1", "2", "3", "4", "5", "6", "7", "8", "8a", "9", "10", "11", "12"]:
    bm_labels["JHEP04BM{}".format(bm)] = (bm, "JHEP04")
# JHEP03
for bm in ["1", "2", "3", "4", "5", "6", "7"]:
    bm_labels["JHEP03BM{}".format(bm)] = (bm, "JHEP03")
# others
bm_labels["SM"] = ("SM", "")

# SH procs
single_higgs_processes = [
    "ggH", "qqH", "ttH", "ZH", "WH", "VH", "tHW", "tHq",
]
