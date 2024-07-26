# coding: utf-8

"""
Custom HH physics model implementing gluon gluon fusion (ggf / gghh), vector boson fusion
(vbf / qqhh) and V boson associated production (vhh) modes.

This model is meant to be used for both legacy run 2, re-analyzed run 2, run 3 analyses, and
combinations across them. It is intended to be a full superset of the previous HH model which was
use for legacy run 2 only. In introduces an extended naming scheme through updated HH formula
classes, allowing / enforcing the inclusion of the center-of-mass energy in the sample labels
according to the following rules:

- ggHH_kl_1_kt_1:
    implicit run 2 sample, should be normalized to old rules (NLO) so that the NLO -> NNLO scaling
    is still done by the model; this ensures that old run 2 datacards can still be used and combined
- ggHH_kl_1_kt_1_13p0TeV:
    explicit run 2 sample, should be normalized to new rules (NNLO), so no scaling is done by the
    model
- ggHH_kl_1_kt_1_13p6TeV:
    explicit run 3 sample, should be normalized to new rules (NNLO) as above

Same rules apply for vbf and vhh samples.

Authors:
    - Marcel Rieger
    - Fabio Monti
"""

__all__ = [
    # samples
    "HHSample", "GGFSample", "VBFSample", "VHHSample", "ggf_samples", "vbf_samples", "vhh_samples",
    # formulae
    "HHFormula", "GGFFormula", "VBFFormula", "VHHFormula",
    # br and h scaling
    "SM_HIGG_DECAYS", "SM_HIGG_PROD", "coeffs_br", "cxs", "ewk", "dzh", "HBRScaler",
    # model
    "HHModelBase", "HHModel", "create_model", "model_default_run2l", "model_default_run2",
    "model_default_run3", "model_default_run2l3", "model_default_run23",
    # naming convention helpers
    "parse_hh_process", "parse_hh_ggf_process", "parse_hh_vbf_process", "parse_hh_vhh_process",
    "parse_h_process",
    # xsec helpers
    "ggf_k_factor_13p0TeV", "ggf_kl_coeffs", "create_ggf_xsec_str", "create_ggf_xsec_func",
    "create_vbf_xsec_func", "create_vhh_xsec_func", "create_hh_xsec_func",
]

import os
import re
from collections import OrderedDict, defaultdict

import sympy

from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModelBase
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder


no_value = object()

# default data directory of the SMHiggsBuilder, as used by the HBRScaler
default_data_dir = None
if "CMSSW_BASE" in os.environ:
    default_data_dir = "$CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg"


####################################################################################################
# Constants and external values
####################################################################################################

# single H production modes that are supported in the scaling
SM_HIGG_PROD = ["ggZH", "tHq", "tHW", "ggH", "qqH", "ZH", "WH", "VH", "ttH"]

# H decay names that are supported in the br scaling
SM_HIGG_DECAYS = ["hww", "hzz", "hgg", "htt", "hbb", "hzg", "hmm", "hcc", "hgluglu", "hss"]

# ggf NLO -> NNLO k-factor for 13p0TeV (and only needed there!)
ggf_k_factor_13p0TeV = 1.115

# VBF NLO -> N3LO k-factors (computed from cross section from twiki and XSDB value)
vbf_k_factor_13p0TeV = 1.687 / 1.626
vbf_k_factor_13p6TeV = 1.874 / 1.912

# coefficients for modeling the kl dependence of the ggf cross section (in fb) and its uncertainty
# (a0, a1, a2) -> a0 + a1 * kl + a2 * kl**2
# values from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGHH?rev=70
ggf_kl_coeffs = {
    # nlo coefficients, corresponding to _old_ normalization used to create legacy run2 datacards
    # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGHH?rev=65
    "nlo": {
        "13p0TeV": (62.5339, -44.3231, 9.6340),
    },
    # updated nnlo coefficients
    # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?rev=98
    "nnlo": {
        "13p0TeV": (68.5624, -48.3673, 10.5635),
        "13p6TeV": (75.7617, -53.2855, 11.6126),
    },
    # QCDscale + mtop uncertainty coefficients
    "unc_u": {
        "13p0TeV": (75.4551, -55.4010, 12.4555),
        "13p6TeV": (83.3897, -61.0213, 13.6898),
    },
    "unc_d": {
        "13p0TeV": (56.5063, -41.9131, 9.30669),
        "13p6TeV": (62.4328, -46.1854, 10.2342),
    },
}

# coefficients for the BR scaling with kl
# formula from https://arxiv.org/abs/1709.08649, Eq 22
# values from https://arxiv.org/pdf/1607.04251.pdf
coeffs_br = {
    "hgg": 0.49e-2,
    "hzz": 0.83e-2,
    "hww": 0.73e-2,
    "hgluglu": 0.66e-2,
    "htt": 0,
    "hbb": 0,
    "hcc": 0,
    "hss": 0,
    "hmm": 0,
}

# coefficients for the single H scaling with kl
# WH and ZH coeff are very similar, so use their average
cxs = {
    "13p0TeV": {
        "ggH": 0.66e-2,
        "qqH": 0.64e-2,
        "WH": 1.03e-2,
        "ZH": 1.19e-2,
        "ttH": 3.51e-2,
        "VH": (0.5 * (1.03e-2 + 1.19e-2)),
    },
    # TODO: these values are bare copies of the 13p0TeV values, need to update!
    "13p6TeV": {
        "ggH": 0.66e-2,
        "qqH": 0.64e-2,
        "WH": 1.03e-2,
        "ZH": 1.19e-2,
        "ttH": 3.51e-2,
        "VH": (0.5 * (1.03e-2 + 1.19e-2)),
    },
}
ewk = {
    "13p0TeV": {
        "ggH": 1.049,
        "qqH": 0.932,
        "WH": 0.93,
        "ZH": 0.947,
        "ttH": 1.014,
        "VH": (0.5 * (0.93 + 0.947)),
    },
    # TODO: these values are bare copies of the 13p0TeV values, need to update!
    "13p6TeV": {
        "ggH": 1.049,
        "qqH": 0.932,
        "WH": 0.93,
        "ZH": 0.947,
        "ttH": 1.014,
        "VH": (0.5 * (0.93 + 0.947)),
    },
}
dzh = {
    "13p0TeV": -1.536e-3,
    # TODO: bare copy of the 13p0TeV value, need to update! (if ecm dependent in the first place)
    "13p6TeV": -1.536e-3,
}


####################################################################################################
# Samples
####################################################################################################

class HHSample(object):
    """
    Base class for samples that currently only stores a cross section *xs* in pb and a sample
    *label*.
    """

    # regular expression to check sample labels
    # to be overwritten in subclasses
    label_re = None

    # known ecm values (as a string for simpler treatment) for which model
    # behavior is implemented
    known_ecms = {None, "13p0TeV", "13p6TeV"}

    def __init__(self, xs, label):
        super().__init__()

        # check if the label matches the format
        m = re.match(self.label_re, label)
        if not m:
            raise Exception(
                f"{self.__class__.__name__} label '{label}' does not match configured format "
                f"{self.label_re}",
            )

        # extract additional info
        ecm = m.group("ecm")
        ecm = ecm[1:] if ecm else None

        # check the ecm value
        if ecm not in self.known_ecms:
            raise ValueError(
                f"unknown ecm value '{ecm}' for sample '{label}', must be one of "
                f"{self.known_ecms}",
            )

        self.xs = xs
        self.label = label
        self.ecm = ecm
        self.effective_ecm = ecm or "13p0TeV"

    def matches_process(self, process):
        """
        Returns *True* if *process* matches the sample :py:attr:`label`, and *False* otherwise.
        """
        return process == self.label or process.startswith(self.label + "_")

    @property
    def key(self):
        """
        Returns a tuple containing the values of couplings in a fixed order.
        To be overwritten in subclasses
        """
        raise NotImplementedError


class GGFSample(HHSample):
    """
    Class describing ggf samples, characterized by values of *kl* and *kt*.
    """

    # label format
    label_re = r"^ggHH_kl_([pm0-9]+)_kt_([pm0-9]+)(?P<ecm>|_13p\dTeV)$"

    def __init__(self, kl, kt, xs, label):
        super().__init__(xs, label)

        self.kl = kl
        self.kt = kt

    @property
    def key(self):
        return (self.kl, self.kt, self.ecm)


class VBFSample(HHSample):
    """
    Class describing vbf samples, characterized by values of *CV* (kV), *C2V* (k2V) and *kl*.
    """

    # label format
    label_re = r"^qqHH_CV_([pm0-9]+)_C2V_([pm0-9]+)_kl_([pm0-9]+)(?P<ecm>|_13p\dTeV)$"

    def __init__(self, CV, C2V, kl, xs, label):
        super().__init__(xs, label)

        self.CV = CV
        self.C2V = C2V
        self.kl = kl

    @property
    def key(self):
        return (self.CV, self.C2V, self.kl, self.ecm)


class VHHSample(HHSample):
    """
    Class describing vhh samples, characterized by values of *CV* (kV), *C2V* (k2V) and *kl*.
    """

    # label format
    label_re = r"^VHH_CV_([pm0-9]+)_C2V_([pm0-9]+)_kl_([pm0-9]+)(?P<ecm>|_13p\dTeV)$"

    def __init__(self, CV, C2V, kl, xs, label):
        super().__init__(xs, label)

        self.CV = CV
        self.C2V = C2V
        self.kl = kl

    @property
    def key(self):
        return (self.CV, self.C2V, self.kl, self.ecm)


# helper to create functions that add samples to specific dicts
def _create_add_sample_func(sample_cls, samples_dict):
    def add_sample(*args, **kwargs):
        sample = sample_cls(*args, **kwargs)
        samples_dict[sample.key] = sample
    return add_sample


# helper to cast a string into an integer of float
def try_number(s):
    if not isinstance(s, str):
        return s

    try:
        return int(s)
    except:
        pass

    try:
        return float(s)
    except:
        pass

    return s


# implicit and explicit run 2 ggf samples
# NNLO cross sections from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?rev=98
ggf_samples = OrderedDict()
add_ggf_sample = _create_add_sample_func(GGFSample, ggf_samples)
for ecm in ["", "_13p0TeV"]:
    add_ggf_sample(kl=1.0, kt=1.0, xs=0.03077, label=f"ggHH_kl_1_kt_1{ecm}")
    add_ggf_sample(kl=0.0, kt=1.0, xs=0.06856, label=f"ggHH_kl_0_kt_1{ecm}")
    add_ggf_sample(kl=2.45, kt=1.0, xs=0.01347, label=f"ggHH_kl_2p45_kt_1{ecm}")
    add_ggf_sample(kl=5.0, kt=1.0, xs=0.09082, label=f"ggHH_kl_5_kt_1{ecm}")

# explicit run 3 ggf samples
# NNLO cross sections from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?rev=98
add_ggf_sample(kl=1.0, kt=1.0, xs=0.03413, label="ggHH_kl_1_kt_1_13p6TeV")
add_ggf_sample(kl=0.0, kt=1.0, xs=0.07575, label="ggHH_kl_0_kt_1_13p6TeV")
add_ggf_sample(kl=2.45, kt=1.0, xs=0.01492, label="ggHH_kl_2p45_kt_1_13p6TeV")
add_ggf_sample(kl=5.0, kt=1.0, xs=0.09965, label="ggHH_kl_5_kt_1_13p6TeV")

# implicit and explicit run 2 vbf samples
# cross sections were determined will be following strategy:
# - SM value (N3LO) taken from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?rev=98
# - no values for other coupling combinations are among the recommendations so use a k-factor
# - this k-factor is determined as the ratio between the SM cross section at N3LO and NLO from XSDB
# - it is then applied to the NLO values from XSDB to the other couplings
vbf_samples = OrderedDict()
add_vbf_sample = _create_add_sample_func(VBFSample, vbf_samples)
for ecm in ["", "_13p0TeV"]:
    add_vbf_sample(CV=1.0, C2V=1.0, kl=1.0, xs=0.001687, label=f"qqHH_CV_1_C2V_1_kl_1{ecm}")
    add_vbf_sample(CV=1.0, C2V=1.0, kl=0.0, xs=vbf_k_factor_13p0TeV * 0.004337, label=f"qqHH_CV_1_C2V_1_kl_0{ecm}")
    add_vbf_sample(CV=1.0, C2V=1.0, kl=2.0, xs=vbf_k_factor_13p0TeV * 0.001325, label=f"qqHH_CV_1_C2V_1_kl_2{ecm}")
    add_vbf_sample(CV=1.0, C2V=0.0, kl=1.0, xs=vbf_k_factor_13p0TeV * 0.026080, label=f"qqHH_CV_1_C2V_0_kl_1{ecm}")
    add_vbf_sample(CV=1.0, C2V=2.0, kl=1.0, xs=vbf_k_factor_13p0TeV * 0.014050, label=f"qqHH_CV_1_C2V_2_kl_1{ecm}")
    add_vbf_sample(CV=0.5, C2V=1.0, kl=1.0, xs=vbf_k_factor_13p0TeV * 0.010470, label=f"qqHH_CV_0p5_C2V_1_kl_1{ecm}")
    add_vbf_sample(CV=1.5, C2V=1.0, kl=1.0, xs=vbf_k_factor_13p0TeV * 0.063990, label=f"qqHH_CV_1p5_C2V_1_kl_1{ecm}")

# explicit run 3 vbf samples (different basis of points)
add_vbf_sample(CV=1.0, C2V=1.0, kl=1.0, xs=0.001874, label="qqHH_CV_1_C2V_1_kl_1_13p6TeV")
add_vbf_sample(CV=1.0, C2V=1.0, kl=2.0, xs=0.001593, label="qqHH_CV_1_C2V_1_kl_2_13p6TeV")
add_vbf_sample(CV=1.0, C2V=0.0, kl=1.0, xs=0.029320, label="qqHH_CV_1_C2V_0_kl_1_13p6TeV")
add_vbf_sample(CV=1.0, C2V=2.0, kl=1.0, xs=0.015710, label="qqHH_CV_1_C2V_2_kl_1_13p6TeV")
add_vbf_sample(CV=1.74, C2V=1.37, kl=14.4, xs=0.3954, label="qqHH_CV_1p74_C2V_1p37_kl_14p4_13p6TeV")
add_vbf_sample(CV=-0.012, C2V=0.03, kl=10.2, xs=0.00001257, label="qqHH_CV_m0p012_C2V_0p03_kl_10p2_13p6TeV")
add_vbf_sample(CV=-0.758, C2V=1.44, kl=-19.3, xs=0.3550, label="qqHH_CV_m0p758_C2V_1p44_kl_m19p3_13p6TeV")
add_vbf_sample(CV=-0.962, C2V=0.959, kl=-1.43, xs=0.001114, label="qqHH_CV_m0p962_C2V_0p959_kl_m1p43_13p6TeV")
add_vbf_sample(CV=-1.21, C2V=1.94, kl=-0.94, xs=0.003753, label="qqHH_CV_m1p21_C2V_1p94_kl_m0p94_13p6TeV")
add_vbf_sample(CV=-1.6, C2V=2.72, kl=-1.36, xs=0.01156, label="qqHH_CV_m1p6_C2V_2p72_kl_m1p36_13p6TeV")
add_vbf_sample(CV=-1.83, C2V=3.57, kl=-3.39, xs=0.01665, label="qqHH_CV_m1p83_C2V_3p57_kl_m3p39_13p6TeV")
add_vbf_sample(CV=-2.12, C2V=3.87, kl=-5.96, xs=0.006719, label="qqHH_CV_m2p12_C2V_3p87_kl_m5p96_13p6TeV")

# implicit and explicit run 2 vhh samples
# cross section values are NLO WHH + NNLO ZHH (no k-factor applied)
# TODO: there might be updates for these values, as well as 13p6TeV values
vhh_samples = OrderedDict()
add_vhh_sample = _create_add_sample_func(VHHSample, vhh_samples)
for ecm in ["", "_13p0TeV"]:
    add_vhh_sample(CV=1.0, C2V=1.0, kl=1.0, xs=0.0008850, label=f"VHH_CV_1_C2V_1_kl_1{ecm}")
    add_vhh_sample(CV=1.0, C2V=1.0, kl=2.0, xs=0.0014405, label=f"VHH_CV_1_C2V_1_kl_2{ecm}")
    add_vhh_sample(CV=1.0, C2V=0.0, kl=1.0, xs=0.0003182, label=f"VHH_CV_1_C2V_0_kl_1{ecm}")
    add_vhh_sample(CV=1.0, C2V=2.0, kl=1.0, xs=0.0023437, label=f"VHH_CV_1_C2V_2_kl_1{ecm}")
    add_vhh_sample(CV=0.5, C2V=1.0, kl=1.0, xs=0.0005946, label=f"VHH_CV_0p5_C2V_1_kl_1{ecm}")
    add_vhh_sample(CV=1.5, C2V=1.0, kl=1.0, xs=0.0019173, label=f"VHH_CV_1p5_C2V_1_kl_1{ecm}")
    add_vhh_sample(CV=1.0, C2V=1.0, kl=0.0, xs=0.0005127, label=f"VHH_CV_1_C2V_1_kl_0{ecm}")
    add_vhh_sample(CV=1.0, C2V=1.0, kl=20.0, xs=0.0428974, label=f"VHH_CV_1_C2V_1_kl_20{ecm}")


####################################################################################################
# symbolic cross section formulae
####################################################################################################

class HHFormula(object):
    """
    Base class for scaling formulae used to compute cross sections of various samples, depending on
    various coupling strengths given in a list *samples* containing :py:class:`Sample` instances of
    a specific type. All computed values are in units of pb.
    """

    # to be overwritten in subclasses
    sample_cls = None
    min_samples = None
    r_poi = None
    couplings = None

    @classmethod
    def check_samples(cls, samples):
        ecm_values = set()

        for sample in samples:
            if not isinstance(sample, cls.sample_cls):
                raise ValueError(
                    f"{cls.__name__} expects samples to be of type {cls.sample_cls}, but got "
                    f"{sample}",
                )
            ecm_values.add(sample.ecm)

        if len(samples) < cls.min_samples:
            raise ValueError(
                f"{cls.__name__} expects at least {cls.min_samples} samples, but got {len(samples)}",
            )

        if len(ecm_values) != 1:
            raise ValueError(
                f"{cls.__name__} expects all samples to have the same center-of-mass energy, but "
                f"got {len(ecm_values)} different values: {ecm_values}",
            )

        return {"ecm": ecm_values.pop()}

    def __init__(self, samples):
        super().__init__()

        # check sample classes
        check_data = self.check_samples(samples)

        # store samples and ecm value
        self.samples = list(samples)
        self.ecm = check_data["ecm"]
        self.effective_ecm = ecm or "13p0TeV"

        # symbolic expressions
        self.M = None  # the matrix to be inverted
        self.coeffs = None  # scaling coefficients per sample
        self.sigma = None  # cross section

        # eagerly build all matrix, coefficient and cross section expressions
        self.build_expressions()

    @property
    def n_samples(self):
        return len(self.samples)

    def build_expressions(self):
        """
        This method is supposed to define the invertible matrix :py:attr:`M`, as well as symbolic
        expressions of the per-sample coefficients :py:attr:`coeffs` and the cross section
        :py:attr:`sigma` itself.
        To be implemented by subclasses.
        """
        raise NotImplementedError


class GGFFormula(HHFormula):
    """
    Scaling formula for ggf samples, based on a n_samples x 3 matrix.
    """

    sample_cls = GGFSample
    min_samples = 3
    r_poi = "r_gghh"
    couplings = ["kl", "kt"]

    def build_expressions(self):
        # define the matrix with three scalings - box, triangle, interf
        self.M = sympy.Matrix([
            [
                sample.kt**4,
                sample.kt**2 * sample.kl**2,
                sample.kt**3 * sample.kl,
            ]
            for i, sample in enumerate(self.samples)
        ])

        # the vector of couplings
        kl, kt = sympy.symbols("kl kt")
        c = sympy.Matrix([
            [kt**4],
            [kt**2 * kl**2],
            [kt**3 * kl],
        ])

        # the vector of symbolic sample cross sections
        s = sympy.Matrix([
            [sympy.Symbol(f"xs{i}")]
            for i in range(self.n_samples)
        ])

        # actual computation, i.e., matrix inversion and multiplications with vectors
        M_inv = self.M.pinv()
        self.coeffs = c.transpose() * M_inv
        self.sigma = self.coeffs * s


class VBFFormula(HHFormula):
    """
    Scaling formula for vbf samples, based on a n_samples x 6 matrix.
    """

    sample_cls = VBFSample
    min_samples = 6
    r_poi = "r_qqhh"
    couplings = ["CV", "C2V", "kl"]

    def build_expressions(self):
        # define the matrix with three scalings - box, triangle, interf
        self.M = sympy.Matrix([
            [
                sample.CV**2 * sample.kl**2,
                sample.CV**4,
                sample.C2V**2,
                sample.CV**3 * sample.kl,
                sample.CV * sample.C2V * sample.kl,
                sample.CV**2 * sample.C2V,
            ]
            for i, sample in enumerate(self.samples)
        ])

        # the vector of couplings
        CV, C2V, kl = sympy.symbols("CV C2V kl")
        c = sympy.Matrix([
            [CV**2 * kl**2],
            [CV**4],
            [C2V**2],
            [CV**3 * kl],
            [CV * C2V * kl],
            [CV**2 * C2V],
        ])

        # the vector of symbolic sample cross sections
        s = sympy.Matrix([
            [sympy.Symbol(f"xs{i}")]
            for i in range(self.n_samples)
        ])

        # actual computation, i.e., matrix inversion and multiplications with vectors
        M_inv = self.M.pinv()
        self.coeffs = c.transpose() * M_inv
        self.sigma = self.coeffs * s


class VHHFormula(VBFFormula):
    """
    Scaling formula for vhh samples, based on a n_samples x 6 matrix. Currently, this formula is
    neglecting differences between W and Z bosons which makes it identical to the
    :py:class:`VBFFormula`.
    """

    sample_cls = VHHSample
    min_samples = 6
    r_poi = "r_vhh"
    couplings = ["CV", "C2V", "kl"]


####################################################################################################
# BR and single H scaling
####################################################################################################

class HBRScaler(object):
    """
    Class to produce BR scalings for anomalous couplings and XS * BR scalings for both H and HH.
    """

    REQUIRED_POIS = ["kl", "kt", "CV"]

    def __init__(self, model_builder, scale_br=False, scale_h=False, data_dir=default_data_dir):
        super().__init__()

        # setup and store combine related objects
        self.model_builder = model_builder
        data_dir = data_dir and os.path.expandvars(data_dir)
        self.higgs_builder = SMHiggsBuilder(self.model_builder, datadir=data_dir)

        # store names of expressions that model the scalings
        self.br_scalings = []
        self.h_scalings = []

        # build expressions right away
        if scale_br:
            self.build_br_scalings()
        if scale_h:
            self.build_h_scalings()

    def make_expr(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.factory_`.
        """
        return self.model_builder.factory_(*args, **kwargs)

    def get_expr(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.out.function`.
        """
        return self.model_builder.out.function(*args, **kwargs)

    def make_var(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.doVar`.
        """
        return self.model_builder.doVar(*args, **kwargs)

    def get_var(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.out.var`.
        """
        return self.model_builder.out.var(*args, **kwargs)

    def make_constant(self, value):
        """
        The new RooFit version seems to have dropped support for injecting bare values into formulas
        using positional expression arguments. This method takes an integer or float *value* and
        converts it into a constant, named singleton value that can be used instead. The name of the
        variable is returned. Example:

        .. code-block:: python

            make_constant(1)
            # creates a RooFit variable "hh_const_1", referring to 1

            make_constant(1.5)
            # creates a RooFit variable "hh_const_1p5", referring to 1.5

            make_constant(-1.5)
            # creates a RooFit variable "hh_const_m1p5", referring to -1.5
        """
        # build the name
        if isinstance(value, int):
            s = str(value)
        elif isinstance(value, float):
            s = str(value).replace("-", "m").replace(".", "p")
        else:
            raise Exception(f"cannot build constant from value '{value}'")
        name = f"hh_const_{s}"

        # get or create ot
        c = self.get_var(name)
        if not c:
            self.make_var(f"{name}[{value}]")
            c = self.get_var(name)
            c.setConstant(True)

        return name

    def make_br(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`higgs_builder.makeBR`.
        """
        return self.higgs_builder.makeBR(*args, **kwargs)

    def make_scaling(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`higgs_builder.makeScaling`.
        """
        # replace all values with constants
        for key, value in list(kwargs.items()):
            num = try_number(value)
            if isinstance(num, (int, float)):
                kwargs[key] = self.make_constant(num)

        return self.higgs_builder.makeScaling(*args, **kwargs)

    def build_br_scalings(self):
        """
        Registers all expressions required for the BR scaling.
        """
        # ensure all br's of the H decay exist
        for d in SM_HIGG_DECAYS:
            self.make_br(d)

            # keep BR uncertainties frozen for now
            self.make_expr(f"HiggsDecayWidth_UncertaintyScaling_{d}[1.0]")

        # define resolved loops
        self.make_scaling("hgluglu", Cb="1", Ctop="kt")
        self.make_scaling("hgg", Cb="1", Ctop="kt", CW="CV", Ctau="1")
        self.make_scaling("hzg", Cb="1", Ctop="kt", CW="CV", Ctau="1")

        # BR scaling vs kl (https://arxiv.org/abs/1709.08649 Eq 22)
        for d, c1 in coeffs_br.items():
            self.make_expr(f"expr::kl_scalBR_{d}('(@0 - 1) * {c1}', kl)")

        # define partial widths as a function of kl, kt, and CV
        self.make_expr("expr::CVktkl_Gscal_Z('(@0 * @0 + @3) * @1 * @2', CV, SM_BR_hzz, HiggsDecayWidth_UncertaintyScaling_hzz, kl_scalBR_hzz)")  # noqa
        self.make_expr("expr::CVktkl_Gscal_W('(@0 * @0 + @3) * @1 * @2', CV, SM_BR_hww, HiggsDecayWidth_UncertaintyScaling_hww, kl_scalBR_hww)")  # noqa
        self.make_expr("expr::CVktkl_Gscal_tau('(1 + @4) * @0 * @2 + (1 + @5) * @1 * @3', SM_BR_htt, SM_BR_hmm, HiggsDecayWidth_UncertaintyScaling_htt, HiggsDecayWidth_UncertaintyScaling_hmm,kl_scalBR_htt, kl_scalBR_hmm)")  # noqa
        self.make_expr("expr::CVktkl_Gscal_top('(1 + @2) * @0 * @1', SM_BR_hcc, HiggsDecayWidth_UncertaintyScaling_hcc, kl_scalBR_hcc)")  # noqa
        self.make_expr("expr::CVktkl_Gscal_bottom('(1 + @3) * (@0 * @2 + @1)', SM_BR_hbb, SM_BR_hss, HiggsDecayWidth_UncertaintyScaling_hbb, kl_scalBR_hbb)")  # noqa
        self.make_expr("expr::CVktkl_Gscal_gluon('(@0 + @3) * @1 * @2', Scaling_hgluglu, SM_BR_hgluglu, HiggsDecayWidth_UncertaintyScaling_hgluglu, kl_scalBR_hgluglu)")  # noqa
        self.make_expr("expr::CVktkl_Gscal_gamma('(@0 + @6) * @1 * @4 + @2 * @3 * @5', Scaling_hgg, SM_BR_hgg, Scaling_hzg, SM_BR_hzg, HiggsDecayWidth_UncertaintyScaling_hgg, HiggsDecayWidth_UncertaintyScaling_hzg, kl_scalBR_hgg)")  # noqa

        # fix to have all BRs add up to unity
        self.make_expr("sum::CVktkl_SMBRs({})".format(", ".join(
            f"SM_BR_{d}" for d in SM_HIGG_DECAYS
        )))

        # total width, normalized to SM
        # (just the sum over the partial widths/SM total BR)
        self.make_expr("expr::CVktkl_Gscal_tot('(@0 + @1 + @2 + @3 + @4 + @5 + @6) / @7', CVktkl_Gscal_Z, CVktkl_Gscal_W, CVktkl_Gscal_tau, CVktkl_Gscal_top, CVktkl_Gscal_bottom, CVktkl_Gscal_gluon, CVktkl_Gscal_gamma, CVktkl_SMBRs)")  # noqa

        # BRs, normalized to SM
        # (scaling as (partial/partial_SM) / (total/total_SM))
        self.make_expr("expr::CVktkl_BRscal_hww('(@0 * @0 + @3) * @2 / @1', CV, CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hww, kl_scalBR_hww)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hzz('(@0 * @0 + @3) * @2 / @1', CV, CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hzz, kl_scalBR_hzz)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_htt('(1 + @2) * @1 / @0', CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_htt, kl_scalBR_htt)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hmm('(1 + @2) * @1 / @0', CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hmm, kl_scalBR_hmm)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hbb('(1 + @2) * @1 / @0', CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hbb, kl_scalBR_hbb)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hcc('(1 + @2) * @1 / @0', CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hcc, kl_scalBR_hcc)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hss('(1 + @2) * @1 / @0', CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hss, kl_scalBR_hss)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hgg('(@0 + @3) * @2 / @1', Scaling_hgg, CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hgg,kl_scalBR_hgg)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hzg('@0 * @2 / @1', Scaling_hzg, CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hzg)")  # noqa
        self.make_expr("expr::CVktkl_BRscal_hgluglu('(@0 + @3) * @2 / @1', Scaling_hgluglu, CVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hgluglu, kl_scalBR_hgluglu)")  # noqa

        # store the final scaling expression names
        for d in SM_HIGG_DECAYS:
            self.br_scalings.append(f"CVktkl_BRscal_{d}")

    def build_h_scalings(self):
        """
        Registers all expressions required for the single H scaling.
        """
        # get VBF, tHq, tHW, ggZH cross section and resolved loops
        self.make_scaling("qqH", CW="CV", CZ="CV")
        self.make_scaling("tHq", CW="CV", Ctop="kt")
        self.make_scaling("tHW", CW="CV", Ctop="kt")
        self.make_scaling("ggZH", CZ="CV", Ctop="kt", Cb="1")
        self.make_scaling("ggH", Cb="1", Ctop="kt", Cc="1")

        # create scalings for different production processes which require different formulae
        for ecm in ["13p0TeV", "13p6TeV"]:
            for p in ["ggH", "qqH"]:
                d = {"prod": p, "ecm": ecm, "cxs": cxs[ecm][p], "ewk": ewk[ecm][p], "dzh": dzh[ecm]}
                self.make_expr("expr::CVktkl_XSscal_{prod}_{ecm}('(@1 + (@0 - 1) * {cxs} / {ewk}) / ((1 - (@0 * @0 - 1) * {dzh}))', kl, Scaling_{prod}_{ecm})".format(**d))  # noqa
                self.make_expr("expr::CVktkl_pos_XSscal_{prod}_{ecm}('0. + @0 * (@0 > 0)', CVktkl_XSscal_{prod}_{ecm})".format(**d))  # noqa
                self.h_scalings.append("CVktkl_pos_XSscal_{prod}_{ecm}".format(**d))

            for p in ["ggZH", "tHq", "tHW"]:
                d = {"prod": p, "ecm": ecm}
                self.make_expr("expr::CVktkl_pos_XSscal_{prod}_{ecm}('0. + @0 * (@0 > 0)', Scaling_{prod}_{ecm})".format(**d))  # noqa
                self.h_scalings.append("CVktkl_pos_XSscal_{prod}_{ecm}".format(**d))

            for p in ["ZH", "WH", "VH"]:
                d = {"prod": p, "ecm": ecm, "cxs": cxs[ecm][p], "ewk": ewk[ecm][p], "dzh": dzh[ecm]}
                self.make_expr("expr::CVktkl_XSscal_{prod}_{ecm}('(@1 * @1 + (@0 - 1) * {cxs} / {ewk}) / ((1 - (@0 * @0 - 1) * {dzh}))', kl, CV)".format(**d))  # noqa
                self.make_expr("expr::CVktkl_pos_XSscal_{prod}_{ecm}('0. + @0 * (@0 > 0)', CVktkl_XSscal_{prod}_{ecm})".format(**d))  # noqa
                self.h_scalings.append("CVktkl_pos_XSscal_{prod}_{ecm}".format(**d))

            for p in ["ttH"]:
                d = {"prod": p, "ecm": ecm, "cxs": cxs[ecm][p], "ewk": ewk[ecm][p], "dzh": dzh[ecm]}
                self.make_expr("expr::CVktkl_XSscal_{prod}_{ecm}('(@1 * @1 + (@0 - 1) * {cxs} / {ewk}) / ((1 - (@0 * @0 - 1) * {dzh}))', kl, kt)".format(**d))  # noqa
                self.make_expr("expr::CVktkl_pos_XSscal_{prod}_{ecm}('0. + @0 * (@0 > 0)', CVktkl_XSscal_{prod}_{ecm})".format(**d))  # noqa
                self.h_scalings.append("CVktkl_pos_XSscal_{prod}_{ecm}".format(**d))

    def find_br_scalings(self, process, bin=None):
        """
        Given the name of a *process*, extracts the trailing decay string (separated by uncerscore)
        as well as names the matching br scaling expressions and returns them in a 2-tuple. Example:

        .. code-block:: python

            find_br_scalings("ggHH_hbbhtt")
            # -> ("hbbhtt", ["CVktkl_BRscal_hbb", "CVktkl_BRscal_htt"])
        """
        # to automatically parse the higgs decay from the process name, we need a fixed format,
        # "*_hXX[hYY[...]]Z" where hXX, hYY, etc must be in the offical SM_HIGG_DECAYS, and Z can be
        # an additional sub decay identifier that does not contain underscores
        br = process.split("_")[-1]

        # extract valid decays
        decays = []
        m = re.match("^(({})+)(.*)$".format("|".join(SM_HIGG_DECAYS)), br)
        if m:
            decays = re.findall("|".join(SM_HIGG_DECAYS), m.group(1))

        # find scaling names
        br_scalings = []
        for d in decays:
            for s in self.br_scalings:
                if s.endswith(f"_{d}"):
                    br_scalings.append(s)
                    break
            else:
                raise Exception(
                    f"unsupported BR scaling of decay {d} for process {process} (bin {bin})",
                )

        return br, br_scalings

    def find_h_scaling(self, process, bin=None):
        """
        Given the name of a *process*, extracts the leading single H name (separated by
        uncerscore) as well as names the matching H scaling expression and optionally the
        center-of-mass energy, and returns them in a 3-tuple. Example:

        .. code-block:: python

            find_h_scaling("ggH_hbb")  # implicitely using 13p0TeV
            # -> ("ggH", "13p0TeV", "CVktkl_pos_XSscal_ggH_13p0TeV")

            find_h_scaling("ttH_13p6TeV_hbb")
            # -> ("ttH", "13p6TeV", "CVktkl_pos_XSscal_ttH_13p6TeV")
        """
        # single H process names must have the format "PROD_*" where PROD must be in SM_HIGG_PROD
        # None is returned when this is not the case
        proc_data = parse_h_process(process, silent=True)
        if not proc_data:
            return None, None
        prod = proc_data["prod"]
        ecm = proc_data["ecm"] or "13p0TeV"

        for h_scaling in self.h_scalings:
            if h_scaling.endswith(f"_{prod}_{ecm}"):
                break
        else:
            raise Exception(f"unsupported H scaling of process {process} (bin {bin})")

        return prod, ecm, h_scaling

    def build_xsbr_scaling_hh(self, xs_scaling, process, bin=None):
        """
        Creates the full xs times br scaling expression for a HH *process* given the name of the
        current expression for scaling its cross section, and returns the new expression name.
        """
        br, br_scalings = self.find_br_scalings(process, bin=bin)

        # when no scalings were found, print a warning since this might be intentional and return,
        # when != 2 scalings were found, this is most likely an error
        if not br_scalings:
            print(
                f"WARNING: the HH process {process} (bin {bin}) does not contain valid decay "
                "strings to extract branching ratios to apply the scaling with model parameters",
            )
            return None
        elif len(br_scalings) != 2:
            raise Exception(
                f"the HH process {process} (bin {bin}) contains {len(br_scalings)} valid decay "
                "string(s) while two were expected",
            )

        # build the new scaling
        xsbr_scaling = f"{xs_scaling}_BRscal_{br}"
        if not self.get_expr(xsbr_scaling):
            self.make_expr(f"expr::{xsbr_scaling}('@0 * @1 * @2', {xs_scaling}, {br_scalings[0]}, {br_scalings[1]})")

        return xsbr_scaling

    def build_xsbr_scaling_h(self, xs_scaling, process, bin=None):
        """
        Creates the full xs times br scaling expression for a single H *process* given the name of
        the current expression for scaling its cross section, and returns the new expression name.
        """
        br, br_scalings = self.find_br_scalings(process, bin=bin)

        # when no scalings were found, print a warning since this might be intentional and return,
        # when != 1 scalings were found, this is most likely an error
        if not br_scalings:
            print(
                f"WARNING: the H process {process} (bin {bin}) does not contain valid decay "
                "strings to extract branching ratios to apply the scaling with model parameters",
            )
            return None
        elif len(br_scalings) != 1:
            raise Exception(
                f"the H process {process} (bin {bin}) contains {len(br_scalings)} valid decay "
                "string(s) while one was expected",
            )

        # build the new scaling
        xsbr_scaling = f"{xs_scaling}_BRscal_{br}"
        if not self.get_expr(xsbr_scaling):
            self.make_expr(f"expr::{xsbr_scaling}('@0 * @1', {xs_scaling}, {br_scalings[0]})")

        return xsbr_scaling


####################################################################################################
# process name parsing and validation helpers
####################################################################################################

def parse_hh_process(name, silent=False):
    if name.startswith("ggHH"):
        return parse_hh_ggf_process(name, silent=silent)
    if name.startswith("qqHH"):
        return parse_hh_vbf_process(name, silent=silent)
    if name.startswith("VHH"):
        return parse_hh_vhh_process(name, silent=silent)
    if silent:
        return None
    raise ValueError(f"unknown HH process name '{name}'")


def parse_hh_ggf_process(name, silent=False):
    # format: ggHH_kl_XX_kt_YY[_ECM][_*][_DECAY][*]
    m = re.match(rf"^{GGFSample.label_re[:-1]}(_.+)?(_h[^_]+)$", name)
    if m:
        kl, kt, ecm, opt, decay = m.groups()
        return {
            "prod": "ggf",
            "kl": kl,
            "kt": kt,
            "ecm": ecm and ecm[1:],
            "decay": decay and decay[1:],
            "opt": opt and opt[1:],
        }
    if silent:
        return None
    raise ValueError(f"invalid HH ggf process name '{name}'")


def parse_hh_vbf_process(name, silent=False):
    # format: qqHH_CV_XX_C2V_YY_kl_YY[_ECM][_*][_DECAY][*]
    m = re.match(rf"^{VBFSample.label_re[:-1]}(_.+)?(_h[^_]+)$", name)
    if m:
        cv, c2v, kl, ecm, opt, decay = m.groups()
        return {
            "prod": "vbf",
            "cv": cv,
            "c2v": c2v,
            "kl": kl,
            "ecm": ecm and ecm[1:],
            "decay": decay and decay[1:],
            "opt": opt and opt[1:],
        }
    if silent:
        return None
    raise ValueError(f"invalid HH vbf process name '{name}'")


def parse_hh_vhh_process(name, silent=False):
    # format: VHH_CV_XX_C2V_YY_kl_ZZ[_ECM][_*][_DECAY][*]
    m = re.match(rf"^{VHHSample.label_re[:-1]}(_.+)?(_h[^_]+)$", name)
    if m:
        cv, c2v, kl, ecm, opt, decay = m.groups()
        return {
            "prod": "vhh",
            "cv": cv,
            "c2v": c2v,
            "kl": kl,
            "ecm": ecm and ecm[1:],
            "decay": decay and decay[1:],
            "opt": opt and opt[1:],
        }
    if silent:
        return None
    raise ValueError(f"invalid HH vhh process name '{name}'")


def parse_h_process(name, silent=False):
    # format: PROD[_ECM][_*][_DECAY][*]
    m = re.match(rf"^({'|'.join(SM_HIGG_PROD)})(_13p\dTeV)?(_.+)?(_h[^_]+)$", name)
    if m:
        prod, ecm, opt, decay = m.groups()
        return {
            "prod": prod,
            "ecm": ecm and ecm[1:],
            "decay": decay and decay[1:],
            "opt": opt and opt[1:],
        }
    if silent:
        return None
    raise ValueError(f"invalid H process name '{name}'")


####################################################################################################
# model classes
####################################################################################################

class HHModelBase(PhysicsModelBase):
    """
    Base class for HH physics models providing a common interface for subclasses such as the default
    HH model or a potential EFT model (e.g. kt-kl-C2).
    """

    # pois with initial (SM) value, start and stop
    # to be defined by subclasses
    R_POIS = OrderedDict()
    K_POIS = OrderedDict()

    def __init__(self, name):
        super().__init__()

        # attributes
        self.name = name

        # names and values of physics options
        self.hh_options = OrderedDict()

        # actual r and k pois, depending on used formulae and profiling options, set in reset_pois
        self.r_pois = None
        self.k_pois = None

        # mapping of (formula, sample) -> expression name that models the linear sample scales
        self.r_expressions = {}

        # nested mapping of formula -> sample -> matched processes for book keeping
        self.hh_process_scales = defaultdict(lambda: defaultdict(set))

    def register_opt(self, name, default, is_flag=False):
        """
        Registers a physics option which is automatically parsed and set by
        :py:meth:`setPhysicsOptions`. Example:

        .. code-block:: python

            register_opt("myName", "some_default_value")
            # -> parses "--physics-option myName='other_value'"
        """
        self.hh_options[name] = {
            "value": default,
            "is_flag": is_flag,
        }

    def set_opt(self, name, value):
        """
        Sets the value of a physics option named *name*, previoulsy registered with
        :py:meth:`register_opt`, to *value*.
        """
        self.hh_options[name]["value"] = value

    def opt(self, name, default=no_value):
        """
        Helper to get the value of a physics option defined by *name* with an optional *default*
        value that is returned when no option with that *name* is registered.
        """
        if name in self.hh_options or default == no_value:
            return self.hh_options[name]["value"]

        return default

    def setPhysicsOptions(self, options):
        """
        Hook called by the super class to parse physics options received externally, e.g. via
        ``--physics-option`` or ``--PO``.
        """
        # split by "=" and check one by one
        pairs = [opt.split("=", 1) for opt in options if "=" in opt]
        for name, value in pairs:
            if name not in self.hh_options:
                print(f"[WARNING] unknown physics option '{name}'")
                continue

            if self.hh_options[name]["is_flag"]:
                # boolean flag
                value = value.lower() in ["yes", "true", "1"]
            else:
                # string value, catch special cases
                value = None if value.lower() in ["", "none"] else value

            self.set_opt(name, value)
            print(f"[INFO] using model option {name} = {value}")

        # since settings might have changed, reset pois again
        self.reset_pois()

    def reset_pois(self):
        """
        Sets the instance-level :py:attr:`r_pois` and :py:attr:`k_pois` based on registered
        formulae.
        """
        all_formulae = sum((list(f.values()) for f in self.get_formulae().values()), [])

        # r pois
        self.r_pois = OrderedDict()
        for p, v in self.R_POIS.items():
            keep = p == "r"
            keep |= any(p == formula.r_poi for formula in all_formulae)
            if keep:
                self.r_pois[p] = v

        # k pois
        self.k_pois = OrderedDict()
        for p, v in self.K_POIS.items():
            keep = any(p in formula.couplings for formula in all_formulae)
            if keep:
                self.k_pois[p] = v

    def get_formulae(self, xs_only=False, ecm=no_value):
        """
        Method that returns a dictionary of all used :py:class:`HHFormula` instances mapped to
        production mode specific names, such as ``"ggf_formula"`` or ``"vbf_formula"``.

        When no *ecm* value is requested, the returned dictionary is nested by the center-of-mass
        energy. When *ecm* is given, per production mode only the formulae for that energy are
        returned. When *xs_only* is *True*, only those formulae are returned that should enter cross
        section calculations.
        To be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def model_builder(self):
        # for compatibility
        return self.modelBuilder

    def make_expr(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.factory_`.
        """
        return self.model_builder.factory_(*args, **kwargs)

    def get_expr(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.out.function`.
        """
        return self.model_builder.out.function(*args, **kwargs)

    def make_var(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.doVar`.
        """
        return self.model_builder.doVar(*args, **kwargs)

    def get_var(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.out.var`.
        """
        return self.model_builder.out.var(*args, **kwargs)

    def make_set(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.doSet`.
        """
        return self.model_builder.doSet(*args, **kwargs)

    def done(self):
        """
        Hook called by the super class after the workspace is created.

        Here, we make sure that each formula will function as expected by checking that each of its
        samples is matched by exactly one process. When no processes was matched for _any_ sample of
        a formula, the scaling is disabled and the underlying r POI will have no effect.
        """
        super().done()

        errors = []
        for formula_key, formulae in self.get_formulae().items():
            for ecm, formula in formulae.items():
                print(f"\nmatching processes for {formula_key} at ecm {ecm}:")

                if not self.hh_process_scales[formula]:
                    print("  none")
                    continue

                # print matched process per expected sample
                max_len = max(len(sample.label) for sample in formula.samples)
                unmatched_samples = []
                for sample in formula.samples:
                    processes = self.hh_process_scales[formula][sample]
                    if not processes:
                        unmatched_samples.append(sample)
                    offset = " " * (max_len - len(sample.label))
                    procs_repr = ", ".join(processes) if processes else "MISSING"
                    print(f"  {sample.label}{offset} -> {procs_repr}")

                # complain about samples that were not matched by any process
                if len(unmatched_samples) not in [0, formula.n_samples]:
                    unmatched_samples_repr = ", ".join(sample.label for sample in unmatched_samples)
                    errors.append(
                        f"{len(unmatched_samples)} {formula_key} for ecm {ecm} samples were not "
                        f"matched by any process: {unmatched_samples_repr}",
                    )

        if errors:
            raise Exception("\n".join(errors))


class HHModel(HHModelBase):
    """
    Models the HH production as a linear sum of the input components for ggf, vbf and vhh processes.
    The following physics options are supported:

    - doNNLOscaling (bool)   : Convert ggf HH yields (that are given in NLO by convention) to NNLO.
                               Only applies to the GGFFormula for 13p0TeV.
    - doBRscaling (bool)     : Enable scaling Higgs branching ratios with model parameters.
    - doHscaling (bool)      : Enable scaling single Higgs cross sections with model parameters.
    - doklDependentUnc (bool): Add a theory uncertainty on ggf HH production that depends on model
                               parameters.
    - doProfileX (string)    : Either "flat" to enable the profiling of parameter X with a flat
      X in {rgghh,rqqhh,rvhh,  prior, or "gauss,FLOAT" (or "gauss,-FLOAT/+FLOAT") to use a gaussian
      kl,kt,CV,C2V}            (asymmetric) prior. In any case, X will be profiled and is hence
                               removed from the list of POIs.

    A string encoded boolean flag is interpreted as *True* when it is either ``"yes"``, ``"true"``
    or ``1`` (case-insensitive).
    """

    # pois with initial (SM) value, start and stop
    R_POIS = OrderedDict([
        ("r", (1, -20, 20)),
        ("r_gghh", (1, -20, 20)),
        ("r_qqhh", (1, -200, 200)),
        ("r_vhh", (1, -20, 20)),
    ])
    K_POIS = OrderedDict([
        ("kl", (1, -30, 30)),
        ("kt", (1, -10, 10)),
        ("CV", (1, -10, 10)),
        ("C2V", (1, -10, 10)),
    ])

    # the class of the HBRScaler to use
    h_br_scaler_cls = HBRScaler

    # formula classes to use
    ggf_formula_cls = GGFFormula
    vbf_formula_cls = VBFFormula
    vhh_formula_cls = VHHFormula

    def __init__(self, name, ggf_samples=None, vbf_samples=None, vhh_samples=None):
        super().__init__(name)

        # helper to create potentially multiple formulae of the same mode but different ecm's
        def build(formula_cls, samples):
            # split samples by ecm
            samples_by_ecm = defaultdict(list)
            for sample in samples:
                samples_by_ecm[sample.ecm].append(sample)
            return {ecm: formula_cls(samples) for ecm, samples in samples_by_ecm.items()}

        # attributes
        self.ggf_formulae = build(self.ggf_formula_cls, ggf_samples) if ggf_samples else None
        self.vbf_formulae = build(self.vbf_formula_cls, vbf_samples) if vbf_samples else None
        self.vhh_formulae = build(self.vhh_formula_cls, vhh_samples) if vhh_samples else None
        self.ggf_kl_dep_unc = "THU_HH"  # name for kl-dependent QCDscale + mtop uncertainty on ggf
        self.h_br_scaler = None  # initialized in create_scalings

        # register options
        self.register_opt("doNNLOscaling", True, is_flag=True)
        self.register_opt("doklDependentUnc", True, is_flag=True)
        self.register_opt("doBRscaling", True, is_flag=True)
        self.register_opt("doHscaling", True, is_flag=True)
        for p in list(self.R_POIS.keys()) + list(self.K_POIS.keys()):
            if p != "r":
                self.register_opt("doProfile" + p.replace("_", ""), None)

        # reset instance-level pois
        self.reset_pois()

    def get_effective_ecms(self):
        # determine which center of pass energies are requested through formulae
        ecms = set()
        for formulae in self.get_formulae().values():
            for formula in formulae.values():
                ecms.add(formula.effective_ecm)
        return sorted(ecms)

    @property
    def effective_ecm(self):
        ecms = self.get_effective_ecms()
        if len(ecms) != 1:
            raise ValueError(
                f"cannot determine single center-of-mass value; multiple different values were "
                f"found to be used by the formulae: {', '.join(ecms)}",
            )
        return ecms[0]

    def reset_pois(self):
        super().reset_pois()

        # remove profiled r pois
        for p in list(self.r_pois):
            if self.opt("doProfile" + p.replace("_", ""), False):
                del self.r_pois[p]

        # remove profiled k pois
        for p in list(self.k_pois):
            if self.opt("doProfile" + p.replace("_", ""), False):
                del self.k_pois[p]

    def get_formulae(self, xs_only=False, ecm=no_value):
        formulae = OrderedDict()

        def select_ecm(formulae):
            return formulae.copy() if ecm == no_value else formulae[ecm]

        if self.ggf_formulae:
            formulae["ggf_formula"] = select_ecm(self.ggf_formulae)
        if self.vbf_formulae:
            formulae["vbf_formula"] = select_ecm(self.vbf_formulae)
        if self.vhh_formulae:
            formulae["vhh_formula"] = select_ecm(self.vhh_formulae)

        return formulae

    def _create_ggf_xsec_str(self, *args, **kwargs):
        # forward to the modul-level implementation
        return create_ggf_xsec_str(*args, **kwargs)

    def _create_hh_xsec_func(self, *args, **kwargs):
        # forward to the modul-level implementation
        return create_hh_xsec_func(*args, **kwargs)

    def create_hh_xsec_func(self, ecm, **kwargs):
        """
        Returns a function that can be used to compute cross sections, based on all formulae
        returned by :py:meth:`get_formulae` with *xs_only* set to *True*.
        """
        assert isinstance(ecm, str)
        _kwargs = self.get_formulae(xs_only=True, ecm=ecm)
        _kwargs.update(kwargs)
        return self._create_hh_xsec_func(**_kwargs)

    def doParametersOfInterest(self):
        """
        Hook called by the super class to add parameters (of interest) to the model.

        Here, we add all r and k POIs depending on profiling options, define the POI group and
        re-initialize the MH parameter. By default, the main r POI will be the only floating one,
        whereas the others are either fixed or fully profiled.
        """
        # first, add all known r and k POIs
        for p in self.R_POIS:
            value, start, stop = self.r_pois.get(p, self.R_POIS[p])
            self.make_var(f"{p}[{value},{start},{stop}]")
        for p in self.K_POIS:
            value, start, stop = self.k_pois.get(p, self.K_POIS[p])
            self.make_var(f"{p}[{value},{start},{stop}]")

        # make certain r parameters pois, freeze all but the main r
        pois = []
        for p, (value, start, stop) in self.r_pois.items():
            if p != "r":
                self.get_var(p).setConstant(True)
            pois.append(p)

        # make certain coupling modifiers pois
        for p, (value, start, stop) in self.k_pois.items():
            self.get_var(p).setConstant(True)
            pois.append(p)

        # when the HBRScaler is used, make sure that its required pois are existing
        if self.opt("doBRscaling") or self.opt("doHscaling"):
            for p in self.h_br_scaler_cls.REQUIRED_POIS:
                if not self.get_var(p):
                    self.make_var(f"{p}[1]")
                    self.get_var(p).setConstant(True)

        # define the POI group
        self.make_set("POI", ",".join(pois))
        print(f"using POIs {','.join(pois)}")

        # set or redefine the MH variable on which some of the BRs depend
        if not self.options.mass:
            raise Exception(
                f"invalid mass value '{self.options.mass}', please provide a valid value using the "
                "--mass option",
            )
        if self.get_var("MH"):
            self.get_var("MH").removeRange()
            self.get_var("MH").setVal(self.options.mass)
        else:
            self.make_var(f"MH[{self.options.mass}]")
        self.get_var("MH").setConstant(True)

        # create cross section scaling functions
        self.create_scalings()

    def create_scalings(self):
        """
        Creates the scaling expressions for morphing the registered signal samples into a single
        shape.
        """
        # initialize the HBRScaler
        self.h_br_scaler = self.h_br_scaler_cls(
            self.model_builder,
            scale_br=self.opt("doBRscaling"),
            scale_h=self.opt("doHscaling"),
        )

        # get requested center-of-mass energies
        effective_ecms = self.get_effective_ecms()

        def pow_to_mul_string(expr):
            """
            Convert powers in a sympy expression to Muls (e.g. "a**2 => a*a") and return a string.
            """
            pows = list(expr.atoms(sympy.Pow))
            if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):
                raise ValueError(f"a power in '{expr}' contains a non-integer exponent")
            s = str(expr)
            for p in pows:
                base, exp = p.as_base_exp()
                repl = sympy.Mul(*(exp * [base]), evaluate=False)
                s = s.replace(str(p), str(repl))
            return s

        def replace_coupling(c, repl, s):
            """
            Replaces the coupling *c* by *repl* in a string *s*, ensuring that couplings consisting
            of a substring of *c* remain unchanged. Example:

            .. code-block:: python

                replace_coupling("kl", "@0", "3 * kl * kt")
                # -> "3 * @0 * kt"

                replace_coupling("c2", "@0", "3 * c2 * c2g")
                # -> "3 * @0 * c2g
                # note the unchanged "c2g"
            """
            return re.sub(r"{}([^0-9a-zA-Z])".format(c), r"{}\1".format(repl), s + " ")[:-1]

        # build scaling parameters for the kl-dependent QCDscale + mtop uncertainty for all required
        # center-of-mass energies
        if self.opt("doklDependentUnc"):
            self.create_ggf_kl_dep_unc(effective_ecms)

        # add sample scalings
        for formula in sum((list(f.values()) for f in self.get_formulae().values()), []):
            if isinstance(formula, GGFFormula):
                for sample, coeff in zip(formula.samples, formula.coeffs):
                    # create the expression that scales this particular sample based on the formula
                    name = f"f_scale_ggf_sample_{sample.label}"
                    expr = pow_to_mul_string(coeff)
                    for i, coupling in enumerate(formula.couplings):
                        expr = replace_coupling(coupling, f"@{i}", expr)
                    self.make_expr(f"expr::{name}('{expr}', {', '.join(formula.couplings)})")

                    # optionally multiply the theory uncertainty scaling to the expression
                    if self.opt("doklDependentUnc"):
                        new_name = f"{name}__kl_dep_unc"
                        self.make_expr(f"prod::{new_name}(scaling_{self.ggf_kl_dep_unc}_{formula.effective_ecm}, {name})")  # noqa
                        name = new_name

                    # optionally rescale to nnlo for legacy run 2 samples
                    # (expecting the normalization to be nlo*k initially)
                    if self.opt("doNNLOscaling") and not formula.ecm:
                        new_name = f"{name}__nlo2nnlo"
                        nlo_expr = self._create_ggf_xsec_str(formula.effective_ecm, "nlo", "@0")
                        nnlo_expr = self._create_ggf_xsec_str(formula.effective_ecm, "nnlo", "@0")
                        self.make_expr(f"expr::{new_name}('@1 * ({nnlo_expr}) / (({ggf_k_factor_13p0TeV}) * ({nlo_expr}))', kl, {name})")  # noqa
                        name = new_name

                    # scale it by the channel specific r POI
                    new_name = f"{name}__{formula.r_poi}"
                    self.make_expr(f"prod::{new_name}({formula.r_poi}, {name})")
                    name = new_name

                    # scale it by the common r POI
                    new_name = f"{name}__r"
                    self.make_expr(f"prod::{new_name}(r, {name})")
                    name = new_name

                    # store the final expression name
                    self.r_expressions[(formula, sample)] = name

            elif isinstance(formula, VBFFormula):
                for sample, coeff in zip(formula.samples, formula.coeffs):
                    # create the expression that scales this particular sample based on the formula
                    name = f"f_scale_vbf_sample_{sample.label}"
                    expr = pow_to_mul_string(coeff)
                    for i, coupling in enumerate(formula.couplings):
                        expr = replace_coupling(coupling, f"@{i}", expr)
                    self.make_expr(f"expr::{name}('{expr}', {', '.join(formula.couplings)})")

                    # scale it by the channel specific r POI
                    new_name = f"{name}__{formula.r_poi}"
                    self.make_expr(f"prod::{new_name}({formula.r_poi}, {name})")
                    name = new_name

                    # scale it by the common r POI
                    new_name = f"{name}__r"
                    self.make_expr(f"prod::{new_name}(r, {name})")
                    name = new_name

                    # store the final expression name
                    self.r_expressions[(formula, sample)] = name

            elif isinstance(formula, VHHFormula):
                for sample, coeff in zip(formula.samples, formula.coeffs):
                    # create the expression that scales this particular sample based on the formula
                    name = f"f_scale_vhh_sample_{sample.label}"
                    expr = pow_to_mul_string(coeff)
                    for i, coupling in enumerate(formula.couplings):
                        expr = replace_coupling(coupling, f"@{i}", expr)
                    self.make_expr(f"expr::{name}('{expr}', {', '.join(formula.couplings)})")

                    # scale it by the channel specific r POI
                    new_name = f"{name}__{formula.r_poi}"
                    self.make_expr(f"prod::{new_name}({formula.r_poi}, {name})")
                    name = new_name

                    # scale it by the common r POI
                    new_name = f"{name}__r"
                    self.make_expr(f"prod::{new_name}(r, {name})")
                    name = new_name

                    # store the final expression name
                    self.r_expressions[(formula, sample)] = name

            else:
                raise Exception(f"unhandled formula {formula}")

    def create_ggf_kl_dep_unc(self, ecms, scale=1.0):
        """
        Creates the expressions used to scale ggf process rates depending on kl, including the
        QCDscale + mtop uncertainty for multiple center-of-mass energies at once (since they depend
        on the same, single parameter but induce different effects based on the ecm).

        *scale* should be a float that can increase or reduce the effect to the desired fraction,
        e.g. for projection studies.
        """
        # add the parameter
        p = self.ggf_kl_dep_unc
        self.make_var(f"{p}[-7,7]")

        for ecm in ecms:
            # sanity check
            assert isinstance(ecm, str) and ecm

            # shorter names for easier formatting below
            e = f"{self.ggf_kl_dep_unc}_{ecm}"

            # kappa names
            kappa = f"{e}_kappa"
            hi = f"{e}_kappaHi"
            lo = f"{e}_kappaLo"

            # add uncertainty bands
            expr_nom = self._create_ggf_xsec_str(ecm, "nnlo", "@0")
            expr_hi = self._create_ggf_xsec_str(ecm, "unc_u", "@0")
            expr_lo = self._create_ggf_xsec_str(ecm, "unc_d", "@0")
            self.make_expr(f"expr::{hi}('1.0 + ({scale}) * ((({expr_hi}) / ({expr_nom})) - 1.0)', kl)")  # noqa
            self.make_expr(f"expr::{lo}('1.0 + ({scale}) * ((({expr_lo}) / ({expr_nom})) - 1.0)', kl)")  # noqa

            # create the interpolation as in
            # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/102x/interface/ProcessNormalization.h
            logKhi = f"log({hi})"
            logKlo = f"-log({lo})"
            avg = f"0.5 * ({logKhi} + {logKlo})"
            halfdiff = f"0.5 * ({logKhi} - {logKlo})"
            twop = f"2 * {p}"
            twop2 = f"({twop}) * ({twop})"
            alpha = f"0.125 * {twop} * ({twop2} * ({twop2} - 10.) + 15.)"
            # scaling expression in the center (-0.5, +0.5) and outside
            retCent = f"{avg} + {alpha} * {halfdiff}"
            retLow = logKlo
            retHigh = logKhi
            # combined expression
            retFull = f"{p} <= -0.5 ? ({retLow}) : {p} >= 0.5 ? ({retHigh}) : ({retCent})"

            # add the scaling
            self.make_expr(f"expr::{kappa}('exp({retFull})', {{{hi}, {lo}, {p}}})")  # TODO: outer braces correct?
            self.make_expr(f"expr::scaling_{e}('pow(@0, @1)', {kappa}, {p})")

    def preProcessNuisances(self, nuisances):
        """
        Hook called by the super class before nuisances are processed.

        Here, we make sure that all custom parameters are properly added to the model:

        - kl-dependent ggf theory uncertainty
        - parameters of coupling modifiers when profiling
        """
        # enable profiling of r and k POIs with a configurable prior when requested
        for p in list(self.R_POIS.keys()) + list(self.K_POIS.keys()):
            value = self.opt("doProfile" + p.replace("_", ""), False)
            if not value:
                continue

            # get the prior and add it
            prior, width = value.split(",", 1) if "," in value else (value, None)
            if prior == "flat":
                self.model_builder.DC.flatParamNuisances[p] = True
                print(f"adding flat prior for parameter {p}")
            elif prior == "gauss":
                nuisances.append((p, False, "param", ["1", width, "[-7,7]"], []))
                print(f"adding gaussian prior for parameter {p} with width {width}")
            else:
                raise Exception(f"unknown prior '{prior}' for parameter {p}")

        # add the theory uncertainty on ggf whwn configured
        if self.opt("doklDependentUnc"):
            for ecm in self.get_effective_ecms():
                nuisances.append((f"{self.ggf_kl_dep_unc}_{ecm}", False, "param", ["0", "1"], []))

    def getYieldScale(self, bin, process):
        """
        Hook called by the super class to determine the scaling, or an expression modeling the
        scaling of a *process* in a specific datacard *bin* (however, we decided not to encode
        any important information into the bin name).

        Here, we distinguish several cases, depending on which type of process is considered:

        - HH signal:
            - use the scaling expression build in :py:meth:`create_scalings`
            - optionally add kappa dependence on branching ratios

        - single H backgrounds:
            - optionally add kappa dependence on production cross sections
            - optionally add kappa dependence on branching ratios

        - other backgrounds:
            - return 1 to express that we are using the rate as saved in datacards
        """
        # find signal matches
        for formula_key, formulae in self.get_formulae().items():
            for ecm, formula in formulae.items():
                # get matching samples
                matching_samples = []
                for sample in formula.samples:
                    if sample.matches_process(process):
                        matching_samples.append(sample)

                # nothing to do when there is no hit
                if not matching_samples:
                    continue

                # complain when there is more than one hit
                if len(matching_samples) > 1:
                    raise Exception(
                        f"found {len(matching_samples)} matches for {formula_key} and ecm {ecm} "
                        f"checking for signal process {process} in bin {bin}",
                    )

                # get the scale
                sample = matching_samples[0]
                # store the process
                self.hh_process_scales[formula][sample].add(process)
                # get the scale expression
                scaling = self.r_expressions[(formula, sample)]
                # when the BR scaling is enabled, try to extract the decays from the process name
                if self.opt("doBRscaling"):
                    scaling = self.h_br_scaler.build_xsbr_scaling_hh(scaling, process, bin) or scaling  # noqa
                return scaling

        # complain when the process is a signal but no sample matched
        if self.DC.isSignal[process]:
            raise Exception(f"signal process {process} did not match any HH sample in bin {bin}")

        # single H match?
        if self.opt("doHscaling"):
            scaling = self.h_br_scaler.find_h_scaling(process, bin)[-1]
            # when the BR scaling is enabled, try to extract the decay from the process name
            if scaling and self.opt("doBRscaling"):
                scaling = self.h_br_scaler.build_xsbr_scaling_h(scaling, process, bin) or scaling
            return scaling or 1.0

        # at this point we are dealing with a background process that is also not single-H-scaled,
        # so it is safe to return 1 since any misconfiguration should have been raised already
        return 1.0


def create_model(name, ggf=None, vbf=None, vhh=None, **kwargs):
    """
    Returns a new :py:class:`HHModel` instance named *name*. Its *ggf*, *vbf* and *vhh* samples can
    configured through lists that should either contain valid sample instances or keys of samples
    listed in the global *ggf_samples*, *vbf_samples* and *vhh_samples* dictionaries. All
    additional *kwargs* are forwarded to the model constructor.
    """
    # helper to get samples
    def get_samples(selected_samples, all_samples, sample_cls):
        if not selected_samples:
            return None
        samples = []
        for s in selected_samples:
            if isinstance(s, sample_cls):
                samples.append(s)
            elif s in all_samples:
                samples.append(all_samples[s])
            else:
                raise Exception(
                    f"sample '{s}' is neither an instance of {sample_cls}, nor does it correspond "
                    "to a known sample",
                )
        return samples

    # create the return the model
    return HHModel(
        name=name,
        ggf_samples=get_samples(ggf, ggf_samples, GGFSample),
        vbf_samples=get_samples(vbf, vbf_samples, VBFSample),
        vhh_samples=get_samples(vhh, vhh_samples, VHHSample),
        **kwargs,
    )


# some named, default models
# model_all = create_model(
#     "model_all",
#     ggf=[(0, 1), (1, 1), (2.45, 1), (5, 1)],
#     vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (0.5, 1, 1), (1.5, 1, 1)],
# )
# model_all_vhh = create_model(
#     "model_all_vhh",
#     ggf=model_all.ggf_formula.samples,
#     vbf=model_all.vbf_formula.samples,
#     vhh=[(1, 1, 1), (1, 1, 2), (1, 0, 1), (1, 2, 1), (0.5, 1, 1), (1.5, 1, 1), (1, 1, 0), (1, 1, 20)],  # noqa
# )
# model_vbf_reweight = create_model(
#     "model_vbf_reweight",
#     vbf=[
#         (1.0, 1.0, 1.0),
#         (0.4, 1.0, 2.9),
#         (1.1, 1.0, 0.2),
#         (1.0, 0.2, 1.3),
#         (1.0, 1.3, 0.2),
#         (1.1, 0.2, 0.9),
#         (0.4, 1.3, 2.2),
#         (1.74, 1.37, 14.4),
#         (-0.962, 0.959, -1.43),
#         (-0.758, 1.44, -19.3),
#         (-1.6, 2.72, -1.36),
#         (-0.012, 0.03, 10.2),
#         (-1.21, 1.94, -0.94),
#         (2.12, 3.87, -5.96),
#         (-1.83, 3.57, -3.39),
#         (-0.65, -0.382, 19.9),
#         (0.008, -0.047, 19.9),
#         (0.906, 0.878, 1.55),
#         (1.27, 1.89, 1.17),
#         (1.0, 0.0, 1.0),
#     ],
# )

#
# predefined models
#

default_ggf_points = [(1, 1), (2.45, 1), (5, 1)]  # no (1, 0)
default_vbf_points_run2l = [(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)]  # no (0.5, 1, 1)
default_vbf_points = [(1, 1, 1), (1, 1, 2), (1, 0, 1), (1, 2, 1), (-1.21, 1.94, -0.94), (-0.012, 0.03, 10.2)]  # draft!
default_vhh_points = [(1, 1, 1), (1, 1, 2), (1, 0, 1), (1, 2, 1), (0.5, 1, 1), (1.5, 1, 1), (1, 1, 0), (1, 1, 20)]


def make_points(defaults, ecm):
    return [p + (ecm,) for p in defaults]


# model used for the combination
model_default_run2l = create_model(
    "model_default_run2l",
    ggf=make_points(default_ggf_points, None),
    vbf=make_points(default_vbf_points_run2l, None),
    vhh=make_points(default_vhh_points, None),
)
model_default_run2 = create_model(
    "model_default_run2",
    ggf=make_points(default_ggf_points, "13p0TeV"),
    vbf=make_points(default_vbf_points, "13p0TeV"),
    vhh=make_points(default_vhh_points, "13p0TeV"),
)
model_default_run3 = create_model(
    "model_default_run3",
    ggf=make_points(default_ggf_points, "13p6TeV"),
    vbf=make_points(default_vbf_points, "13p6TeV"),
    # TODO: no vhh points for 13p6TeV yet
    # vhh=make_points(default_vhh_points, "13p6TeV"),
)
model_default_run2l3 = create_model(
    "model_default_run2l3",
    ggf=make_points(default_ggf_points, None) + make_points(default_ggf_points, "13p6TeV"),
    vbf=make_points(default_vbf_points_run2l, None) + make_points(default_vbf_points, "13p6TeV"),
    # TODO: no vhh points for 13p6TeV yet
    # vhh=make_points(default_vhh_points, None) + make_points(default_vhh_points, "13p6TeV"),
)
model_default_run23 = create_model(
    "model_default_run23",
    ggf=make_points(default_ggf_points, "13p0TeV") + make_points(default_ggf_points, "13p6TeV"),
    vbf=make_points(default_vbf_points, "13p0TeV") + make_points(default_vbf_points, "13p6TeV"),
    # TODO: no vhh points for 13p6TeV yet
    # vhh=make_points(default_vhh_points, "13p0TeV") + make_points(default_vhh_points, "13p6TeV"),
)


####################################################################################################
# cross section helpers
####################################################################################################

def create_ggf_xsec_str(ecm, coeffs, s):
    """
    Returns a string expression of one set of coefficients stored in :py:attr:`ggf_kl_coeffs` as
    *coeffs* with kl being replaced by *s*. Example:

    .. code-block:: python

        create_ggf_xsec_str("13p0TeV", "nnlo", "kl")
        # -> "68.5624 - 48.3673 * kl + 10.5635 * kl * kl"
    """
    a0, a1, a2 = ggf_kl_coeffs[coeffs][ecm]
    sign1, abs1 = ["+", "-"][a1 < 0], abs(a1)
    sign2, abs2 = ["+", "-"][a2 < 0], abs(a2)
    return f"{a0} {sign1} {abs1} * {s} {sign2} {abs2} * {s} * {s}"


def create_ggf_xsec_func(ggf_formula):
    """
    Creates and returns a function that can be used to calculate numeric ggf cross section values in
    pb given an appropriate :py:class:`GGFFormula` instance *ggf_formula*. The returned function has
    the signature ``(kl=1.0, kt=1.0, unc=None)``.

    The returned value is in full next-to-next-to-leading order. In this case, *unc* can be set to
    eiher "up" or "down" to return the up / down varied cross section instead where the uncertainty
    is composed of a *kl* dependent QCDscale + mtop uncertainty and a flat PDF uncertainty of 3%.

    Example:

    .. code-block:: python

        get_ggf_xsec = create_ggf_xsec_func()

        print(get_ggf_xsec(kl=2.))
        # -> 0.013803...

        print(get_ggf_xsec(kl=2., unc="up"))
        # -> 0.014305...

    Formulae are taken from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?rev=70.
    """
    # effective center-of-mass energy
    ecm = ggf_formula.effective_ecm

    # create the lambdify'ed evaluation function
    symbol_names = ["kl", "kt"] + list(map("xs{}".format, range(ggf_formula.n_samples)))
    xsec_func = sympy.lambdify(sympy.symbols(symbol_names), ggf_formula.sigma)

    # scale+mtop uncertainty in case unc is set
    expr_u = create_ggf_xsec_str(ecm, "unc_u", "kl")
    expr_d = create_ggf_xsec_str(ecm, "unc_d", "kl")
    xsec_nnlo_scale_up = eval(f"lambda kl: 0.001 * ({expr_u})")
    xsec_nnlo_scale_down = eval(f"lambda kl: 0.001 * ({expr_d})")

    # flat pdf uncertainty
    pdf_unc = 0.03  # ecm independent

    def apply_uncertainty_nnlo(kl, xsec_nom, unc):
        # note on kt: in the twiki linked above, uncertainties on the ggf production cross section
        # are quoted for different kl values but otherwise fully SM parameters, esp. kt=1;
        # however, the nominal cross section *xsec_nom* might be subject to a different kt value
        # and thus, the following implementation assumes that the relative uncertainties according
        # to the SM recommendation are preserved; for instance, if the the scale+mtop uncertainty
        # for kl=2,kt=1 would be 10%, then the code below will assume an uncertainty for kl=2,kt!=1
        # of 10% as well

        # compute the relative, signed scale+mtop uncertainty
        if unc.lower() not in ("up", "down"):
            raise ValueError(f"unc must be 'up' or 'down', got '{unc}'")
        scale_func = {"up": xsec_nnlo_scale_up, "down": xsec_nnlo_scale_down}[unc.lower()]  # noqa
        xsec_nom_sm = xsec_func(kl, 1.0, *(sample.xs for sample in ggf_formula.samples))[0, 0]
        xsec_unc = (scale_func(kl) - xsec_nom_sm) / xsec_nom_sm

        # combine with flat 3% PDF uncertainty, preserving the sign
        unc_sign = 1 if xsec_unc > 0 else -1
        xsec_unc = unc_sign * (xsec_unc**2 + pdf_unc**2)**0.5

        # compute the shifted absolute value
        xsec = xsec_nom * (1.0 + xsec_unc)

        return xsec

    # wrap into another function to apply defaults
    def wrapper(kl=1.0, kt=1.0, unc=None):
        xsec = xsec_func(kl, kt, *(sample.xs for sample in ggf_formula.samples))[0, 0]

        # apply uncertainties?
        if unc:
            xsec = apply_uncertainty_nnlo(kl, xsec, unc)

        return xsec

    # store names of kwargs in the signature for easier access to features
    wrapper.xsec_kwargs = {"kl", "kt", "unc"}

    # store a function that evaluates whether the wrapper has uncertainties based on other settings
    wrapper.has_unc = lambda **kwargs: True

    return wrapper


def create_vbf_xsec_func(vbf_formula):
    """
    Creates and returns a function that can be used to calculate numeric vbf cross section values in
    pb given an appropriate :py:class:`VBFFormula` instance *vbf_formula*. The returned function has
    the signature ``(C2V=1.0, CV=1.0, kl=1.0, unc=None)``.

    *unc* can be set to eiher "up" or "down" to return the up / down varied cross section instead
    where the uncertainty is composed of scale variations and pdf+alpha_s.

    Example:

    .. code-block:: python

        get_vbf_xsec = create_vbf_xsec_func()

        print(get_vbf_xsec(C2V=2.0))
        # -> 0.014218... (or similar)

    Uncertainties taken from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGHH?rev=70.
    """
    # effective center-of-mass energy
    ecm = vbf_formula.effective_ecm

    # create the lambdify'ed evaluation function
    symbol_names = ["C2V", "CV", "kl"] + list(map("xs{}".format, range(vbf_formula.n_samples)))
    xsec_func = sympy.lambdify(sympy.symbols(symbol_names), vbf_formula.sigma)

    # flat pdf and scale uncertainties
    pdf_unc = 0.027  # ecm independent
    scale_unc = {
        "13p0TeV": {"up": 0.0005, "down": 0.0004},
        "13p6TeV": {"up": 0.0005, "down": 0.0003},
    }[ecm]

    # wrap into another function to apply defaults
    def wrapper(C2V=1.0, CV=1.0, kl=1.0, unc=None):
        xsec = xsec_func(C2V, CV, kl, *(sample.xs for sample in vbf_formula.samples))[0, 0]

        # apply uncertainties?
        if unc:
            if unc.lower() not in ["up", "down"]:
                raise ValueError(f"unc must be 'up' or 'down', got '{unc}'")
            unc_rel = (1 if unc.lower() == "up" else -1) * (scale_unc[unc.lower()]**2 + pdf_unc**2)**0.5
            xsec *= 1 + unc_rel

        return xsec

    # store a function that evaluates whether the wrapper has uncertainties based on other settings
    wrapper.xsec_kwargs = {"C2V", "CV", "kl", "unc"}

    # store a function that evaluates whether the wrapper has uncertainties
    wrapper.has_unc = lambda **kwargs: True

    return wrapper


def create_vhh_xsec_func(vhh_formula):
    """
    Creates and returns a function that can be used to calculate numeric vhh cross section values in
    pb given an appropriate :py:class:`VHHFormula` instance *vhh_formula*. The returned function has
    the signature ``(C2V=1.0, CV=1.0, kl=1.0)``.

    Example:

    .. code-block:: python

        get_vhh_xsec = create_vhh_xsec_func()

        print(get_vhh_xsec(C2V=2.))
        # -> 0.005316... (or similar)
    """
    # create the lambdify'ed evaluation function
    symbol_names = ["C2V", "CV", "kl"] + list(map("xs{}".format, range(vhh_formula.n_samples)))
    xsec_func = sympy.lambdify(sympy.symbols(symbol_names), vhh_formula.sigma)

    # wrap into another function to apply defaults
    def wrapper(C2V=1.0, CV=1.0, kl=1.0):
        xsec = xsec_func(C2V, CV, kl, *(sample.xs for sample in vhh_formula.samples))[0, 0]
        return xsec

    # store names of kwargs in the signature for easier access to features
    wrapper.xsec_kwargs = {"C2V", "CV", "kl"}

    # store a function that evaluates whether the wrapper has uncertainties based on other settings
    wrapper.has_unc = lambda **kwargs: False

    return wrapper


def create_hh_xsec_func(ggf_formula=None, vbf_formula=None, vhh_formula=None):
    """
    Creates and returns a function that can be used to calculate numeric HH cross section values in
    pb given appropriate *ggf_formula*, *vbf_formula* and *vhh_formula* instances. When a forumla
    evaluates to *False* (the default), the corresponding process is not considered in the inclusive
    calculation. The returned function has the signature
    ``(kl=1.0, kt=1.0, C2V=1.0, CV=1.0, unc=None)``.

    *unc* can be set to eiher "up" or "down" to return the up / down varied cross section instead
    where the uncertainty is composed of a *kl* dependent scale+mtop uncertainty and an independent
    PDF uncertainty of 3% for ggf, and a scale and pdf+alpha_s uncertainty for vbf. The
    uncertainties of the ggf and vbf processes are treated as uncorrelated.

    Example:

    .. code-block:: python

        get_hh_xsec = create_hh_xsec_func()

        print(get_hh_xsec(kl=2.0))
        # -> 0.015226...

        print(get_hh_xsec(kl=2.0, unc="up"))
        # -> 0.015702...
    """
    if not any([ggf_formula, vbf_formula, vhh_formula]):
        raise ValueError("at least one of the cross section formulae is required")

    # ecm must match across all formulae
    ecms = {f.ecm for f in [ggf_formula, vbf_formula, vhh_formula] if f}
    if len(ecms) != 1:
        raise ValueError(
            "to build a consistent cross section function, all formulae must have the same ecm, "
            f"but got {ecms}",
        )

    # default function for a disabled process
    no_xsec = lambda *args, **kwargs: 0.0

    # get the particular wrappers of the components
    get_ggf_xsec = create_ggf_xsec_func(ggf_formula) if ggf_formula else no_xsec
    get_vbf_xsec = create_vbf_xsec_func(vbf_formula) if vbf_formula else no_xsec
    get_vhh_xsec = create_vhh_xsec_func(vhh_formula) if vhh_formula else no_xsec

    # create a combined wrapper with the merged signature
    def wrapper(kl=1.0, kt=1.0, C2V=1.0, CV=1.0, unc=None):
        ggf_xsec = get_ggf_xsec(kl=kl, kt=kt)
        vbf_xsec = get_vbf_xsec(C2V=C2V, CV=CV, kl=kl)
        vhh_xsec = get_vhh_xsec(C2V=C2V, CV=CV, kl=kl)
        xsec = ggf_xsec + vbf_xsec + vhh_xsec

        # apply uncertainties?
        if unc:
            if unc.lower() not in ["up", "down"]:
                raise ValueError(f"unc must be 'up' or 'down', got '{unc}'")
            # ggf uncertainty
            ggf_unc = get_ggf_xsec(kl=kl, kt=kt, unc=unc) - ggf_xsec
            # vbf uncertainty
            vbf_unc = get_vbf_xsec(C2V=C2V, CV=CV, kl=kl, unc=unc) - vbf_xsec
            # vhh uncertainty
            vhh_unc = 0.0
            # combine
            sign = 1 if unc.lower() == "up" else -1
            unc = sign * (ggf_unc**2 + vbf_unc**2 + vhh_unc**2)**0.5
            xsec += unc

        return xsec

    # store names of kwargs in the signature for easier access to features
    getters = [get_ggf_xsec, get_vbf_xsec, get_vhh_xsec]
    wrapper.xsec_kwargs = set.union(*(g.xsec_kwargs for g in getters if g != no_xsec))

    # store a function that evaluates whether the wrapper has uncertainties based on other settings
    wrapper.has_unc = lambda **kwargs: any((g != no_xsec and g.has_unc(**kwargs)) for g in getters)

    return wrapper
