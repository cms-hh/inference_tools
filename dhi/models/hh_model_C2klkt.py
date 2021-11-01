# coding: utf-8

"""
Custom HH physics model implementing the gluon gluon fusion (ggf / gghh) mode (depending on C2, kl
and kt), and the vector boson fusion (vbf / qqhh) mode.

Authors:
  - Torben Lange
  - Marcel Rieger
"""


from collections import OrderedDict

import sympy

# wildcard import to have everything available locally, plus specific imports for linting
from hh_model import *  # noqa
from hh_model import (
    HHSample, HHFormula, VBFFormula, HHModelBase, HHModel as DefaultHHModel, vbf_samples,
    create_ggf_xsec_str, ggf_k_factor, create_vbf_xsec_func,
)


####################################################################################################
### c2 based ggf sample
####################################################################################################

class GGFSample(HHSample):
    """
    Class describing ggf samples, characterized by values of *kl*, *kt* and *C2*.
    """

    label_re = r"^ggHH_kl_([pm0-9]+)_kt_([pm0-9]+)_c2_([pm0-9]+)$"

    def __init__(self, kl, kt, C2, xs, label):
        super(GGFSample, self).__init__(xs, label)

        self.kl = kl
        self.kt = kt
        self.C2 = C2


# ggf samples with keys (kl, kt, C2)
ggf_samples = OrderedDict([
    ((0.0,  1.0, 0.0),  GGFSample(kl=0.0,  kt=1.0, C2=0.0,  xs=0.069725, label="ggHH_kl_0p00_kt_1p00_c2_0p00")),
    ((1.0,  1.0, 0.0),  GGFSample(kl=1.0,  kt=1.0, C2=0.0,  xs=0.031047, label="ggHH_kl_1p00_kt_1p00_c2_0p00")),
    ((2.45, 1.0, 0.0),  GGFSample(kl=2.45, kt=1.0, C2=0.0,  xs=0.013124, label="ggHH_kl_2p45_kt_1p00_c2_0p00")),
    ((5.0,  1.0, 0.0),  GGFSample(kl=5.0,  kt=1.0, C2=0.0,  xs=0.091172, label="ggHH_kl_5p00_kt_1p00_c2_0p00")),
    ((0.0,  1.0, 1.0),  GGFSample(kl=0.0,  kt=1.0, C2=1.0,  xs=0.155508, label="ggHH_kl_0p00_kt_1p00_c2_1p00")),
    ((1.0,  1.0, 0.1),  GGFSample(kl=1.0,  kt=1.0, C2=0.1,  xs=0.015720, label="ggHH_kl_1p00_kt_1p00_c2_0p10")),
    ((1.0,  1.0, 0.35), GGFSample(kl=1.0,  kt=1.0, C2=0.35, xs=0.011103, label="ggHH_kl_1p00_kt_1p00_c2_0p35")),
    ((1.0,  1.0, 3.0),  GGFSample(kl=1.0,  kt=1.0, C2=3.0,  xs=2.923567, label="ggHH_kl_1p00_kt_1p00_c2_3p00")),
    ((1.0,  1.0, -2.0), GGFSample(kl=1.0,  kt=1.0, C2=-2.0, xs=1.956196, label="ggHH_kl_1p00_kt_1p00_c2_m2p00")),
])


####################################################################################################
### c2 based ggf formula
####################################################################################################

class GGFFormula(HHFormula):
    """
    Scaling formula for ggf samples, based on a n_samples x 3 matrix.
    """

    sample_cls = GGFSample
    min_samples = 6
    channel = "ggf"
    r_poi = "r_gghh"
    couplings = ["kl", "kt", "C2"]

    def build_expressions(self):
        # define the matrix with three scalings - box, triangle, interf
        self.M = sympy.Matrix([
            [
                sample.kt**2 * sample.kl**2,
                sample.kt**4,
                sample.C2**2,
                sample.C2 * sample.kl * sample.kt,
                sample.C2 * sample.kt**2,
                sample.kl * sample.kt**3,

            ]
            for i, sample in enumerate(self.samples)
        ])

        # the vector of couplings
        kl, kt, C2, box, tria, cross, itc, ibc, itb = sympy.symbols("kl kt C2 box tria cross itc ibc itb")
        c = sympy.Matrix([
            [kt**2 * kl**2],
            [kt**4],
            [C2**2],
            [C2 * kl * kt],
            [C2 * kt**2],
            [kl * kt**3],
        ])

        # the vector of symbolic sample cross sections
        s = sympy.Matrix([
            [sympy.Symbol("xs{}".format(i))]
            for i in range(self.n_samples)
        ])

        # actual computation, i.e., matrix inversion and multiplications with vectors
        M_inv = self.M.pinv()
        self.coeffs = c.transpose() * M_inv
        self.sigma = self.coeffs * s


####################################################################################################
### c2 based model
####################################################################################################

class HHModel(DefaultHHModel):
    """
    Models the HH production as linear sum of the input components for >= 6 ggf (EFT) and >= 6 vbf
    samples. The following physics options are supported:

    - doNNLOscaling (bool)   : Convert ggF HH yields (that are given in NLO by convention) to NNLO.
    - doBRscaling (bool)     : Enable scaling Higgs branching ratios with model parameters.
    - doHscaling (bool)      : Enable scaling single Higgs cross sections with model parameters.
    - doklDependentUnc (bool): Add a theory uncertainty on ggF HH production that depends on model
                               parameters.
    - doProfileX (string)    : Either "flat" to enable the profiling of parameter X with a flat
      X in {rgghh,rqqhh,rvhh,  prior, or "gauss,FLOAT" (or "gauss,-FLOAT/+FLOAT") to use a gaussian
      kl,kt,CV,C2V,C2}         (asymmetric) prior. In any case, X will be profiled and is hence
                               removed from the list of POIs.

    A string encoded boolean flag is interpreted as *True* when it is either ``"yes"``, ``"true"``
    or ``1`` (case-insensitive).
    """

    # pois with initial (SM) value, start and stop
    R_POIS = OrderedDict([
        ("r", (1, -20, 20)),
        ("r_gghh", (1, -20, 20)),
        ("r_qqhh", (1, -20, 20)),
    ])
    K_POIS = OrderedDict([
        ("kl", (1, -30, 30)),
        ("kt", (1, -10, 10)),
        ("CV", (1, -10, 10)),
        ("C2V", (1, -10, 10)),
        ("C2", (0, -10, 10)),
    ])

    def __init__(self, name, ggf_samples=None, vbf_samples=None):
        # skip the DefaultHHModel init
        HHModelBase.__init__(self, name)

        # attributes
        self.ggf_formula = GGFFormula(ggf_samples) if ggf_samples else None
        self.vbf_formula = VBFFormula(vbf_samples) if vbf_samples else None
        self.vhh_formula = None
        self.ggf_kl_dep_unc = "THU_HH"  # name for kl-dependent QCDscale + mtop uncertainty on ggf
        self.h_br_scaler = None  # initialized in create_scalings

        # register options
        self.register_opt("doNNLOscaling", True, is_flag=True)
        self.register_opt("doklDependentUnc", True, is_flag=True)
        self.register_opt("doBRscaling", True, is_flag=True)
        self.register_opt("doHscaling", True, is_flag=True)
        for p in self.R_POIS.keys() + self.K_POIS.keys():
            if p != "r":
                self.register_opt("doProfile" + p.replace("_", ""), None)

        # reset instance-level pois
        self.reset_pois()

    def _create_hh_xsec_func(self, *args, **kwargs):
        # forward to the modul-level implementation
        return create_hh_xsec_func(*args, **kwargs)


def create_model(name, ggf_keys=None, vbf_keys=None, **kwargs):
    """
    Returns a new :py:class:`HHModel` instance named *name*. Its ggf sample list can be configured
    by passing a list of *ggf_keys* which defaults to all availabe samples. The order of
    passed keys to be skipped does not matter. All additional *kwargs* are forwarded to the model
    constructor.
    """
    # expand ggf keys
    if not ggf_keys:
        ggf_keys = ggf_samples.keys()

    # expand vbf keys
    if not vbf_keys:
        vbf_keys = vbf_samples.keys()

    # create the return the model
    return HHModel(
        name=name,
        ggf_samples=[ggf_samples[key] for key in ggf_keys],
        vbf_samples=[vbf_samples[key] for key in vbf_keys],
        **kwargs
    )


# default model
model_default = create_model("model_default",
    ggf_keys=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf_keys=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

# default model without vbf
model_default_novbf = create_model("model_default_novbf",
    ggf_keys=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf_keys=[],
)


####################################################################################################
### Updated cross section helpers
####################################################################################################

def create_ggf_xsec_func(ggf_formula):
    """
    Creates and returns a function that can be used to calculate numeric ggf cross section values in
    pb given an appropriate :py:class:`GGFFormula` instance *formula*. The returned function has the
    signature ``(kl=1.0, kt=1.0, C2=0.0, nnlo=True, unc=None)``.

    When *nnlo* is *False*, the constant k-factor is still applied. Otherwise, the returned value is
    in full next-to-next-to-leading order. In this case, *unc* can be set to eiher "up" or "down" to
    return the up / down varied cross section instead where the uncertainty is composed of a *kl*
    dependent QCDscale + mtop uncertainty and a flat PDF uncertainty of 3%.

    Example:

    .. code-block:: python

        get_ggf_xsec = create_ggf_xsec_func()

        print(get_ggf_xsec(kl=2.))
        # -> 0.013803...

        print(get_ggf_xsec(kl=2., nnlo=False))
        # -> 0.013852...

        print(get_ggf_xsec(kl=2., unc="up"))
        # -> 0.014305...

    Formulae are taken from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?rev=70.
    """
    # create the lambdify'ed evaluation function
    symbol_names = ["kl", "kt", "C2"] + list(map("xs{}".format, range(ggf_formula.n_samples)))
    xsec_func = sympy.lambdify(sympy.symbols(symbol_names), ggf_formula.sigma)

    # nlo-to-nnlo scaling functions in case nnlo is set
    # (not really clean to use eval here, but this way create_ggf_xsec_str remains the only
    # place where the formula needs to be mainted)
    expr_nlo = create_ggf_xsec_str("nlo", "kl")
    expr_nnlo = create_ggf_xsec_str("nnlo", "kl")
    xsec_nlo = eval("lambda kl: 0.001 * {} * ({})".format(ggf_k_factor, expr_nlo))
    xsec_nnlo = eval("lambda kl: 0.001 * ({})".format(expr_nnlo))
    nlo2nnlo = lambda xsec, kl: xsec * xsec_nnlo(kl) / xsec_nlo(kl)  # noqa

    # scale+mtop uncertainty in case unc is set
    expr_u1 = create_ggf_xsec_str("unc_u1", "kl")
    expr_u2 = create_ggf_xsec_str("unc_u2", "kl")
    expr_d1 = create_ggf_xsec_str("unc_d1", "kl")
    expr_d2 = create_ggf_xsec_str("unc_d2", "kl")
    xsec_nnlo_scale_up = eval("lambda kl: 0.001 * max({}, {})".format(expr_u1, expr_u2))
    xsec_nnlo_scale_down = eval("lambda kl: 0.001 * min({}, {})".format(expr_d1, expr_d2))

    def apply_uncertainty_nnlo(kl, xsec_nom, unc):
        # note on kt: in the twiki linked above, uncertainties on the ggf production cross section
        # are quoted for different kl values but otherwise fully SM parameters, esp. kt=1 and C2=0;
        # however, the nominal cross section *xsec_nom* might be subject to a different kt value
        # and thus, the following implementation assumes that the relative uncertainties according
        # to the SM recommendation are preserved; for instance, if the the scale+mtop uncertainty
        # for kl=2,kt=1 would be 10%, then the code below will assume an uncertainty for kl=2,kt!=1
        # of 10% as well

        # compute the relative, signed scale+mtop uncertainty
        if unc.lower() not in ("up", "down"):
            raise ValueError("unc must be 'up' or 'down', got '{}'".format(unc))
        scale_func = {"up": xsec_nnlo_scale_up, "down": xsec_nnlo_scale_down}[unc.lower()]  # noqa
        xsec_nom_sm = xsec_func(kl, 1., 0., *(sample.xs for sample in ggf_formula.samples))[0, 0]
        xsec_unc = (scale_func(kl) - xsec_nom_sm) / xsec_nom_sm

        # combine with flat 3% PDF uncertainty, preserving the sign
        unc_sign = 1 if xsec_unc > 0 else -1
        xsec_unc = unc_sign * (xsec_unc**2 + 0.03**2)**0.5

        # compute the shifted absolute value
        xsec = xsec_nom * (1. + xsec_unc)

        return xsec

    # wrap into another function to apply defaults and nlo-to-nnlo scaling
    def wrapper(kl=1.0, kt=1.0, C2=0.0, nnlo=True, unc=None):
        xsec = xsec_func(kl, kt, C2, *(sample.xs for sample in ggf_formula.samples))[0, 0]

        # nnlo scaling?
        if nnlo:
            xsec = nlo2nnlo(xsec, kl)

        # apply uncertainties?
        if unc:
            if not nnlo:
                raise NotImplementedError("NLO ggf cross section uncertainties are not implemented")
            xsec = apply_uncertainty_nnlo(kl, xsec, unc)

        return xsec

    return wrapper


def create_hh_xsec_func(ggf_formula=None, vbf_formula=None):
    """
    Creates and returns a function that can be used to calculate numeric HH cross section values in
    pb given the appropriate *ggf_formula* and *vbf_formula* objects. When a forumla evaluates to
    *False* (the default), the corresponding process is not considered in the inclusive calculation.
    The returned function has the signature
    ``(kl=1.0, kt=1.0, C2=0.0, CV=1.0, C2V=1.0, nnlo=True, unc=None)``.

    The *nnlo* setting only affects the ggF component of the cross section. When *nnlo* is *False*,
    the constant k-factor of the ggf calculation is still applied. Otherwise, the returned value is
    in full next-to-next-to-leading order for ggF. *unc* can be set to eiher "up" or "down" to
    return the up / down varied cross section instead where the uncertainty is composed of a *kl*
    dependent scale + mtop uncertainty and an independent PDF uncertainty of 3% for ggF, and a scale
    and pdf+alpha_s uncertainty for vbf. The uncertainties of the ggf and vbf processes are treated
    as uncorrelated.

    Example:

    .. code-block:: python

        get_hh_xsec = create_hh_xsec_func()

        print(get_hh_xsec(kl=2.))
        # -> 0.015226...

        print(get_hh_xsec(kl=2., nnlo=False))
        # -> 0.015275...

        print(get_hh_xsec(kl=2., unc="up"))
        # -> 0.015702...
    """
    if not any([ggf_formula, vbf_formula]):
        raise ValueError("at least one of the cross section formulae is required")

    # default function for a disabled process
    no_xsec = lambda *args, **kwargs: 0.

    # get the particular wrappers of the components
    get_ggf_xsec = create_ggf_xsec_func(ggf_formula) if ggf_formula else no_xsec
    get_vbf_xsec = create_vbf_xsec_func(vbf_formula) if vbf_formula else no_xsec

    # create a combined wrapper with the merged signature
    def wrapper(kl=1.0, kt=1.0, C2=0.0, CV=1.0, C2V=1.0, nnlo=True, unc=None):
        ggf_xsec = get_ggf_xsec(kl=kl, kt=kt, C2=C2, nnlo=nnlo)
        vbf_xsec = get_vbf_xsec(C2V=C2V, CV=CV, kl=kl)
        xsec = ggf_xsec + vbf_xsec

        # apply uncertainties?
        if unc:
            if unc.lower() not in ("up", "down"):
                raise ValueError("unc must be 'up' or 'down', got '{}'".format(unc))

            # ggf uncertainty
            ggf_unc = get_ggf_xsec(kl=kl, kt=kt, C2=C2, nnlo=nnlo, unc=unc) - ggf_xsec
            # vbf uncertainty
            vbf_unc = get_vbf_xsec(C2V=C2V, CV=CV, kl=kl, unc=unc) - vbf_xsec
            # combine
            sign = 1 if unc.lower() == "up" else -1
            unc = sign * (ggf_unc**2 + vbf_unc**2)**0.5
            xsec += unc

        return xsec

    return wrapper


# default ggF cross section getter using the formula of the *model_default* model
get_ggf_xsec = create_ggf_xsec_func(model_default.ggf_formula)

# default vbf cross section getter using the formula of the *model_default* model
get_vbf_xsec = create_vbf_xsec_func(model_default.vbf_formula)

# default combined cross section getter using the formulas of the *model_default* model
get_hh_xsec = create_hh_xsec_func(model_default.ggf_formula, model_default.vbf_formula)
