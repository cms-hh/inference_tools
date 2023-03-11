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

# we need a wildcard import to have everything accessible through this module
from hh_model import *  # noqa
# specific imports for linting
from hh_model import (
    GGFSample as DefaultGGFSample, GGFFormula as DefaultGGFFormula, VBFSample,
    HHModel as DefaultHHModel, vbf_samples, create_ggf_xsec_str, ggf_k_factor,
    _create_add_sample_func, create_vbf_xsec_func,
)


####################################################################################################
# c2 based ggf sample
####################################################################################################

class GGFSample(DefaultGGFSample):
    """
    Class describing ggf samples, characterized by values of *kl*, *kt* and *C2*.
    """

    # label format
    label_re = r"^ggHH_kl_(m?\d+p\d{2})_kt_(m?\d+p\d{2})_c2_(m?\d+p\d{2})$"

    def __init__(self, kl, kt, C2, xs, label):
        super(GGFSample, self).__init__(kl, kt, xs, label)

        self.C2 = C2

    @property
    def key(self):
        return (self.kl, self.kt, self.C2)


# ggf samples with keys (kl, kt, C2)
ggf_samples = OrderedDict()
add_ggf_sample = _create_add_sample_func(GGFSample, ggf_samples)
add_ggf_sample(kl=0.0, kt=1.0, C2=0.0, xs=0.069725, label="ggHH_kl_0p00_kt_1p00_c2_0p00")
add_ggf_sample(kl=1.0, kt=1.0, C2=0.0, xs=0.031047, label="ggHH_kl_1p00_kt_1p00_c2_0p00")
add_ggf_sample(kl=2.45, kt=1.0, C2=0.0, xs=0.013124, label="ggHH_kl_2p45_kt_1p00_c2_0p00")
add_ggf_sample(kl=5.0, kt=1.0, C2=0.0, xs=0.091172, label="ggHH_kl_5p00_kt_1p00_c2_0p00")
add_ggf_sample(kl=0.0, kt=1.0, C2=1.0, xs=0.155508, label="ggHH_kl_0p00_kt_1p00_c2_1p00")
add_ggf_sample(kl=1.0, kt=1.0, C2=0.1, xs=0.015720, label="ggHH_kl_1p00_kt_1p00_c2_0p10")
add_ggf_sample(kl=1.0, kt=1.0, C2=0.35, xs=0.011103, label="ggHH_kl_1p00_kt_1p00_c2_0p35")
add_ggf_sample(kl=1.0, kt=1.0, C2=3.0, xs=2.923567, label="ggHH_kl_1p00_kt_1p00_c2_3p00")
add_ggf_sample(kl=1.0, kt=1.0, C2=-2.0, xs=1.956196, label="ggHH_kl_1p00_kt_1p00_c2_m2p00")


####################################################################################################
# c2 based ggf formula
####################################################################################################

class GGFFormula(DefaultGGFFormula):
    """
    Scaling formula for ggf samples, based on a n_samples x 3 matrix.
    """

    sample_cls = GGFSample
    min_samples = 6
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
        kl, kt, C2 = sympy.symbols("kl kt C2")
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
# c2 based model
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

    # updated ggf formula class
    ggf_formula_cls = GGFFormula

    def __init__(self, name, ggf_samples=None, vbf_samples=None):
        # invoke super init with just ggf and vbf samples
        super(HHModel, self).__init__(name, ggf_samples=ggf_samples, vbf_samples=vbf_samples)

    def _create_hh_xsec_func(self, *args, **kwargs):
        # forward to the modul-level implementation
        return create_hh_xsec_func(*args, **kwargs)


def create_model(name, ggf=None, vbf=None, **kwargs):
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
                raise Exception("sample '{}' is neither an instance of {}, nor does it correspond "
                    "to a known sample".format(s, sample_cls))
        return samples

    # create the return the model
    return HHModel(
        name=name,
        ggf_samples=get_samples(ggf, ggf_samples, GGFSample),
        vbf_samples=get_samples(vbf, vbf_samples, VBFSample),
        **kwargs  # noqa
    )


# default model
model_default = create_model(
    "model_default",
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

# default model without vbf
model_default_novbf = create_model(
    "model_default_novbf",
    ggf=model_default.ggf_formula.samples,
)


####################################################################################################
# updated cross section helpers
####################################################################################################

def create_ggf_xsec_func(ggf_formula):
    """
    Creates and returns a function that can be used to calculate numeric ggf cross section values in
    pb given an appropriate :py:class:`GGFFormula` instance *formula*. The returned function has the
    signature ``(kl=1.0, kt=1.0, C2=0.0, ggf_nnlo=True, unc=None)``.

    When *ggf_nnlo* is *False*, the constant k-factor is still applied. Otherwise, the returned
    value is in full next-to-next-to-leading order. In this case, *unc* can be set to eiher "up" or
    "down" to return the up / down varied cross section instead where the uncertainty is composed of
    a *kl* dependent QCDscale + mtop uncertainty and a flat PDF uncertainty of 3%.

    Example:

    .. code-block:: python

        get_ggf_xsec = create_ggf_xsec_func()

        print(get_ggf_xsec(kl=2.))
        # -> 0.013803...

        print(get_ggf_xsec(kl=2., ggf_nnlo=False))
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
    def wrapper(kl=1.0, kt=1.0, C2=0.0, ggf_nnlo=True, unc=None):
        xsec = xsec_func(kl, kt, C2, *(sample.xs for sample in ggf_formula.samples))[0, 0]

        # nnlo scaling?
        if ggf_nnlo:
            xsec = nlo2nnlo(xsec, kl)

        # apply uncertainties?
        if unc:
            if not ggf_nnlo:
                raise NotImplementedError("NLO ggf cross section uncertainties are not implemented")
            xsec = apply_uncertainty_nnlo(kl, xsec, unc)

        return xsec

    # store names of kwargs in the signature for easier access to features
    wrapper.xsec_kwargs = {"kl", "kt", "C2", "ggf_nnlo", "unc"}

    # store a function that evaluates whether the wrapper has uncertainties based on other settings
    wrapper.has_unc = lambda ggf_nnlo=True, **kwargs: bool(ggf_nnlo)

    return wrapper


def create_hh_xsec_func(ggf_formula=None, vbf_formula=None):
    """
    Creates and returns a function that can be used to calculate numeric HH cross section values in
    pb given the appropriate *ggf_formula* and *vbf_formula* objects. When a forumla evaluates to
    *False* (the default), the corresponding process is not considered in the inclusive calculation.
    The returned function has the signature
    ``(kl=1.0, kt=1.0, C2=0.0, CV=1.0, C2V=1.0, ggf_nnlo=True, unc=None)``.

    The *ggf_nnlo* setting only affects the ggf component of the cross section. When it is *False*,
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

        print(get_hh_xsec(kl=2., ggf_nnlo=False))
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
    def wrapper(kl=1.0, kt=1.0, C2=0.0, CV=1.0, C2V=1.0, ggf_nnlo=True, unc=None):
        ggf_xsec = get_ggf_xsec(kl=kl, kt=kt, C2=C2, ggf_nnlo=ggf_nnlo)
        vbf_xsec = get_vbf_xsec(C2V=C2V, CV=CV, kl=kl)
        xsec = ggf_xsec + vbf_xsec

        # apply uncertainties?
        if unc:
            if unc.lower() not in ("up", "down"):
                raise ValueError("unc must be 'up' or 'down', got '{}'".format(unc))

            # ggf uncertainty
            ggf_unc = get_ggf_xsec(kl=kl, kt=kt, C2=C2, ggf_nnlo=ggf_nnlo, unc=unc) - ggf_xsec
            # vbf uncertainty
            vbf_unc = get_vbf_xsec(C2V=C2V, CV=CV, kl=kl, unc=unc) - vbf_xsec
            # combine
            sign = 1 if unc.lower() == "up" else -1
            unc = sign * (ggf_unc**2 + vbf_unc**2)**0.5
            xsec += unc

        return xsec

    # store names of kwargs in the signature for easier access to features
    getters = [get_ggf_xsec, get_vbf_xsec]
    wrapper.xsec_kwargs = set.union(*(g.xsec_kwargs for g in getters if g != no_xsec))

    # store a function that evaluates whether the wrapper has uncertainties based on other settings
    wrapper.has_unc = lambda **kwargs: any((g != no_xsec and g.has_unc(**kwargs)) for g in getters)

    return wrapper


# default ggF cross section getter using the formula of the *model_default* model
get_ggf_xsec = create_ggf_xsec_func(model_default.ggf_formula)

# default vbf cross section getter using the formula of the *model_default* model
get_vbf_xsec = create_vbf_xsec_func(model_default.vbf_formula)

# default combined cross section getter using the formulas of the *model_default* model
get_hh_xsec = create_hh_xsec_func(model_default.ggf_formula, model_default.vbf_formula)
