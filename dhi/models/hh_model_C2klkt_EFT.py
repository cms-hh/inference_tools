# coding: utf-8

"""
Custom HH physics model implementing the gluon gluon fusion (ggf / gghh) mode (depending on C2, kl
and kt), and the vector boson fusion (vbf / qqhh) mode for the constraints of
various eft physics scenarios from arXiv:1710.08261

Authors:
  - Torben Lange
"""

from collections import OrderedDict

from dhi.models.hh_model import ( # noqa
    VBFSample, HHModel as DefaultHHModel, vbf_samples, create_ggf_xsec_str,
    ggf_k_factor, _create_add_sample_func, create_vbf_xsec_func, HBRScaler,
) # noqa

# necessary imports
from dhi.models.hh_model_C2klkt import (GGFSample, GGFFormula)

"""
Placeholder imports, if we want it, we need to reimplement the xs functions,
if we only plan likelihoods, not really needed however.
"""

from dhi.models.hh_model_C2klkt import ( # noqa
    create_ggf_xsec_func, create_hh_xsec_func,
    get_ggf_xsec, get_vbf_xsec, get_hh_xsec,
)

# ggf samples with keys (kl, kt, C2), same as in hh_model_C2klkt
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


class HHModelEFTBase(DefaultHHModel):
    """
    Models the HH production as linear sum of the input components for >= 6 ggf (EFT) and >= 6 vbf
    samples. Base class to be customized for various eft constraints.
    The following physics options are supported:

    - doNNLOscaling (bool)   : Convert ggF HH yields (that are given in NLO by convention) to NNLO.
    - doBRscaling (bool)     : Enable scaling Higgs branching ratios with model parameters.
    - doHscaling (bool)      : Enable scaling single Higgs cross sections with model parameters.
    - doklDependentUnc (bool): Add a theory uncertainty on ggF HH production that depends on model
                               parameters.
    - doProfileX (string)    : Either "flat" to enable the profiling of parameter X with a flat
      X in {rgghh,rqqhh,rvhh,  prior, or "gauss,FLOAT" (or "gauss,-FLOAT/+FLOAT") to use a gaussian
      {EFTCOUPLINGS} }         (asymmetric) prior. In any case, X will be profiled and is hence
                               removed from the list of POIs.

    A string encoded boolean flag is interpreted as *True* when it is either ``"yes"``, ``"true"``
    or ``1`` (case-insensitive).
    """

    # updated ggf formula class
    ggf_formula_cls = GGFFormula

    def __init__(self, name, ggf_samples=None, vbf_samples=None):
        # invoke super init with just ggf and vbf samples
        super(HHModelEFTBase, self).__init__(name, ggf_samples=ggf_samples, vbf_samples=vbf_samples)

    def _create_hh_xsec_func(self, *args, **kwargs):
        # forward to the modul-level implementation
        return create_hh_xsec_func(*args, **kwargs)

    # overwrite to not remove new POIs
    def reset_pois(self):

        # r pois
        self.r_pois = OrderedDict()
        for p, v in self.R_POIS.items():
            self.r_pois[p] = v

        # k pois
        self.k_pois = OrderedDict()
        for p, v in self.K_POIS.items():
            self.k_pois[p] = v

        # remove profiled r pois
        for p in list(self.r_pois):
            if self.opt("doProfile" + p.replace("_", ""), False):
                del self.r_pois[p]

        # remove profiled k pois
        for p in list(self.k_pois):
            if self.opt("doProfile" + p.replace("_", ""), False):
                del self.k_pois[p]

    """
    Dummy to customize eft constraints of a specific theory scenario.
    Implemented by specific model.
    """
    def make_eftconstraints(self):
        raise NotImplementedError

    """
    redo this as kl,kt C2 is now not a poi anymore but still needs to be set,
    to be able to customize it, we add a hook "make_eftconstraints" to be
    then implemented in a specific model for a specific theory scenario
    """
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
            self.make_var("{}[{},{},{}]".format(p, value, start, stop))
        for p in self.K_POIS:
            value, start, stop = self.k_pois.get(p, self.K_POIS[p])
            self.make_var("{}[{},{},{}]".format(p, value, start, stop))

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

        # set or redefine the MH variable on which some of the BRs depend
        if not self.options.mass:
            raise Exception(
                "invalid mass value '{}', please provide a valid value using the "
                "--mass option".format(self.options.mass),
            )
        if self.get_var("MH"):
            self.get_var("MH").removeRange()
            self.get_var("MH").setVal(self.options.mass)
        else:
            self.make_var("MH[{}]".format(self.options.mass))
        self.get_var("MH").setConstant(True)

        # hook for the model specific eft constraints
        self.make_eftconstraints()

        # define the POI group
        self.make_set("POI", ",".join(pois))
        print("using POIs {}".format(",".join(pois)))

        # when the HBRScaler is used, make sure that its required pois are existing
        if self.opt("doBRscaling") or self.opt("doHscaling"):
            for p in self.h_br_scaler_cls.REQUIRED_POIS:
                if not self.get_var(p):
                    self.make_var("{}[1]".format(p))
                    self.get_var(p).setConstant(True)

        # add objects for kl dependent theory uncertainties
        if self.opt("doklDependentUnc"):
            self.create_ggf_kl_dep_unc()

        # create cross section scaling functions
        self.create_scalings()


class HBRScaler_Alpha(HBRScaler):
    REQUIRED_POIS = ["A"]


class HBRScaler_AlphaM2(HBRScaler):
    REQUIRED_POIS = ["A", "lA", "M2"]


class HBRScaler_BetaMHEMHP(HBRScaler):
    REQUIRED_POIS = ["B", "MHE", "MHP"]


class HBRScaler_BetaMHEMA(HBRScaler):
    REQUIRED_POIS = ["B", "MHE", "MA"]


class HBRScaler_BetaMHEZ6(HBRScaler):
    REQUIRED_POIS = ["B", "MHE", "Z6"]


class HBRScaler_BetaMHE(HBRScaler):
    REQUIRED_POIS = ["B", "MHE"]


class HBRScaler_VLQ(HBRScaler):
    REQUIRED_POIS = ["LQ", "MQ"]


class HBRScaler_Xi(HBRScaler):
    REQUIRED_POIS = ["XI"]


class HBRScaler_kl(HBRScaler):
    REQUIRED_POIS = ["kl_EFT"]


class HBRScaler_kt(HBRScaler):
    REQUIRED_POIS = ["kt_EFT"]


class HBRScaler_klkt(HBRScaler):
    REQUIRED_POIS = ["kl_EFT", "kt_EFT"]


class HBRScaler_klc2(HBRScaler):
    REQUIRED_POIS = ["kl_EFT", "C2_EFT"]


class HBRScaler_klktc2(HBRScaler):
    REQUIRED_POIS = ["kl_EFT", "kt_EFT", "C2_EFT"]


# Define common POIs and ranges
POI_R = ("r", (1, -20, 20))
POI_R_GGHH = ("r_gghh", (1, -20, 20))
POI_R_QQHH = ("r_qqhh", (1, -20, 20))

POI_A = ("A", (0, 0, 6))
POI_LA = ("LA", (0, -10, 10))
POI_M2 = ("M2", (0, 0, 3000))
POI_B = ("B", (0, 0, 6))
POI_MHE = ("MHE", (1000, 100, 3000))  # Heavy Higgs (H)
POI_MHP = ("MHP", (100, 100, 3000))  # Charged Higgs (H+)
POI_MA = ("MA", (100, 0, 3000))  # heavy higgs (A)
POI_Z6 = ("Z6", (0., 0, 3000))
POI_LQ = ("LQ", (0., -10, 10))
POI_MQ = ("MQ", (0., 0, 3000))
POI_XI = ("XI", (0, 0, 1))
POI_CV = ("CV", (1, -10, 10))
POI_C2V = ("C2V", (1, -10, 10))
POI_kl = ("kl_EFT", (1, -10, 10))
POI_kt = ("kt_EFT", (1, -10, 10))
POI_C2 = ("C2_EFT", (1, -10, 10))

DEF_R_POIS = OrderedDict([POI_R, POI_R_GGHH, POI_R_QQHH])


"""
Model I (real scalar singlet with explicit Z2 breaking) [arXiv:1704.07851, arXiv:1412.8480]
POIs: A(α), LA(λ_α), M2(m_2), CV, C2V
"""


class HHModel_Alpha_1(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_AlphaM2

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_A,
        POI_LA,
        POI_M2,
        POI_CV,
        POI_C2V,
    ])

    """
    kl = 1−3/2*tan(α)^2 + tan(α)^2 * (λ_α −tan(α)*(m2/nu))/λSM ; λSM=mh^2/(2*nu^2) ; nu=246 GeV
    kt = 1−tan(α)^2/2
    C2 = −tan(α)^2/2
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1-(3./2.)*pow(tan(@0),2)+pow(tan(@0),2)*(@1-tan(@0)*(@2/246.))/(pow(@3,2)/(2*pow(246,2)))', A, LA, M2, MH)")  # noqa
        self.make_expr("expr::C2('-pow(tan(@0),2)/2.', A)")
        self.make_expr("expr::kt('1-pow(tan(@0),2)/2.', A)")


"""
Model Ib (real scalar singlet with explicit Z2 breaking) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT (kl), kt_EFT (kt) CV, C2V
"""


class HHModel_Alpha_1b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_klkt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_kl,
        POI_kt,
        POI_CV,
        POI_C2V,
    ])

    """
    kl = kl
    kt = kt
    C2 = kt-1
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('@0', kl_EFT)")
        self.make_expr("expr::kt('@0', kt_EFT)")
        self.make_expr("expr::C2('@0', kt_EFT-1)")


"""
Model II (real scalar singlet with spontaneous Z2 breaking) [arXiv:1704.07851]
POIs: A(α), CV, C2V
"""


class HHModel_Alpha_2(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_Alpha

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_A,
        POI_CV,
        POI_C2V,
    ])

    """
    kl = 1-3/2*tan(α)^2
    kt = 1−tan(α)^2/2
    C2 = −tan(α)^2/2
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1-(3./2.)*pow(tan(@0),2)', A)")
        self.make_expr("expr::C2('-pow(tan(@0),2)/2.', A)")
        self.make_expr("expr::kt('1-pow(tan(@0),2)/2.', A)")


"""
Model IIb (real scalar singlet with spontaneous Z2 breaking) [arXiv:1704.07851]
POIs: kt_EFT (kt), CV, C2V
"""


class HHModel_Alpha_2b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_kt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_kt,
        POI_CV,
        POI_C2V,
    ])

    """
    kl = kt
    kt = kt
    C2 = kt-1
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('@0', kt_EFT)")
        self.make_expr("expr::kt('@0', kt_EFT)")
        self.make_expr("expr::C2('@0', kt_EFT-1)")


"""
Model III (real scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MHP(mH+), CV, C2V
"""


class HHModel_BETAMH_3(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHEMHP

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
        POI_MHP,
    ])

    """
    kl = 1 + 4*sin(β)^2*(3+(mH+^2)/(nu^2*λSM))*(mH+^4)/(mH^4); nu^2*λSM=mh^2/2
    kt = 1
    C2 = -2*sin(β)^2*(mH+^4)/(mH^4)
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1+4*pow(sin(@0),2)*(3 + 2*pow(@1,2)/pow(@2,2))*pow(@1,4)/pow(@3,4)', B, MHP, MH, MHE)")  # noqa
        self.make_expr("expr::C2('-2*pow(sin(@0),2)*pow(@1,4)/pow(@2,4)',B, MHP, MHE)")
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)


"""
Model IIIb (real scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT (kl), C2_EFT (C2), CV, C2V
"""


class HHModel_BETAMH_3b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_klc2

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kl,
        POI_C2,
    ])

    """
    kl = kl
    kt = 1
    C2 = C2
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('@0', kl_EFT)")
        self.make_expr("expr::C2('@0',C2_EFT)")
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)


"""
Model IIIc (real scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mH+), CV, C2V
"""


class HHModel_BETAMH_3c(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHE

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
    ])

    """
    kl = 1 + 4*sin(β)^2*(3+(mH+^2)/(nu^2*λSM))*(mH+^4)/(mH^4); nu^2*λSM=mh^2/2 ; mH+=mH
    kt = 1
    C2 = -2*sin(β)^2*(mH+^4)/(mH^4) ; mH+=mH
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1+4*pow(sin(@0),2)*(3 + 2*pow(@1,2)/pow(@2,2))*pow(@1,4)/pow(@1,4)', B, MHE, MH)")  # noqa
        self.make_expr("expr::C2('-2*pow(sin(@0),2)*pow(@1,4)/pow(@1,4)',B, MHE)")
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)


"""
Model IV (complex scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MA(mA), CV, C2V
"""


class HHModel_BETAMH_4(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHEMA

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
        POI_MA,
    ])

    """
    kl = 1 + 4*sin(β)^2*(3+(mA^2)/(nu^2*λSM))*(mA^4)/(mH^4); nu^2*λSM=mh^2/2
    kt = 1-2*sin(β)^2(mA^4)/(mH^4)
    C2 = -4*sin(β)^2*(mA^4)/(mH^4)
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1 + 2 * pow(sin(@0),2) * (3 + (4*pow(@1,2))/(pow(@2,2)/2))*pow(@1,4)/pow(@3,4)', B, MA, MH, MHE)")  # noqa
        self.make_expr("expr::kt('1-2*sin(@0)*sin(@0)*pow(@1,4)/pow(@2,4)',B,MA,MHE)")
        self.make_expr("expr::C2('-4*sin(@0)*pow(@1,4)/pow(@2,4)',B, MA, MHE)")


"""
Model IVb (complex scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT (kl), kt_EFT (kt) , CV, C2V
"""


class HHModel_BETAMH_4b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_klkt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kl,
        POI_kt,
    ])

    """
    kl = kl
    kt = kt
    C2 = 2*kt-1
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('@0', kl_EFT)")
        self.make_expr("expr::kt('@0', kt_EFT)")
        self.make_expr("expr::C2('2*@0-1', kt_EFT)")


"""
Model IVc (complex scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mA), CV, C2V
"""


class HHModel_BETAMH_4c(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHE

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
    ])

    """
    kl = 1 + 4*sin(β)^2*(3+(mA^2)/(nu^2*λSM))*(mA^4)/(mH^4); nu^2*λSM=mh^2/2 ; mA=mH
    kt = 1-2*sin(β)^2(mA^4)/(mH^4) ; mA=mH
    C2 = -4*sin(β)^2*(mA^4)/(mH^4) ; mA=mH
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1 + 2 * pow(sin(@0),2) * (3 + (4*pow(@1,2))/(pow(@2,2)/2))*pow(@1,4)/pow(@1,4)', B, MHE, MH)")  # noqa
        self.make_expr("expr::kt('1-2*sin(@0)*sin(@0)*pow(@1,4)/pow(@1,4)', B, MHE)")
        self.make_expr("expr::C2('-4*sin(@0)*pow(@1,4)/pow(@1,4)',B, MHE)")


"""
Model V (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MA(mA), CV, C2V
"""


class HHModel_BETAMH_5(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHEMA

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
        POI_MA,
    ])

    """
    kl = 1+24/7*tan(β)^2*(mA^4)/(mH^2*nu^2*λSM); nu^2*λSM=mh^2/2
    kt = 1
    C2 = 0
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1 + 24./7. * pow(tan(@0),2) * (pow(@1,4)/(pow(@3,2)*pow(@2,2)/2))', B, MA, MH, MHE)")  # noqa
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)
        self.make_var("{}[0]".format("C2"))
        self.get_var("C2").setConstant(True)


"""
Model Vb (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480]
Model VIb (quartet scalar with Y = 3/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT (kl), CV, C2V
"""


class HHModel_BETAMH_5b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_kl

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kl,
    ])

    """
    kl = kl
    kt = 1
    C2 = 0
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('@0', kl_EFT)")
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)
        self.make_var("{}[0]".format("C2"))
        self.get_var("C2").setConstant(True)


"""
Model Vc (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mA), CV, C2V
"""


class HHModel_BETAMH_5c(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHE

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
    ])

    """
    kl = 1+24/7*tan(β)^2*(mA^4)/(mH^2*nu^2*λSM); nu^2*λSM=mh^2/2 ; mH=mA
    kt = 1
    C2 = 0
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1 + 24./7. * pow(tan(@0),2) * (pow(@1,4)/(pow(@1,2)*pow(@2,2)/2))', B, MHE, MH)")  # noqa
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)
        self.make_var("{}[0]".format("C2"))
        self.get_var("C2").setConstant(True)


"""
Model VI (quartet scalar with Y = 3/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MA(mA), CV, C2V
"""


class HHModel_BETAMH_6(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHEMA

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
        POI_MA,
    ])

    """
    kl = 1+8/3*tan(β)^2*(mA^4)/(mH^2*nu^2*λSM); nu^2*λSM=mh^2/2
    kt = 1
    C2 = 0
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1 + 8./3. * pow(tan(@0),2) * (pow(@1,4)/(pow(@3,2)*pow(@2,2)/2))', B, MA, MH, MHE)")  # noqa
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)
        self.make_var("{}[0]".format("C2"))
        self.get_var("C2").setConstant(True)


"""
Model Vb (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480]
Model VIb (quartet scalar with Y = 3/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT (kl), CV, C2V
"""


HHModel_BETAMH_6b=HHModel_BETAMH_5b


"""
Model VIc (quartet scalar with Y = 3/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mA), CV, C2V
"""


class HHModel_BETAMH_6c(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHE

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
        POI_MA,
    ])

    """
    kl = 1+8/3*tan(β)^2*(mA^4)/(mH^2*nu^2*λSM); nu^2*λSM=mh^2/2 ; mA=mH
    kt = 1
    C2 = 0
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1 + 8./3. * pow(tan(@0),2) * (pow(@1,4)/(pow(@1,2)*pow(@2,2)/2))', B, MHE, MH)")  # noqa
        self.make_var("{}[1]".format("kt"))
        self.get_var("kt").setConstant(True)
        self.make_var("{}[0]".format("C2"))
        self.get_var("C2").setConstant(True)


"""
Model VII (2HDM (addtl. scalars heavy + Z2)) [arXiv:1611.01112]
POIs: B(β), MHE(mH), Z6(Z6), CV, C2V
"""


class HHModel_BETAMH_7(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_BetaMHEZ6

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_B,
        POI_MHE,
        POI_Z6,
    ])

    """
    kl = 1 - 3*(Z6^2*nu^2)/(2*λSM*mH^2); λSM=mh^2/(2*nu^2), nu=246 GeV
    kt = 1-(Z6*nu^2)/(tan(β)*mH^2)
    C2 = 1-(3*Z6*nu^2)/(2*tan(β)*mH^2)
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('1-(3*pow(@0,2)*pow(246,2))/(pow(@1,2)*pow(@2,2)/pow(246,2))',Z6, MH, MHE )")  # noqa
        self.make_expr("expr::kt('1-@0*pow(246,2)/(tan(@1)*pow(@2,2))',Z6, B, MHE )")
        self.make_expr("expr::C2('-3.*@0*pow(246,2)/(2.*tan(@1)*pow(@2,2))',Z6, B, MHE )")


"""
Model VIb (2HDM (addtl. scalars heavy + Z2)) [arXiv:1611.01112]
POIs: kl_EFT (kl), kt_EFT (kt) CV, C2V
"""


class HHModel_BETAMH_7b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_klkt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kl,
        POI_kt,
    ])

    """
    kl = kl
    kt = kt
    C2 = 3/2*(kt-1)
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('@0', kl_EFT)")
        self.make_expr("expr::kt('@0', kt_EFT)")
        self.make_expr("expr::C2('3*(@0-1)/2.', kt_EFT)")


"""
Model VIII (vector-like quark: T) [arXiv:hep-ph/0007316]
POIs: LQ(λ_Tt), MQ (MT), CV, C2V
"""


class HHModel_VLQ_8(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_VLQ

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_LQ,
        POI_MQ,
    ])

    """
    kl = 1
    kt = 1-Vtb * (|λ_Tt|^2 nu^2)/(2*MT^2)  ; nu=246 GeV, |Vtb| =1.009 (LHC+Tevatron average)
    C2 = -3*Vtb * (|λ_Tt|^2 nu^2)/(4*MT^2) ; nu=246 GeV, |Vtb| =1.009 (LHC+Tevatron average)
    """
    def make_eftconstraints(self):
        self.make_var("{}[1]".format("kl"))
        self.get_var("kl").setConstant(True)
        self.make_expr("expr::kt('1-1.009*pow(@0*246,2)/(2*pow(@1,2))', LQ, MQ )")
        self.make_expr("expr::C2('-3*1.009*pow(@0*246,2)/(4*pow(@1,2))', LQ, MQ )")


"""
Model VIIIb (vector-like quark: T) [arXiv:hep-ph/0007316]
POIs: kt_EFT (kt), CV, C2V
"""


class HHModel_VLQ_8b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_kt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kt,
    ])

    """
    kl = 1
    kt = kt
    C2 = 3./2*(kt-1)
    """
    def make_eftconstraints(self):
        self.make_var("{}[1]".format("kl"))
        self.get_var("kl").setConstant(True)
        self.make_expr("expr::kt('@0', kt_EFT )")
        self.make_expr("expr::C2('3*(@0-1)/2', kt_EFT )")


"""
Model IX (vector-like quark: E) [arXiv:0803.4008]
POIs: LQ(λ_El), MQ (ME), CV, C2V
"""


class HHModel_VLQ_9(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_VLQ

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_LQ,
        POI_MQ,
    ])

    """
    kl = 1 + (|λ_El|^2 * nu^2) / (4*ME^2) ; nu=246 GeV
    kt = 1 + (|λ_El|^2 * nu^2) / (4*ME^2) ; nu=246 GeV
    C2 = 0
    """
    def make_eftconstraints(self):
        self.make_var("{}[0]".format("C2"))
        self.get_var("C2").setConstant(True)
        self.make_expr("expr::kl('1+pow(@0*246,2)/(4*pow(@1,2))', LQ, MQ )")
        self.make_expr("expr::kt('1+pow(@0*246,2)/(4*pow(@1,2))', LQ, MQ )")


"""
Model IXb (vector-like quark: E) [arXiv:0803.4008]
POIs: kt_EFT (kt), CV, C2V
"""


class HHModel_VLQ_9b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_kt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kt,
    ])

    """
    kl = kt
    kt = kt
    C2 = 0
    """
    def make_eftconstraints(self):
        self.make_var("{}[0]".format("C2"))
        self.get_var("C2").setConstant(True)
        self.make_expr("expr::kl('@0', kt_EFT )")
        self.make_expr("expr::kt('@0', kt_EFT )")


"""
Model X (MCHM5) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: XI(ξ)
"""


class HHModel_XI_10(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_Xi

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_XI,
    ])

    """
    kl = (1-2*ξ)/sqrt(1-ξ)
    kt = (1-2*ξ)/sqrt(1-ξ)
    C2 = -2*ξ
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('(1-2*@0)/sqrt(1-@0)', XI )")
        self.make_expr("expr::kt('(1-2*@0)/sqrt(1-@0)', XI )")
        self.make_expr("expr::C2('-2*@0', XI )")


"""
Model Xb (MCHM5) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: kt_EFT (kt), CV, C2V
"""


class HHModel_XI_10b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_kt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kt,
    ])

    """
    kl = kt
    kt = kt
    C2 = kt*(kt+sqrt(kt^2+8))/4-1
    """
    def make_eftconstraints(self):
        self.make_expr("expr::kl('@0', kt_EFT )")
        self.make_expr("expr::kt('@0', kt_EFT )")
        self.make_expr("expr::C2('@0*(@0+sqrt(pow(@0,2)+8))/4.-1', kt_EFT )")


"""
Model XI (MCHM4) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: XI(ξ)
"""


class HHModel_XI_11(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_Xi

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_XI,
    ])

    """
    kl = sqrt(1-ξ)
    kt = sqrt(1-ξ)
    C2 = -0.5*ξ
    """
    def make_eftconstraints(self):
        # model 1
        self.make_expr("expr::kl('sqrt(1-@0)', XI )")
        self.make_expr("expr::kt('sqrt(1-@0)', XI )")
        self.make_expr("expr::C2('-0.5*@0', XI )")


"""
Model XIb (MCHM4) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: kt_EFT (kt)
"""


class HHModel_XI_11b(HHModelEFTBase):
    h_br_scaler_cls = HBRScaler_kt

    R_POIS = DEF_R_POIS
    K_POIS = OrderedDict([
        POI_CV,
        POI_C2V,
        POI_kt,
    ])

    """
    kl = kt
    kt = kt
    C2 = (kt^2-1)/2
    """
    def make_eftconstraints(self):
        # model 1
        self.make_expr("expr::kl('@0', kt_EFT )")
        self.make_expr("expr::kt('@0', kt_EFT )")
        self.make_expr("expr::C2('(pow(@0,2)-1)/2', kt_EFT )")


# Addabt to build models for the various theory cases we implemented here
def create_model(name, HHModel=HHModel_Alpha_1, ggf=None, vbf=None, **kwargs):
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


# default models

"""
Model I (real scalar singlet with explicit Z2 breaking) [arXiv:1704.07851, arXiv:1412.8480]
POIs: A(α), LA(λ_α), M2(m_2), CV, C2V
--hh-model hh_model_C2klkt_EFT.alpha_1_model_default
"""
alpha_1_model_default = create_model(
    "alpah_1_model_default",
    HHModel=HHModel_Alpha_1,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model Ib (real scalar singlet with explicit Z2 breaking) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT(kl), kt_EFT(kt), CV, C2V
--hh-model hh_model_C2klkt_EFT.alpha_1b_model_default
"""
alpha_1b_model_default = create_model(
    "alpah_1b_model_default",
    HHModel=HHModel_Alpha_1b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model II (real scalar singlet with spontaneous Z2 breaking) [arXiv:1704.07851]
POIs: A(α), CV, C2V
--hh-model hh_model_C2klkt_EFT.alpha_2_model_default
"""
alpha_2_model_default = create_model(
    "alpha_2_model_default",
    HHModel=HHModel_Alpha_2,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IIb (real scalar singlet with spontaneous Z2 breaking) [arXiv:1704.07851]
POIs: kt_EFT (kt), CV, C2V
--hh-model hh_model_C2klkt_EFT.alpha_2b_model_default
"""
alpha_2b_model_default = create_model(
    "alpha_2b_model_default",
    HHModel=HHModel_Alpha_2b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model III (real scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MHP(mH+), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_3_model_default
"""
betamh_3_model_default = create_model(
    "betamh_3_model_default",
    HHModel=HHModel_BETAMH_3,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IIIb (real scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT(kl), C2_EFT(C2) CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_3b_model_default
"""
betamh_3b_model_default = create_model(
    "betamh_3b_model_default",
    HHModel=HHModel_BETAMH_3b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IIIc (real scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mH+), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_3c_model_default
"""
betamh_3c_model_default = create_model(
    "betamh_3c_model_default",
    HHModel=HHModel_BETAMH_3c,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IV (complex scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MA(mA), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_4_model_default
"""
betamh_4_model_default = create_model(
    "betamh_4_model_default",
    HHModel=HHModel_BETAMH_4,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IVb (complex scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: kl_EFT(kl), kt_EFT(kt), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_4b_model_default
"""
betamh_4b_model_default = create_model(
    "betamh_4b_model_default",
    HHModel=HHModel_BETAMH_4b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IVc (complex scalar triplet) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mA), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_4c_model_default
"""
betamh_4c_model_default = create_model(
    "betamh_4c_model_default",
    HHModel=HHModel_BETAMH_4c,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model V (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MA(mA), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_5_model_default
"""
betamh_5_model_default = create_model(
    "betamh_5_model_default",
    HHModel=HHModel_BETAMH_5,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model Vb (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480
Model VIb (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480]]
POIs: kl, CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_5b_model_default
"""
betamh_5b_model_default = create_model(
    "betamh_5b_model_default",
    HHModel=HHModel_BETAMH_5b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

betamh_6b_model_default = create_model(
    "betamh_6b_model_default",
    HHModel=HHModel_BETAMH_6b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model Vc (quartet scalar with Y = 1/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mA), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_5c_model_default
"""
betamh_5c_model_default = create_model(
    "betamh_5c_model_default",
    HHModel=HHModel_BETAMH_5c,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model VI (quartet scalar with Y = 3/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH), MA(mA), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_6_model_default (β/mH scan)
"""
betamh_6_model_default = create_model(
    "betamh_6_model_default",
    HHModel=HHModel_BETAMH_6,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model VIc (quartet scalar with Y = 3/2) [arXiv:1704.07851, arXiv:1412.8480]
POIs: B(β), MHE(mH=mA), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_6c_model_default (β/mH scan)
"""
betamh_6c_model_default = create_model(
    "betamh_6c_model_default",
    HHModel=HHModel_BETAMH_6c,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model VII (2HDM (addtl. scalars heavy + Z2)) [arXiv:1611.01112]
POIs: B(β), MHE(mH), Z6(Z6), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_7_model_default
"""
betamh_7_model_default = create_model(
    "betamh_7_model_default",
    HHModel=HHModel_BETAMH_7,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model VIIb (2HDM (addtl. scalars heavy + Z2)) [arXiv:1611.01112]
POIs: kl_EFT(kl), kt_EFT(kt), CV, C2V
--hh-model hh_model_C2klkt_EFT.betamh_7b_model_default
"""
betamh_7b_model_default = create_model(
    "betamh_7b_model_default",
    HHModel=HHModel_BETAMH_7b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model VIII (vector-like quark: T) [arXiv:hep-ph/0007316]
POIs: LQ(λ_Tt), MQ (MT), CV, C2V
--hh-model hh_model_C2klkt_EFT.vlq_8_model_default
"""
vlq_8_model_default = create_model(
    "betamh_8_model_default",
    HHModel=HHModel_VLQ_8,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model VIIIb (vector-like quark: T) [arXiv:hep-ph/0007316]
POIs: kt_EFT (kt), CV, C2V
--hh-model hh_model_C2klkt_EFT.vlq_8b_model_default
"""
vlq_8b_model_default = create_model(
    "betamh_8b_model_default",
    HHModel=HHModel_VLQ_8b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IX (vector-like quark: E) [arXiv:0803.4008]
POIs: LQ(λ_El), MQ (ME), CV, C2V
--hh-model hh_model_C2klkt_EFT.vlq_9_model_default
"""
vlq_9_model_default = create_model(
    "betamh_9_model_default",
    HHModel=HHModel_VLQ_9,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model IXb (vector-like quark: E) [arXiv:0803.4008]
POIs: kt_EFT, CV, C2V
--hh-model hh_model_C2klkt_EFT.vlq_9b_model_default
"""
vlq_9b_model_default = create_model(
    "betamh_9b_model_default",
    HHModel=HHModel_VLQ_9b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model X (MCHM5) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: XI(ξ)
--hh-model hh_model_C2klkt_EFT.xi_10_model_default (ξ scan)
"""
xi_10_model_default = create_model(
    "xi_10_model_default",
    HHModel=HHModel_XI_10,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model Xb (MCHM5) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: kt_EFT(kt)
--hh-model hh_model_C2klkt_EFT.xi_10b_model_default (ξ scan)
"""
xi_10b_model_default = create_model(
    "xi_10b_model_default",
    HHModel=HHModel_XI_10b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model XI (MCHM4) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: XI(ξ)
--hh-model hh_model_C2klkt_EFT.xi_11_model_default (ξ scan)
"""
xi_11_model_default = create_model(
    "xi_11_model_default",
    HHModel=HHModel_XI_11,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)

"""
Model XIb (MCHM4) [arXiv:hep-ph/0703164, arXiv:1012.1562, arXiv:1303.3876, arXiv:1005.4269]
POIs: kt_EFT (kt)
--hh-model hh_model_C2klkt_EFT.xi_11b_model_default (ξ scan)
"""
xi_11b_model_default = create_model(
    "xi_11b_model_default",
    HHModel=HHModel_XI_11b,
    ggf=[(0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3)],
    vbf=[(1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1)],
)
