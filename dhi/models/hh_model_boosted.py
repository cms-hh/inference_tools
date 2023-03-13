# coding: utf-8

"""
Custom HH physics model that adds additional boosted samples on top of the normal ggf and vbf
samples to support additional base points that help maintaining positive signal pdfs.

Authors:
  - Javier Duarte
  - Marcel Rieger
"""


from collections import OrderedDict

# we need a wildcard import to have everything accessible through this module
from hh_model import *  # noqa
# specific imports for linting
from hh_model import (
    GGFSample as DefaultGGFSample, VBFSample as DefaultVBFSample, VHHSample,
    GGFFormula as DefaultGGFFormula, VBFFormula as DefaultVBFFormula, HHModel as DefaultHHModel,
    ggf_samples, vbf_samples, vhh_samples, _create_add_sample_func, model_default_vhh,
)


####################################################################################################
# boosted samples
####################################################################################################

class BoostedGGFSample(DefaultGGFSample):

    # label format
    label_re = r"^boosted_ggHH_kl_([pm0-9]+)_kt_([pm0-9]+)$"


class BoostedVBFSample(DefaultVBFSample):

    # label format
    label_re = r"^boosted_qqHH_CV_([pm0-9]+)_C2V_([pm0-9]+)_kl_([pm0-9]+)$"


# boosted ggf samples
# note that all samples listed here are used by model_boosted below
boosted_ggf_samples = OrderedDict()
add_boosted_ggf_sample = _create_add_sample_func(BoostedGGFSample, boosted_ggf_samples)
add_boosted_ggf_sample(kl=0.0, kt=1.0, xs=0.069725, label="boosted_ggHH_kl_0_kt_1")
add_boosted_ggf_sample(kl=1.0, kt=1.0, xs=0.031047, label="boosted_ggHH_kl_1_kt_1")
add_boosted_ggf_sample(kl=2.45, kt=1.0, xs=0.013124, label="boosted_ggHH_kl_2p45_kt_1")
add_boosted_ggf_sample(kl=5.0, kt=1.0, xs=0.091172, label="boosted_ggHH_kl_5_kt_1")
# additional base points for stabilizing signal pdfs
add_boosted_ggf_sample(kl=-10.0, kt=1.0, xs=1.638123, label="boosted_ggHH_kl_m10_kt_1")
add_boosted_ggf_sample(kl=20.0, kt=1.0, xs=3.378093, label="boosted_ggHH_kl_20_kt_1")
add_boosted_ggf_sample(kl=-15.0, kt=1.0, xs=3.227967, label="boosted_ggHH_kl_m15_kt_1")
add_boosted_ggf_sample(kl=25.0, kt=1.0, xs=5.547927, label="boosted_ggHH_kl_25_kt_1")
add_boosted_ggf_sample(kl=-3.0, kt=1.0, xs=0.314664, label="boosted_ggHH_kl_m3_kt_1")

# boosted vbf samples
# note that all samples listed here are used by model_boosted below
boosted_vbf_samples = OrderedDict()
add_boosted_vbf_sample = _create_add_sample_func(BoostedVBFSample, boosted_vbf_samples)
add_boosted_vbf_sample(CV=1.0, C2V=1.0, kl=1.0, xs=0.0017260, label="boosted_qqHH_CV_1_C2V_1_kl_1")
add_boosted_vbf_sample(CV=1.0, C2V=1.0, kl=0.0, xs=0.0046089, label="boosted_qqHH_CV_1_C2V_1_kl_0")
add_boosted_vbf_sample(CV=1.0, C2V=1.0, kl=2.0, xs=0.0014228, label="boosted_qqHH_CV_1_C2V_1_kl_2")
add_boosted_vbf_sample(CV=1.0, C2V=0.0, kl=1.0, xs=0.0270800, label="boosted_qqHH_CV_1_C2V_0_kl_1")
add_boosted_vbf_sample(CV=1.0, C2V=2.0, kl=1.0, xs=0.0142178, label="boosted_qqHH_CV_1_C2V_2_kl_1")
add_boosted_vbf_sample(CV=0.5, C2V=1.0, kl=1.0, xs=0.0108237, label="boosted_qqHH_CV_0p5_C2V_1_kl_1")
add_boosted_vbf_sample(CV=1.5, C2V=1.0, kl=1.0, xs=0.0660185, label="boosted_qqHH_CV_1p5_C2V_1_kl_1")
# additional base points for stabilizing signal pdfs
add_boosted_vbf_sample(CV=1.0, C2V=2.0, kl=-10.0, xs=0.1149221, label="boosted_qqHH_CV_1_C2V_2_kl_m10")  # noqa
add_boosted_vbf_sample(CV=1.0, C2V=2.0, kl=20.0, xs=0.5754886, label="boosted_qqHH_CV_1_C2V_2_kl_20")
add_boosted_vbf_sample(CV=1.0, C2V=0.0, kl=-10.0, xs=0.2735665, label="boosted_qqHH_CV_1_C2V_0_kl_m10")  # noqa
add_boosted_vbf_sample(CV=1.0, C2V=0.0, kl=20.0, xs=0.3365450, label="boosted_qqHH_CV_1_C2V_0_kl_20")
add_boosted_vbf_sample(CV=1.0, C2V=1.5, kl=-15.0, xs=0.3059198, label="boosted_qqHH_CV_1_C2V_1p5_kl_m15")  # noqa
add_boosted_vbf_sample(CV=1.0, C2V=0.5, kl=25.0, xs=0.6348751, label="boosted_qqHH_CV_1_C2V_0p5_kl_25")  # noqa
add_boosted_vbf_sample(CV=1.0, C2V=1.0, kl=-3.0, xs=0.0287358, label="boosted_qqHH_CV_1_C2V_1_kl_m3")


####################################################################################################
# updated cross section formulae to allow custom formulae
####################################################################################################

class GGFFormula(DefaultGGFFormula):

    sample_cls = (DefaultGGFSample, BoostedGGFSample)


class VBFFormula(DefaultVBFFormula):

    sample_cls = (DefaultVBFSample, BoostedVBFSample)


class HHModel(DefaultHHModel):

    # updated formula classes
    ggf_formula_cls = GGFFormula
    vbf_formula_cls = VBFFormula

    def __init__(self, name, ggf_samples=None, vbf_samples=None, vhh_samples=None,
            boosted_ggf_samples=None, boosted_vbf_samples=None):
        # attributes
        self.boosted_ggf_formula = (
            self.ggf_formula_cls(boosted_ggf_samples)
            if boosted_ggf_samples
            else None
        )
        self.boosted_vbf_formula = (
            self.vbf_formula_cls(boosted_vbf_samples)
            if boosted_vbf_samples
            else None
        )

        # invoke super init
        super(HHModel, self).__init__(name, ggf_samples=ggf_samples, vbf_samples=vbf_samples,
            vhh_samples=vhh_samples)

        # reset instance-level pois
        self.reset_pois()

    def get_formulae(self, xs_only=False):
        formulae = super(HHModel, self).get_formulae(xs_only=xs_only)
        if not xs_only and self.boosted_ggf_formula:
            formulae["boosted_ggf_formula"] = self.boosted_ggf_formula
        if not xs_only and self.boosted_vbf_formula:
            formulae["boosted_vbf_formula"] = self.boosted_vbf_formula
        return formulae


def create_model(name, ggf=None, vbf=None, vhh=None, boosted_ggf=None, boosted_vbf=None, **kwargs):
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
                    "sample '{}' is neither an instance of {}, nor does it correspond "
                    "to a known sample".format(s, sample_cls),
                )
        return samples

    # create the return the model
    return HHModel(
        name=name,
        ggf_samples=get_samples(ggf, ggf_samples, DefaultGGFSample),
        vbf_samples=get_samples(vbf, vbf_samples, DefaultVBFSample),
        vhh_samples=get_samples(vhh, vhh_samples, VHHSample),
        boosted_ggf_samples=get_samples(boosted_ggf, boosted_ggf_samples, BoostedGGFSample),
        boosted_vbf_samples=get_samples(boosted_vbf, boosted_vbf_samples, BoostedVBFSample),
        **kwargs  # noqa
    )


# boosted model that does not add additional base points for closure tests
model_boosted_closure = create_model("model_boosted_closure",
    ggf=[
        (1, 1), (2.45, 1), (5, 1),  # no (1, 0)
    ],
    vbf=[
        (1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1),  # no (0.5, 1, 1)
    ],
    boosted_ggf=[
        (1, 1), (2.45, 1), (5, 1),  # no (1, 0)
    ],
    boosted_vbf=[
        (1, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (1.5, 1, 1),  # no (0.5, 1, 1)
    ],
)

# boosted model that includes additional base points
model_boosted = create_model("model_boosted",
    ggf=model_boosted_closure.ggf_formula.samples,
    vbf=model_boosted_closure.vbf_formula.samples,
    boosted_ggf=[
        key
        for key in boosted_ggf_samples.keys()
        if key not in [(-3, 1), (1, 0)]  # no (-3, 1), (1, 0)
    ],
    boosted_vbf=[
        key
        for key in boosted_vbf_samples.keys()
        if key not in [(1, 1, -3), (0.5, 1, 1)]  # no (1, 1, -3), (0.5, 1, 1)
    ],
)

# boosted model that includes default vhh points
model_boosted_vhh = create_model(
    "model_boosted_vhh",
    ggf=model_boosted.ggf_formula.samples,
    vbf=model_boosted.vbf_formula.samples,
    vhh=model_default_vhh.vhh_formula.samples,
)
