# coding: utf-8

"""
Collection of tools for the EFT workflow.
"""

__all__ = []


import re

from dhi.util import make_list


class EFTCrossSectionProvider(object):
    """
    Helper class to calculate HH EFT cross sections in units of pb.
    Coefficients and formulae are taken from
    https://github.com/fabio-mon/HHStatAnalysis/blob/c8fc33d2ae3f7e04cfc83e773e2880657ffdce3b/AnalyticalModels/python/NonResonantModelNLO.py
    with credits to F. Monti and P. Mandrik.
    """

    def __init__(self):
        super(EFTCrossSectionProvider, self).__init__()

        # ggf nlo coefficients in pb, converted from fb values
        # https://github.com/pmandrik/VSEVA/blob/23daf2b9966ddbcef3fa6622957d17640ab9e133/HHWWgg/reweight/reweight_HH.C#L117
        self.coeffs_ggf_nlo_13tev = [0.001 * c for c in [
            62.5088, 345.604, 9.63451, 4.34841, 39.0143, -268.644, -44.2924, 96.5595, 53.515,
            -155.793, -23.678, 54.5601, 12.2273, -26.8654, -19.3723, -0.0904439, 0.321092, 0.452381,
            -0.0190758, -0.607163, 1.27408, 0.364487, -0.499263,
        ]]

        self.ggf_xsec_sm_nnlo = 0.03105  # pb

    def get_ggf_xsec_nlo(self, kl=1., kt=1., c2=0., cg=0., c2g=0., coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs_ggf_nlo_13tev

        return (
            coeffs[0] * kt**4 +
            coeffs[1] * c2**2 +
            coeffs[2] * kt**2 * kl**2 +
            coeffs[3] * cg**2 * kl**2 +
            coeffs[4] * c2g**2 +
            coeffs[5] * c2 * kt**2 +
            coeffs[6] * kl * kt**3 +
            coeffs[7] * kt * kl * c2 +
            coeffs[8] * cg * kl * c2 +
            coeffs[9] * c2 * c2g +
            coeffs[10] * cg * kl * kt**2 +
            coeffs[11] * c2g * kt**2 +
            coeffs[12] * kl**2 * cg * kt +
            coeffs[13] * c2g * kt * kl +
            coeffs[14] * cg * c2g * kl +
            coeffs[15] * kt**3 * cg +
            coeffs[16] * kt * c2 * cg +
            coeffs[17] * kt * cg**2 * kl +
            coeffs[18] * cg * kt * c2g +
            coeffs[19] * kt**2 * cg**2 +
            coeffs[20] * c2 * cg**2 +
            coeffs[21] * cg**3 * kl +
            coeffs[22] * cg**2 * c2g
        )

    def get_ggf_xsec_nnlo(self, kl=1., kt=1., c2=0., cg=0., c2g=0., coeffs=None):
        xsec_bsm_nlo = self.get_ggf_xsec_nlo(kl=kl, kt=kt, c2=c2, cg=cg, c2g=c2g, coeffs=coeffs)

        xsec_sm_nlo = self.get_ggf_xsec_nlo(kl=1., kt=1., c2=0., cg=0., c2g=0., coeffs=coeffs)
        k_factor = self.ggf_xsec_sm_nnlo / xsec_sm_nlo

        return xsec_bsm_nlo * k_factor


#: EFTCrossSectionProvider singleton.
eft_xsec_provider = EFTCrossSectionProvider()

#: Default ggF NLO cross section getter.
get_eft_ggf_xsec_nlo = eft_xsec_provider.get_ggf_xsec_nlo

#: Default ggF NNLO cross section getter.
get_eft_ggf_xsec_nnlo = eft_xsec_provider.get_ggf_xsec_nnlo


def sort_eft_benchmark_names(names):
    """
    Example order: 1, 2, 3, 3a, 3b, 4, 5, a_string, other_string, z_string
    """
    names = make_list(names)

    # split into names being a number or starting with one, and pure strings
    # store numeric names as tuples as sorted() will do exactly what we want
    num_names, str_names = [], []
    for name in names:
        m = re.match(r"^(\d+)(.*)$", name)
        if m:
            num_names.append((int(m.group(1)), m.group(2)))
        else:
            str_names.append(name)

    # sort and add
    num_names.sort()
    str_names.sort()
    return ["{}{}".format(*pair) for pair in num_names] + str_names


def extract_eft_scan_parameter(name):
    """
    c2_1p5 -> c2
    """
    if "_" not in name:
        raise ValueError("invalid datacard name '{}'".format(name))
    return name.split("_", 1)[0]


def sort_eft_scan_names(scan_parameter, names):
    """
    Names have the format "<scan_parameters>_<number>"" where number might have "-" replaced by
    "m" and "." replaced by "d", so revert this and simply sort by number.
    """
    names = make_list(names)

    # extract the scan values
    values = []
    for name in names:
        if not name.startswith(scan_parameter + "_"):
            raise ValueError("invalid datacard name '{}'".format(name))
        v = name[len(scan_parameter) + 1:].replace("d", ".").replace("m", "-")
        try:
            v = float(v)
        except ValueError:
            raise ValueError("invalid scan value '{}' in datacard name '{}'".format(v, name))
        values.append((name, v))

    # sort by value
    values.sort(key=lambda tpl: tpl[1])

    return values
