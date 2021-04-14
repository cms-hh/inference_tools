# coding: utf-8

"""
Collection of tools for the EFT workflow.
"""

__all__ = []


class EFTCrossSectionProvider(object):
    """
    Helper class to calculate HH cross sections in EFT, as usualy in units of pb.
    Coefficients and formulae are taken from
    https://github.com/fabio-mon/HHStatAnalysis/blob/c8fc33d2ae3f7e04cfc83e773e2880657ffdce3b/AnalyticalModels/python/NonResonantModelNLO.py
    with credits to F. Monti and P. Mandrik.
    """

    def __init__(self):
        super(EFTCrossSectionProvider, self).__init__()

        # various coefficients
        # from https://github.com/pmandrik/VSEVA/blob/f7224649297f900a4ae25cf721d65cae8bd7b408/HHWWgg/reweight/reweight_HH.C#L117
        self.coeffs_ggf_nlo_13tev = [
            62.5088, 345.604, 9.63451, 4.34841, 39.0143, -268.644, -44.2924, 96.5595, 53.515,
            -155.793, -23.678, 54.5601, 12.2273, -26.8654, -19.3723, -0.0904439, 0.321092, 0.452381,
            -0.0190758, -0.607163, 1.27408, 0.364487, -0.499263,
        ]

    def get_ggf_xsec(self, kl=1., kt=1., c2=1., cg=1., c2g=1., coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs_ggf_nlo_13tev

        return coeffs[0] * kt**4 + \
            coeffs[1] * c2**2 + \
            coeffs[2] * kt**2 * kl**2 + \
            coeffs[3] * cg**2 * kl**2 + \
            coeffs[4] * c2g**2 + \
            coeffs[5] * c2 * kt**2 + \
            coeffs[6] * kl * kt**3 + \
            coeffs[7] * kt * kl * c2 + \
            coeffs[8] * cg * kl * c2 + \
            coeffs[9] * c2 * c2g + \
            coeffs[10] * cg * kl * kt**2 + \
            coeffs[11] * c2g * kt**2 + \
            coeffs[12] * kl**2 * cg * kt + \
            coeffs[13] * c2g * kt * kl + \
            coeffs[14] * cg * c2g * kl + \
            coeffs[15] * kt**3 * cg + \
            coeffs[16] * kt * c2 * cg + \
            coeffs[17] * kt * cg**2 * kl + \
            coeffs[18] * cg * kt * c2g + \
            coeffs[19] * kt**2 * cg**2 + \
            coeffs[20] * c2 * cg**2 + \
            coeffs[21] * cg**3 * kl + \
            coeffs[22] * cg**2 * c2g


#: EFTCrossSectionProvider singleton.
eft_xsec_provider = EFTCrossSectionProvider()

#: Default ggF cross section getter.
get_ggf_xsec = eft_xsec_provider.get_ggf_xsec
