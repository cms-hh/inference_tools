# coding: utf-8

"""
Task tests based on the $DHI_EXAMPLE_CARDS.
"""


import os
from collections import OrderedDict

import luigi

from dhi.tasks.base import AnalysisTask, PlotTask, view_output_plots
from dhi.tasks.limits import (
    PlotUpperLimits, PlotMultipleUpperLimits, PlotMultipleUpperLimitsByModel, PlotUpperLimitsAtPOI,
)
from dhi.tasks.likelihoods import PlotLikelihoodScan
from dhi.tasks.significances import PlotSignificanceScan, PlotMultipleSignificanceScans
from dhi.tasks.pulls_impacts import PlotPullsAndImpacts
from dhi.tasks.exclusion import PlotExclusionAndBestFit, PlotExclusionAndBestFit2D
from dhi.tasks.postfit_shapes import PlotPostfitSOverB
from dhi.tasks.studies.model_selection import (
    PlotMorphingScales, PlotMorphedDiscriminant, PlotStatErrorScan,
)


class TestPlots(AnalysisTask):

    task_namespace = "test"

    no_upper_limits = luigi.BoolParameter(default=False)
    no_multiple_upper_limits = luigi.BoolParameter(default=False)
    no_multiple_upper_limits_by_model = luigi.BoolParameter(default=False)
    no_upper_limits_at_poi = luigi.BoolParameter(default=False)
    no_likelihood_scan = luigi.BoolParameter(default=False)
    no_likelihood_scan_2d = luigi.BoolParameter(default=False)
    no_significance_scan = luigi.BoolParameter(default=False)
    no_multiple_significance_scans = luigi.BoolParameter(default=False)
    no_pulls_and_impacts = luigi.BoolParameter(default=False)
    no_exclusion_and_bestfit = luigi.BoolParameter(default=False)
    no_exclusion_and_bestfit_2d = luigi.BoolParameter(default=False)
    no_postfit_s_over_b = luigi.BoolParameter(default=False)
    no_morphing_scales = luigi.BoolParameter(default=False)
    no_morphed_discriminant = luigi.BoolParameter(default=False)
    no_stat_error_scan = luigi.BoolParameter(default=False)

    file_type = PlotTask.file_type
    campaign = PlotTask.campaign
    view_cmd = PlotTask.view_cmd

    exclude_params_req = {"view_cmd"}

    def requires(self):
        reqs = OrderedDict()

        all_cards = tuple(os.environ["DHI_EXAMPLE_CARDS"].split(","))
        test_cards = tuple(card for card in all_cards if "GluGlu" in card)
        test_models = tuple("HHModelPinv.model_no_ggf_kl" + kl for kl in "0 1 2p45 5".split())

        if not self.no_upper_limits:
            reqs["upper_limits"] = PlotUpperLimits.req(self,
                datacards=test_cards,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if not self.no_multiple_upper_limits:
            reqs["multiple_upper_limits"] = PlotMultipleUpperLimits.req(self,
                multi_datacards=tuple((card,) for card in test_cards),
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if not self.no_multiple_upper_limits_by_model:
            reqs["multiple_upper_limits_by_model"] = PlotMultipleUpperLimitsByModel.req(self,
                datacards=test_cards,
                hh_models=test_models,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if not self.no_upper_limits_at_poi:
            reqs["upper_limits_at_poi"] = PlotUpperLimitsAtPOI.req(self,
                multi_datacards=tuple((card,) for card in test_cards) + (test_cards,),
                pois=("r",),
                parameter_values=("kl=1", "kt=1"),
            )

        if not self.no_likelihood_scan:
            reqs["likelihood_scan"] = PlotLikelihoodScan.req(self,
                datacards=test_cards,
                pois=("kl",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if not self.no_likelihood_scan_2d:
            reqs["likelihood_scan_2d"] = PlotLikelihoodScan.req(self,
                datacards=test_cards,
                pois=("kl", "kt"),
                scan_parameters=(("kl", -5., 5.), ("kt", -5., 5.),),
            )

        if not self.no_significance_scan:
            reqs["significance_scan"] = PlotSignificanceScan.req(self,
                datacards=test_cards,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if not self.no_multiple_significance_scans:
            reqs["multiple_significance_scans"] = PlotMultipleSignificanceScans.req(self,
                multi_datacards=tuple((card,) for card in test_cards) + (test_cards,),
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if not self.no_pulls_and_impacts:
            reqs["pulls_and_impacts"] = PlotPullsAndImpacts.req(self,
                datacards=test_cards,
                pois=("r",),
            )
            reqs["pulls_and_impacts"].requires().requires().end_branch = 10

        if not self.no_exclusion_and_bestfit:
            reqs["exclusion_and_bestfit"] = PlotExclusionAndBestFit.req(self,
                multi_datacards=tuple((card,) for card in test_cards) + (test_cards,),
                pois=("r",),
                scan_parameters=(("kl", -30., 30., 61),),
            )

        if not self.no_exclusion_and_bestfit_2d:
            reqs["exclusion_and_bestfit_2d"] = PlotExclusionAndBestFit2D.req(self,
                datacards=test_cards,
                pois=("r",),
                scan_parameters=(("kl", -30., 30., 61), ("kt", -6., 9., 31)),
            )

        if not self.no_postfit_s_over_b:
            reqs["postfit_s_over_b"] = PlotPostfitSOverB.req(self,
                datacards=test_cards,
                pois=("r",),
            )

        if not self.no_morphing_scales:
            reqs["morphing_scales"] = PlotMorphingScales.req(self,
                hh_model="HHModelPinv.model_default",
                signal="ggf",
                scan_parameters=(("kl", -10., 10.),),
                parameter_values=("kt=1",),
            )

        if not self.no_morphed_discriminant:
            reqs["morphed_discriminant"] = PlotMorphedDiscriminant.req(self,
                datacards=test_cards[:1],
                hh_models=test_models,
                signal="ggf",
                bins=("ee_HH_ggF",),
                parameter_values=("kl=1", "kt=1"),
            )

        if not self.no_stat_error_scan:
            reqs["stat_error_scan"] = PlotStatErrorScan.req(self,
                datacards=test_cards[:1],
                hh_models=test_models,
                signal="ggf",
                bins=("ee_HH_ggF",),
                scan_parameters=(("kl", -20, 20, 81),),
                parameter_values=("kt=1",),
            )

        return reqs

    def output(self):
        return self.input()

    @view_output_plots
    def run(self):
        pass
