# coding: utf-8

"""
Task tests based on the $DHI_EXAMPLE_CARDS.
"""

import os
from collections import OrderedDict

import law
import luigi
import six

from dhi.tasks.base import AnalysisTask, PlotTask, view_output_plots
from dhi.tasks.limits import (
    PlotUpperLimits, PlotMultipleUpperLimits, PlotMultipleUpperLimitsByModel, PlotUpperLimitsAtPoint,
)
from dhi.tasks.likelihoods import (
    PlotLikelihoodScan,
    PlotMultipleLikelihoodScans,
    PlotMultipleLikelihoodScansByModel,
)
from dhi.tasks.significances import PlotSignificanceScan, PlotMultipleSignificanceScans
from dhi.tasks.pulls_impacts import PlotPullsAndImpacts
from dhi.tasks.exclusion import PlotExclusionAndBestFit, PlotExclusionAndBestFit2D
from dhi.tasks.postfit_shapes import PlotPostfitSOverB
from dhi.tasks.gof import PlotGoodnessOfFit, PlotMultipleGoodnessOfFits
from dhi.tasks.studies.model_selection import (
    PlotMorphingScales, PlotMorphedDiscriminant, PlotStatErrorScan,
)


class TestRegister(law.task.base.Register):

    def __new__(metacls, classname, bases, classdict):
        # convert test names into "--no-<name>" and "--only-<name>" task parameters
        for test_name in classdict.get("test_names", []):
            classdict["no_" + test_name] = luigi.BoolParameter(default=False)
            classdict["only_" + test_name] = luigi.BoolParameter(default=False)

        return law.task.base.Register.__new__(metacls, classname, bases, classdict)


class TestPlots(six.with_metaclass(TestRegister, AnalysisTask)):

    task_namespace = "test"

    # test names that will to translated to task parameters "--no-<name>" and "--only-<name>"
    # by the meta class
    test_names = [
        "upper_limits",
        "multiple_upper_limits",
        "multiple_upper_limits_by_model",
        "upper_limits_at_point",
        "likelihood_scan",
        "likelihood_scan_2d",
        "multiple_likelihood_scans",
        "multiple_likelihood_scans_2d",
        "multiple_likelihood_scans_by_model",
        "multiple_likelihood_scans_by_model_2d",
        "significance_scan",
        "multiple_significance_scans",
        "pulls_and_impacts",
        "exclusion_and_bestfit",
        "exclusion_and_bestfit_2d",
        "postfit_s_over_b",
        "goodness_of_fit",
        "multiple_goodness_of_fits",
        "morphing_scales",
        "morphed_discriminant",
        "stat_error_scan",
    ]

    file_type = PlotTask.file_type
    campaign = PlotTask.campaign
    view_cmd = PlotTask.view_cmd

    exclude_params_req = {"view_cmd"}

    def check_enabled(self, test_name):
        assert(test_name in self.test_names)
        if any(getattr(self, "only_" + n) for n in self.test_names):
            return getattr(self, "only_" + test_name)
        else:
            return not getattr(self, "no_" + test_name)

    def requires(self):
        reqs = OrderedDict()

        ggf_cards = tuple(os.environ["DHI_EXAMPLE_CARDS_GGF"].split(","))
        vbf_cards = tuple(os.environ["DHI_EXAMPLE_CARDS_VBF"].split(","))
        multi_cards = (ggf_cards, vbf_cards, ggf_cards + vbf_cards)
        multi_cards_names = ("ggF", "VBF", "All")
        test_models = tuple("HHModelPinv.model_no_ggf_kl" + kl for kl in "1 2p45 5".split())

        if self.check_enabled("upper_limits"):
            reqs["upper_limits"] = PlotUpperLimits.req(self,
                datacards=ggf_cards,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("multiple_upper_limits"):
            reqs["multiple_upper_limits"] = PlotMultipleUpperLimits.req(self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("multiple_upper_limits_by_model"):
            reqs["multiple_upper_limits_by_model"] = PlotMultipleUpperLimitsByModel.req(self,
                datacards=ggf_cards,
                hh_models=test_models,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("upper_limits_at_point"):
            reqs["upper_limits_at_point"] = PlotUpperLimitsAtPoint.req(self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                parameter_values=("kl=1", "kt=1"),
            )

        if self.check_enabled("likelihood_scan"):
            reqs["likelihood_scan"] = PlotLikelihoodScan.req(self,
                datacards=ggf_cards,
                pois=("kl",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("likelihood_scan_2d"):
            reqs["likelihood_scan_2d"] = PlotLikelihoodScan.req(self,
                datacards=ggf_cards,
                pois=("kl", "kt"),
                scan_parameters=(("kl", -5., 5.), ("kt", -5., 5.),),
            )

        if self.check_enabled("multiple_likelihood_scans"):
            reqs["multiple_likelihood_scans"] = PlotMultipleLikelihoodScans.req(self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("kl",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("multiple_likelihood_scans_2d"):
            reqs["multiple_likelihood_scans_2d"] = PlotMultipleLikelihoodScans.req(self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("kl", "kt"),
                scan_parameters=(("kl", -5., 5.), ("kt", -5., 5.),),
            )

        if self.check_enabled("multiple_likelihood_scans_by_model"):
            reqs["multiple_likelihood_scans_by_model"] = PlotMultipleLikelihoodScansByModel.req(
                self,
                datacards=ggf_cards,
                hh_models=test_models,
                pois=("kl",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("multiple_likelihood_scans_by_model_2d"):
            reqs["multiple_likelihood_scans_by_model_2d"] = PlotMultipleLikelihoodScansByModel.req(
                self,
                datacards=ggf_cards,
                hh_models=test_models,
                pois=("kl", "kt"),
                scan_parameters=(("kl", -5., 5.), ("kt", -5., 5.),),
            )

        if self.check_enabled("significance_scan"):
            reqs["significance_scan"] = PlotSignificanceScan.req(self,
                datacards=ggf_cards,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("multiple_significance_scans"):
            reqs["multiple_significance_scans"] = PlotMultipleSignificanceScans.req(self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                scan_parameters=(("kl", -5., 5.),),
            )

        if self.check_enabled("pulls_and_impacts"):
            reqs["pulls_and_impacts"] = PlotPullsAndImpacts.req(self,
                datacards=ggf_cards,
                pois=("r",),
            )
            reqs["pulls_and_impacts"].requires().requires().end_branch = 10

        if self.check_enabled("exclusion_and_bestfit"):
            reqs["exclusion_and_bestfit"] = PlotExclusionAndBestFit.req(self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                scan_parameters=(("kl", -30., 30., 61),),
            )

        if self.check_enabled("exclusion_and_bestfit_2d"):
            reqs["exclusion_and_bestfit_2d"] = PlotExclusionAndBestFit2D.req(self,
                datacards=ggf_cards,
                pois=("r",),
                scan_parameters=(("kl", -30., 30., 61), ("kt", -6., 9., 31)),
            )

        if self.check_enabled("postfit_s_over_b"):
            reqs["postfit_s_over_b"] = PlotPostfitSOverB.req(self,
                datacards=ggf_cards,
                pois=("r",),
            )

        if self.check_enabled("goodness_of_fit"):
            reqs["goodness_of_fit"] = PlotGoodnessOfFit.req(self,
                datacards=ggf_cards,
                pois=("r",),
                toys=300,
                toys_per_task=15,
            )

        if self.check_enabled("multiple_goodness_of_fits"):
            reqs["multiple_goodness_of_fits"] = PlotMultipleGoodnessOfFits.req(self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                toys=(300,),
                toys_per_task=(15,),
            )

        if self.check_enabled("morphing_scales"):
            reqs["morphing_scales"] = PlotMorphingScales.req(self,
                hh_model="HHModelPinv.model_default",
                signal="ggf",
                scan_parameters=(("kl", -10., 10.),),
                parameter_values=("kt=1",),
            )

        if self.check_enabled("morphed_discriminant"):
            reqs["morphed_discriminant"] = PlotMorphedDiscriminant.req(self,
                datacards=ggf_cards[:1],
                hh_models=test_models,
                signal="ggf",
                bins=("ch1",),
                parameter_values=("kl=1", "kt=1"),
            )

        if self.check_enabled("stat_error_scan"):
            reqs["stat_error_scan"] = PlotStatErrorScan.req(self,
                datacards=ggf_cards[:1],
                hh_models=test_models,
                signal="ggf",
                bins=("ch1",),
                scan_parameters=(("kl", -20, 20, 81),),
                parameter_values=("kt=1",),
            )

        return reqs

    def output(self):
        return self.input()

    @view_output_plots
    def run(self):
        pass
