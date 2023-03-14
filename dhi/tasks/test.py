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
    PlotUpperLimits, PlotMultipleUpperLimits, PlotMultipleUpperLimitsByModel,
    PlotUpperLimitsAtPoint,
)
from dhi.tasks.likelihoods import (
    PlotLikelihoodScan, PlotMultipleLikelihoodScans, PlotMultipleLikelihoodScansByModel,
)
from dhi.tasks.significances import PlotSignificanceScan, PlotMultipleSignificanceScans
from dhi.tasks.pulls_impacts import PlotPullsAndImpacts
from dhi.tasks.exclusion import PlotExclusionAndBestFit, PlotExclusionAndBestFit2D
from dhi.tasks.postfit import PlotPostfitSOverB, PlotNuisanceLikelihoodScans
from dhi.tasks.gof import PlotGoodnessOfFit, PlotMultipleGoodnessOfFits
from dhi.tasks.eft import PlotEFTBenchmarkLimits
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
        "nuisance_likelihood_scans",
        "goodness_of_fit",
        "eft_benchmark_limits",
        "multiple_goodness_of_fits",
        "morphing_scales",
        "morphed_discriminant",
        "stat_error_scan",
    ]

    file_types = PlotTask.file_types
    campaign = PlotTask.campaign
    cms_postfix = PlotTask.cms_postfix
    style = PlotTask.style
    view_cmd = PlotTask.view_cmd

    exclude_params_req = {"view_cmd"}

    def check_enabled(self, test_name):
        assert test_name in self.test_names
        if any(getattr(self, "only_" + n) for n in self.test_names):
            return getattr(self, "only_" + test_name)
        else:
            return not getattr(self, "no_" + test_name)

    def requires(self):
        reqs = OrderedDict()

        cards = tuple(os.environ["DHI_EXAMPLE_CARDS"].split(","))
        multi_cards = tuple((c,) for c in cards) + (cards,)
        multi_cards_names = tuple(map("Cards {}".format, range(1, len(cards) + 1))) + ("All",)
        test_models = tuple(map("model_default{}".format, ["", "@noBRscaling", "@noHscaling"]))
        eft_cards_bm = tuple(os.environ["DHI_EXAMPLE_CARDS_EFT_BM"].split(","))

        if self.check_enabled("upper_limits"):
            reqs["upper_limits"] = PlotUpperLimits.req(
                self,
                datacards=cards,
                pois=("r",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("multiple_upper_limits"):
            reqs["multiple_upper_limits"] = PlotMultipleUpperLimits.req(
                self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("multiple_upper_limits_by_model"):
            reqs["multiple_upper_limits_by_model"] = PlotMultipleUpperLimitsByModel.req(
                self,
                datacards=cards,
                hh_models=test_models,
                pois=("r",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("upper_limits_at_point"):
            reqs["upper_limits_at_point"] = PlotUpperLimitsAtPoint.req(
                self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                parameter_values=(("kl", 1), ("kt", 1)),
                show_parameters=(("kl", "kt", "CV"),),
            )

        if self.check_enabled("likelihood_scan"):
            reqs["likelihood_scan"] = PlotLikelihoodScan.req(
                self,
                datacards=cards,
                pois=("kl",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("likelihood_scan_2d"):
            reqs["likelihood_scan_2d"] = PlotLikelihoodScan.req(
                self,
                datacards=cards,
                pois=("kl", "kt"),
                scan_parameters=(
                    ("kl", -5.0, 5.0),
                    ("kt", -5.0, 5.0),
                ),
            )

        if self.check_enabled("multiple_likelihood_scans"):
            reqs["multiple_likelihood_scans"] = PlotMultipleLikelihoodScans.req(
                self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("kl",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("multiple_likelihood_scans_2d"):
            reqs["multiple_likelihood_scans_2d"] = PlotMultipleLikelihoodScans.req(
                self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("kl", "kt"),
                scan_parameters=(
                    ("kl", -5.0, 5.0),
                    ("kt", -5.0, 5.0),
                ),
            )

        if self.check_enabled("multiple_likelihood_scans_by_model"):
            reqs["multiple_likelihood_scans_by_model"] = PlotMultipleLikelihoodScansByModel.req(
                self,
                datacards=cards,
                hh_models=test_models,
                pois=("kl",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("multiple_likelihood_scans_by_model_2d"):
            reqs["multiple_likelihood_scans_by_model_2d"] = PlotMultipleLikelihoodScansByModel.req(
                self,
                datacards=cards,
                hh_models=test_models,
                pois=("kl", "kt"),
                scan_parameters=(
                    ("kl", -5.0, 5.0),
                    ("kt", -5.0, 5.0),
                ),
            )

        if self.check_enabled("significance_scan"):
            reqs["significance_scan"] = PlotSignificanceScan.req(
                self,
                datacards=cards,
                pois=("r",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("multiple_significance_scans"):
            reqs["multiple_significance_scans"] = PlotMultipleSignificanceScans.req(
                self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                scan_parameters=(("kl", -5.0, 5.0),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("pulls_and_impacts"):
            reqs["pulls_and_impacts"] = PlotPullsAndImpacts.req(
                self,
                datacards=cards,
                pois=("r",),
            )
            reqs["pulls_and_impacts"].requires().requires().end_branch = 10

        if self.check_enabled("exclusion_and_bestfit"):
            reqs["exclusion_and_bestfit"] = PlotExclusionAndBestFit.req(
                self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 61),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("exclusion_and_bestfit_2d"):
            reqs["exclusion_and_bestfit_2d"] = PlotExclusionAndBestFit2D.req(
                self,
                datacards=cards,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 61), ("kt", -6.0, 9.0, 31)),
                show_parameters=(("CV",),),
            )

        if self.check_enabled("postfit_s_over_b"):
            reqs["postfit_s_over_b"] = PlotPostfitSOverB.req(
                self,
                datacards=cards,
                pois=("r",),
                show_parameters=(("kl", "kt"),),
            )

        if self.check_enabled("nuisance_likelihood_scans"):
            reqs["nuisance_likelihood_scans"] = PlotNuisanceLikelihoodScans.req(
                self,
                datacards=cards,
                pois=("r",),
                parameters_per_page=6,
                y_log=True,
                show_parameters=(("kl", "kt"),),
            )

        if self.check_enabled("goodness_of_fit"):
            reqs["goodness_of_fit"] = PlotGoodnessOfFit.req(
                self,
                datacards=cards,
                pois=("r",),
                toys=300,
                toys_per_task=15,
                algorithm="saturated",
                frequentist_toys=True,
                show_parameters=(("kl", "kt"),),
            )

        if self.check_enabled("multiple_goodness_of_fits"):
            reqs["multiple_goodness_of_fits"] = PlotMultipleGoodnessOfFits.req(
                self,
                multi_datacards=multi_cards,
                datacard_names=multi_cards_names,
                pois=("r",),
                toys=(300,),
                toys_per_task=(15,),
                algorithm="saturated",
                frequentist_toys=True,
                show_parameters=(("kl", "kt"),),
            )

        if self.check_enabled("eft_benchmark_limits"):
            reqs["eft_benchmark_limits"] = PlotEFTBenchmarkLimits.req(
                self,
                multi_datacards=(eft_cards_bm,),
                xsec="fb",
            )

        if self.check_enabled("morphing_scales"):
            reqs["morphing_scales"] = PlotMorphingScales.req(
                self,
                hh_model="HHModelPinv.model_default",
                signal="ggf",
                scan_parameters=(("kl", -10.0, 10.0),),
                parameter_values=(("kt", 1),),
            )

        if self.check_enabled("morphed_discriminant"):
            reqs["morphed_discriminant"] = PlotMorphedDiscriminant.req(
                self,
                datacards=cards[:1],
                hh_models=test_models,
                signal="ggf",
                bins=("ch1",),
                parameter_values=(("kl", 1), ("kt", 1)),
            )

        if self.check_enabled("stat_error_scan"):
            reqs["stat_error_scan"] = PlotStatErrorScan.req(
                self,
                datacards=cards[:1],
                hh_models=test_models,
                signal="ggf",
                bins=("ch1",),
                scan_parameters=(("kl", -20, 20, 81),),
                parameter_values=(("kt", 1),),
            )

        return reqs

    def output(self):
        return self.input()

    @view_output_plots
    def run(self):
        pass
