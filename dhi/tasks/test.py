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
    PlotUpperLimits,
    PlotMultipleUpperLimits,
    PlotMultipleUpperLimitsByModel,
    PlotUpperLimitsAtPoint,
)
from dhi.tasks.likelihoods import (
    PlotLikelihoodScan,
    PlotMultipleLikelihoodScans,
    PlotMultipleLikelihoodScansByModel,
)
from dhi.tasks.significances import PlotSignificanceScan, PlotMultipleSignificanceScans
from dhi.tasks.pulls_impacts import PlotPullsAndImpacts
from dhi.tasks.exclusion import PlotExclusionAndBestFit, PlotExclusionAndBestFit2D
from dhi.tasks.postfit import PlotPostfitSOverB, PlotNuisanceLikelihoodScans
from dhi.tasks.gof import PlotGoodnessOfFit, PlotMultipleGoodnessOfFits
from dhi.tasks.eft import PlotEFTBenchmarkLimits, PlotMultipleEFTBenchmarkLimits
from dhi.tasks.studies.model_selection import (
    PlotMorphingScales,
    PlotMorphedDiscriminant,
    PlotStatErrorScan,
)


class TestRegister(law.task.base.Register):

    def __new__(metacls, classname, bases, classdict):
        # convert test names into "--no-<name>" and "--only-<name>" task parameters
        for test_name in classdict.get("test_names", []):
            classdict["no_" + test_name] = luigi.BoolParameter(default=False)
            classdict["only_" + test_name] = luigi.BoolParameter(default=False)

        return law.task.base.Register.__new__(metacls, classname, bases, classdict)


class TestPlotsDefs(six.with_metaclass(TestRegister, AnalysisTask)):

    task_namespace = "test"

    # test names that will be translated to task parameters
    # "--no-<name>" and "--only-<name>" by the meta class
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
        "multiple_eft_benchmark_limits",
        "upper_limits_c2",
        "likelihood_scan_c2_2d",
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
    exclude_index = True

    def check_enabled(self, test_name):
        assert test_name in self.test_names

        if any(getattr(self, "only_" + n) for n in self.test_names):
            return getattr(self, "only_" + test_name)

        return not getattr(self, "no_" + test_name)

    def requires(self):
        reqs = OrderedDict()

        # helper
        get_cards = lambda name: tuple(os.environ["DHI_EXAMPLE_{}".format(name)].split(","))

        # prepare variables
        sm_cards = get_cards("CARDS")
        sm_cards_1 = get_cards("CARDS_1")
        # sm_cards_large = get_cards("CARDS_1") + get_cards("CARDS_2")
        sm_cards_multi = (get_cards("CARDS_1"), get_cards("CARDS_2"))
        sm_cards_multi_large = (
            get_cards("CARDS_1") + get_cards("CARDS_2"),
            get_cards("CARDS_2") + get_cards("CARDS_3"),
        )
        sm_cards_multi_names = ("Cards 1", "Cards 2")

        eft_c2_cards = get_cards("CARDS_EFT_C2")
        eft_bm_cards = get_cards("CARDS_EFT_BM")
        eft_bm_cards_multi = (get_cards("CARDS_EFT_BM_1"), get_cards("CARDS_EFT_BM_2"))

        # res_cards = get_cards("CARDS_RES")
        # res_cards_multi = (get_cards("CARDS_RES_1"), get_cards("CARDS_RES_2"))

        test_models = tuple(map("model_default{}".format, ["", "@noBRscaling", "@noHscaling"]))
        c2_model = "hh_model_C2klkt.model_default"

        if self.check_enabled("upper_limits"):
            reqs["upper_limits"] = PlotUpperLimits.req(
                self,
                datacards=sm_cards,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
                unblinded=True,
                xsec="fb",
                y_log=True,
            )

        if self.check_enabled("multiple_upper_limits"):
            reqs["multiple_upper_limits"] = PlotMultipleUpperLimits.req(
                self,
                multi_datacards=sm_cards_multi,
                datacard_names=sm_cards_multi_names,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
                unblinded=(True,),
                xsec="fb",
                y_log=True,
            )

        if self.check_enabled("multiple_upper_limits_by_model"):
            reqs["multiple_upper_limits_by_model"] = PlotMultipleUpperLimitsByModel.req(
                self,
                datacards=sm_cards,
                hh_models=test_models,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
                unblinded=(True,),
                xsec="fb",
                y_log=True,
            )

        if self.check_enabled("upper_limits_at_point"):
            reqs["upper_limits_at_point"] = PlotUpperLimitsAtPoint.req(
                self,
                multi_datacards=sm_cards_multi,
                datacard_names=sm_cards_multi_names,
                pois=("r",),
                parameter_values=(("kl", 1), ("kt", 1)),
                show_parameters=(("kl", "kt", "CV"),),
                unblinded=(True,),
                xsec="fb",
                x_log=True,
            )

        if self.check_enabled("likelihood_scan"):
            reqs["likelihood_scan"] = PlotLikelihoodScan.req(
                self,
                datacards=sm_cards,
                pois=("kl",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
                unblinded=True,
                y_log=True,
            )

        if self.check_enabled("likelihood_scan_2d"):
            reqs["likelihood_scan_2d"] = PlotLikelihoodScan.req(
                self,
                datacards=sm_cards,
                pois=("kl", "kt"),
                scan_parameters=(
                    ("kl", -30.0, 30.0, 7),
                    ("kt", -10.0, 10.0, 6),
                ),
                show_parameters=(("CV",),),
                unblinded=True,
            )

        if self.check_enabled("multiple_likelihood_scans"):
            reqs["multiple_likelihood_scans"] = PlotMultipleLikelihoodScans.req(
                self,
                multi_datacards=sm_cards_multi,
                datacard_names=sm_cards_multi_names,
                pois=("kl",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
                unblinded=(True,),
            )

        if self.check_enabled("multiple_likelihood_scans_2d"):
            reqs["multiple_likelihood_scans_2d"] = PlotMultipleLikelihoodScans.req(
                self,
                multi_datacards=sm_cards_multi_large,
                datacard_names=sm_cards_multi_names,
                pois=("kl", "kt"),
                scan_parameters=(
                    ("kl", -30.0, 30.0, 7),
                    ("kt", -10.0, 10.0, 6),
                ),
                unblinded=(True,),
            )

        if self.check_enabled("multiple_likelihood_scans_by_model"):
            reqs["multiple_likelihood_scans_by_model"] = PlotMultipleLikelihoodScansByModel.req(
                self,
                datacards=sm_cards,
                hh_models=test_models,
                pois=("kl",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
                unblinded=(True,),
            )

        if self.check_enabled("multiple_likelihood_scans_by_model_2d"):
            reqs["multiple_likelihood_scans_by_model_2d"] = PlotMultipleLikelihoodScansByModel.req(
                self,
                datacards=sm_cards,
                hh_models=test_models,
                pois=("kl", "kt"),
                scan_parameters=(
                    ("kl", -30.0, 30.0, 7),
                    ("kt", -10.0, 10.0, 6),
                ),
                unblinded=(True,),
            )

        if self.check_enabled("significance_scan"):
            reqs["significance_scan"] = PlotSignificanceScan.req(
                self,
                datacards=sm_cards,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("multiple_significance_scans"):
            reqs["multiple_significance_scans"] = PlotMultipleSignificanceScans.req(
                self,
                multi_datacards=sm_cards_multi,
                datacard_names=sm_cards_multi_names,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
            )

        if self.check_enabled("pulls_and_impacts"):
            reqs["pulls_and_impacts"] = PlotPullsAndImpacts.req(
                self,
                datacards=sm_cards,
                pois=("r",),
                unblinded=True,
            )

        if self.check_enabled("exclusion_and_bestfit"):
            reqs["exclusion_and_bestfit"] = PlotExclusionAndBestFit.req(
                self,
                multi_datacards=sm_cards_multi,
                datacard_names=sm_cards_multi_names,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 7),),
                show_parameters=(("kt", "CV"),),
                unblinded=(True,),
            )

        if self.check_enabled("exclusion_and_bestfit_2d"):
            reqs["exclusion_and_bestfit_2d"] = PlotExclusionAndBestFit2D.req(
                self,
                datacards=sm_cards,
                pois=("r",),
                scan_parameters=(("kl", -30.0, 30.0, 7), ("kt", -10.0, 10.0, 6)),
                show_parameters=(("CV",),),
                unblinded=True,
            )

        if self.check_enabled("postfit_s_over_b"):
            reqs["postfit_s_over_b"] = PlotPostfitSOverB.req(
                self,
                datacards=sm_cards,
                pois=("r",),
                show_parameters=(("kl", "kt"),),
            )

        if self.check_enabled("nuisance_likelihood_scans"):
            reqs["nuisance_likelihood_scans"] = PlotNuisanceLikelihoodScans.req(
                self,
                datacards=sm_cards,
                pois=("r",),
                show_parameters=(("kl", "kt"),),
                parameters_per_page=6,
                y_log=True,
            )

        if self.check_enabled("goodness_of_fit"):
            reqs["goodness_of_fit"] = PlotGoodnessOfFit.req(
                self,
                datacards=sm_cards,
                pois=("r",),
                toys=100,
                toys_per_branch=10,
                algorithm="saturated",
                frequentist_toys=True,
                show_parameters=(("kl", "kt"),),
            )

        if self.check_enabled("multiple_goodness_of_fits"):
            reqs["multiple_goodness_of_fits"] = PlotMultipleGoodnessOfFits.req(
                self,
                multi_datacards=sm_cards_multi,
                datacard_names=sm_cards_multi_names,
                pois=("r",),
                toys=(100,),
                toys_per_branch=(10,),
                algorithm="saturated",
                frequentist_toys=True,
                show_parameters=(("kl", "kt"),),
            )

        if self.check_enabled("eft_benchmark_limits"):
            reqs["eft_benchmark_limits"] = PlotEFTBenchmarkLimits.req(
                self,
                multi_datacards=(eft_bm_cards,),
                unblinded=True,
                xsec="fb",
                y_log=True,
            )

        if self.check_enabled("multiple_eft_benchmark_limits"):
            reqs["multiple_eft_benchmark_limits"] = PlotMultipleEFTBenchmarkLimits.req(
                self,
                multi_datacards=eft_bm_cards_multi,
                unblinded=True,
                xsec="fb",
                y_log=True,
            )

        if self.check_enabled("upper_limits_c2"):
            reqs["upper_limits_c2"] = PlotUpperLimits.req(
                self,
                datacards=eft_c2_cards,
                hh_model=c2_model,
                pois=("r",),
                scan_parameters=(("C2", -4.0, 4.0, 9),),
                show_parameters=(("kl", "kt", "CV"),),
                unblinded=True,
                xsec="fb",
                y_log=True,
            )

        if self.check_enabled("likelihood_scan_c2_2d"):
            reqs["likelihood_scan_c2_2d"] = PlotLikelihoodScan.req(
                self,
                datacards=sm_cards,
                hh_model=c2_model,
                pois=("kl", "C2"),
                scan_parameters=(
                    ("kl", -30.0, 30.0, 7),
                    ("C2", -4.0, 4.0, 9),
                ),
                show_parameters=(("CV", "kt"),),
                unblinded=True,
            )

        if self.check_enabled("morphing_scales"):
            reqs["morphing_scales"] = PlotMorphingScales.req(
                self,
                signal="ggf",
                scan_parameters=(("kl", -10.0, 10.0),),
                parameter_values=(("kt", 1),),
            )

        if self.check_enabled("morphed_discriminant"):
            reqs["morphed_discriminant"] = PlotMorphedDiscriminant.req(
                self,
                datacards=sm_cards_1,
                hh_models=test_models,
                signal="ggf",
                bins=("bin_1",),
                parameter_values=(("kl", 1), ("kt", 1)),
            )

        if self.check_enabled("stat_error_scan"):
            reqs["stat_error_scan"] = PlotStatErrorScan.req(
                self,
                datacards=sm_cards_1,
                hh_models=test_models,
                signal="ggf",
                bins=("bin_1",),
                scan_parameters=(("kl", 0, 10, 41),),
                parameter_values=(("kt", 1),),
            )

        return reqs

    def output(self):
        return self.input()

    @view_output_plots
    def run(self):
        pass


class TestPlots(AnalysisTask):

    task_namespace = "test"

    # plot group names
    plot_groups = {
        "limits": [
            "upper_limits",
            "multiple_upper_limits",
            "multiple_upper_limits_by_model",
            "upper_limits_at_point",
        ],
        "likelihoods": [
            "likelihood_scan",
            "likelihood_scan_2d",
            "multiple_likelihood_scans",
            "multiple_likelihood_scans_2d",
            "multiple_likelihood_scans_by_model",
            "multiple_likelihood_scans_by_model_2d",
        ],
        "significances": [
            "significance_scan",
            "multiple_significance_scans",
        ],
        "pulls": [
            "pulls_and_impacts",
        ],
        "exclusions": [
            "exclusion_and_bestfit",
            "exclusion_and_bestfit_2d",
        ],
        "postfit": [
            "postfit_s_over_b",
            "nuisance_likelihood_scans",
        ],
        "gof": [
            "goodness_of_fit",
            "multiple_goodness_of_fits",
        ],
        "eft_bm": [
            "eft_benchmark_limits",
            "multiple_eft_benchmark_limits",
        ],
        "eft_c2": [
            "upper_limits_c2",
            "likelihood_scan_c2_2d",
        ],
        "studies": [
            "morphing_scales",
            "morphed_discriminant",
            "stat_error_scan",
        ],
    }

    plots = law.CSVParameter(
        default=("limits",),
        description="comma-separated list of test to run; default: 'limits'",
    )

    file_types = TestPlotsDefs.file_types
    campaign = TestPlotsDefs.campaign
    cms_postfix = TestPlotsDefs.cms_postfix
    style = TestPlotsDefs.style
    view_cmd = TestPlotsDefs.view_cmd

    exclude_params_req = {"view_cmd"}

    def requires(self):
        # determine the full list of plots to enable
        plots = []

        for name in self.plots:
            if name == "all":
                plots += sum(self.plot_groups.values(), [])
                break

            if name in self.plot_groups:
                plots += self.plot_groups[name]
                continue

            for group_name, plot_names in self.plot_groups.items():
                if name in plot_names:
                    plots.append(name)
                    break

        # remove duplicates
        plots = sorted(set(plots), key=plots.index)

        # define requirements
        return TestPlotsDefs.req(
            self,
            **{"only_{}".format(name): True for name in plots}  # noqa
        )

    def output(self):
        return self.input()

    @view_output_plots
    def run(self):
        pass
