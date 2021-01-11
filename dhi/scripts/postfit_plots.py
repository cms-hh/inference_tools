#!/usr/bin/env python
# coding: utf-8

"""
Script to extract and plot shapes from a ROOT file create by combine's FitDiagnostics.
"""

import os

from dhi.util import import_ROOT


def create_postfit_plots(path, fit_diagnostics_path, bin_name):
    ROOT = import_ROOT()

    canvas = ROOT.TCanvas()

    canvas.SaveAs(path)



if __name__ == "__main__":
    # test
    data_dir = "/eos/user/m/mrieger/dhi/store/PostFitShapes/HHModelPinv__model_default/datacards_efa2f8de51/m125.0/poi_r/dev"

    create_postfit_plots(
        path="plot_ch1.pdf",
        fit_diagnostics_path=data_dir + "/fitdiagnostics__poi_r__params_r_qqhh1.0_r_gghh1.0_kl1.0_kt1.0_CV1.0_C2V1.0.root",
        bin_name="ch1",
    )
