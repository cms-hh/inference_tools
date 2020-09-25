# coding: utf-8

import re

import law
import luigi

from dhi.tasks.base import AnalysisTask
from dhi.tasks.nlo.mixins import PlotMixin, LabelsMixin, ViewMixin


class CompareScan(PlotMixin, LabelsMixin, ViewMixin, AnalysisTask):
    """
    Example usage:
        law run dhi.CompareScan --version dev1 --limit-directories "/eos/user/u/username/dhi/store/NLOLimit/mjj/125/kl_-40_40,/eos/user/u/username/dhi/store/NLOLimit/classic_dnn/125/kl_-40_40" --limit-labels "mjj,dnn"
    """

    poi = luigi.ChoiceParameter(default="kl", choices=("kl", "C2V"))
    limit_directories = law.CSVParameter(
        description="usage: --limit-directories '/path/to/mjj/limits/directory,/path/to/dnn/limits/directory'"
    )
    limit_labels = law.CSVParameter(description="usage: --limit-labels 'mjj,dnn'", default=())

    def __init__(self, *args, **kwargs):
        super(CompareScan, self).__init__(*args, **kwargs)
        if not self.limit_labels:
            self.limit_labels = list(map(str, range(len(self.limit_directories))))
        assert len(self.limit_directories) == len(self.limit_labels)
        self.inputs = {
            limit_label: law.SiblingFileCollection.from_directory(limit_dir)
            for limit_label, limit_dir in zip(self.limit_labels, self.limit_directories)
        }

    def output(self):
        return self.local_target("scan.pdf")

    @ViewMixin.view_output_plots
    def run(self):
        import numpy as np
        plt = import_plt()

        limits = {}

        for name, collection in self.inputs.items():
            inputs_ = {
                int(re.findall(r"-?\d+", k.basename)[0]): k
                for k in collection.targets
                if k.basename.endswith(".json")
            }

            data = [
                [kl] + [data[i] for i in range(-2, 3)]
                for kl, data in (
                    (key, {int(k[3:]): v for k, v in inp.load()["125.0"].items()})
                    for key, inp in sorted(inputs_.items(), key=lambda x: x)
                )
            ]
            arr = np.array(data)
            limits[name] = arr
        # rescale r to xsec:

        br_hww_hbb = 0.0264
        k_factor = 1.115
        sigma_sm = np.ones_like(arr[:, 0])
        if self.poi == "kl":
            from dhi.models import ggf_formula

            sigma_sm = (
                np.array(
                    [
                        ggf_formula.sigma.evalf(
                            subs={
                                "kl": x,
                                "kt": 1.0,
                                "s1": ggf_formula.sample_list[0].val_xs,
                                "s2": ggf_formula.sample_list[1].val_xs,
                                "s3": ggf_formula.sample_list[2].val_xs,
                            }
                        )[0]
                        for x in arr[:, 0]
                    ]
                ).astype(np.float64)
                * br_hww_hbb
                * k_factor  # NLO -> NNLO
                * 1000.0  # convert pb -> fb
            )
        if self.poi == "C2V":
            from dhi.models import vbf_formula

            sigma_sm = (
                np.array(
                    [
                        vbf_formula.sigma.evalf(
                            subs={
                                "C2V": x,
                                "CV": 1.0,
                                "kl": 1.0,
                                "s1": vbf_formula.sample_list[0].val_xs,
                                "s2": vbf_formula.sample_list[1].val_xs,
                                "s3": vbf_formula.sample_list[2].val_xs,
                                "s4": vbf_formula.sample_list[3].val_xs,
                                "s5": vbf_formula.sample_list[4].val_xs,
                                "s6": vbf_formula.sample_list[5].val_xs,
                            }
                        )[0]
                        for x in arr[:, 0]
                    ]
                ).astype(np.float64)
                * br_hww_hbb
                * k_factor  # NLO -> NNLO
                * 1000.0  # convert pb -> fb
            )
        plt.figure()
        plt.title(r"\textbf{CMS} \textit{Preliminary}", loc="left")
        plt.title(self.top_right_text, loc="right")
        plt.plot(
            arr[:, 0],
            sigma_sm,
            label=r"theoretical $\sigma$",
            linestyle="-",
        )
        for k, arr in limits.items():
            plt.plot(
                arr[:, 0],
                arr[:, 3] * sigma_sm,
                label=r"expected limit - {}".format(k),
                linestyle="-",
            )
        plt.legend(loc="best")
        plt.xlabel(self.poi_label)
        plt.ylabel(r"$95\%$ CL on ${\sigma}$[fb]")
        plt.yscale("log")
        plt.grid()

        self.save_plt(plt)


class CompareNLL1D(PlotMixin, LabelsMixin, ViewMixin, AnalysisTask):
    """
    Example usage:
        law run dhi.CompareNLL1D --version dev1 --nll-files "/eos/user/u/username/dhi/store/NLOScan1D/dev1/125/kl_-40_40/higgsCombineTest.MultiDimFit.mH125.root,/eos/user/u/username/dhi/store/NLOScan1D/classic_dnn/125/kl_-40_40/higgsCombineTest.MultiDimFit.mH125.root,/eos/user/u/username/dhi/store/NLOScan1D/mjj/125/kl_-40_40/higgsCombineTest.MultiDimFit.mH125.root" --nll-labels "multiclass DNN,classic DNN,mjj"
    """

    poi = luigi.ChoiceParameter(default="kl", choices=("kl", "C2V"))
    nll_files = law.CSVParameter(
        description="usage: --nll-files '/path/to/mjj/nll/higgsCombineTest.MultiDimFit.mH125.root,/path/to/dnn/nll/higgsCombineTest.MultiDimFit.mH125.root'"
    )
    nll_labels = law.CSVParameter(description="usage: --nll-labels 'mjj,dnn'", default=())

    def __init__(self, *args, **kwargs):
        super(CompareNLL1D, self).__init__(*args, **kwargs)

        if not self.nll_labels:
            self.nll_labels = list(map(str, range(len(self.nll_files))))
        assert len(self.nll_files) == len(self.nll_labels)

    def output(self):
        return self.local_target("nll.pdf")

    @ViewMixin.view_output_plots
    def run(self):
        import numpy as np
        import uproot
        plt = import_plt()

        nll = {}
        for nll_label, nll_file in zip(self.nll_labels, self.nll_files):
            data = {}
            inp = uproot.open(nll_file)
            tree = inp["limit"]
            # first element is trash
            data["poi"] = np.delete(tree[self.poi].array(), 0)
            data["deltaNLL"] = 2 * np.delete(tree["deltaNLL"].array(), 0)
            nll[nll_label] = data

        plt.figure()
        plt.title(r"\textbf{CMS} \textit{Preliminary}", loc="left")
        plt.title(self.top_right_text, loc="right")

        for label, data in nll.items():
            plt.plot(
                data["poi"],
                data["deltaNLL"],
                linestyle="-",
                marker=".",
                label=r"expected - {}".format(label),
            )
        plt.xlabel(self.poi_label)
        plt.ylabel(r"$-2 \Delta \text{ln}\mathcal{L}$")
        plt.ylim(bottom=0.0)
        plt.legend(loc="best")
        plt.grid()

        self.save_plt(plt)
