# coding: utf-8

from math import ceil, log

import law
import luigi
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt

from dhi.util import rgb


class LabelsMixin(object):

    poi_labels = {
        "kl": r"$\kappa_\lambda$",
        "kt": r"$\kappa_t$",
        "C2V": r"$C_{2V}$",
        "CV": r"$C_{V}$",
    }

    @property
    def poi_label(self):
        return self.poi_labels[self.poi]

    @property
    def poi1_label(self):
        return self.poi_labels[self.poi1]

    @property
    def poi2_label(self):
        return self.poi_labels[self.poi2]


class PlotMixin(object):

    top_right_text = luigi.Parameter(default="2017 (13 TeV)", significant=False)

    def save_plt(self, plt, *args, **kwargs):
        plt.tight_layout()
        self.output().dump(plt, formatter="mpl")
        plt.close()

    def plot(self):
        raise NotImplementedError


class ScanMixin(PlotMixin):

    def plot(self, arr):
        # rescale r to xsec:

        br_hww_hbb = 0.0264
        k_factor = 1.115
        sigma_sm = np.ones_like(arr[:, 0])
        if self.poi == "kl":
            from dhi.utils.models import ggf_formula

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
            from dhi.utils.models import vbf_formula

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
            color="red",
            linestyle="-",
        )
        plt.plot(
            arr[:, 0],
            arr[:, 3] * sigma_sm,
            label="expected limit",
            color="black",
            linestyle="dashed",
        )
        plt.fill_between(
            arr[:, 0],
            arr[:, 5] * sigma_sm,
            arr[:, 1] * sigma_sm,
            color="yellow",
            label=r"2$\sigma$ expected",
            interpolate=True,
        )
        plt.fill_between(
            arr[:, 0],
            arr[:, 4] * sigma_sm,
            arr[:, 2] * sigma_sm,
            color="limegreen",
            label=r"1$\sigma$ expected",
            interpolate=True,
        )
        plt.legend(loc="best")
        plt.xlabel(self.poi_label)
        plt.ylabel(r"$95\%$ CL on ${\sigma}$[fb]")
        plt.yscale("log")
        plt.grid()

        self.save_plt(plt)


class NLL1DMixin(PlotMixin):

    def plot(self, poi, deltaNLL):
        import numpy as np
        from scipy.interpolate import interp1d

        interpol = interp1d(poi, deltaNLL)
        precision = poi.size * 100
        fine_poi = np.linspace(poi.min(), poi.max(), precision)
        fine_deltaNLL = interpol(fine_poi)
        # 1 sigma interval:
        idx1 = np.argwhere(
            np.diff(np.sign(fine_deltaNLL - np.full(fine_deltaNLL.shape, 1)))
        ).flatten()
        # 2 sigma interval:
        idx2 = np.argwhere(
            np.diff(np.sign(fine_deltaNLL - np.full(fine_deltaNLL.shape, 4)))
        ).flatten()
        best_fit_value = fine_poi[np.argmin(fine_deltaNLL)]
        plt.figure()
        plt.title(r"\textbf{CMS} \textit{Preliminary}", loc="left")
        plt.title(self.top_right_text, loc="right")
        for idx in idx1:
            plt.plot(
                (fine_poi[idx], fine_poi[idx]),
                (0.0, fine_deltaNLL[idx]),
                linestyle="--",
                color=rgb(161, 16, 53),
            )
        for idx in idx2:
            plt.plot(
                (fine_poi[idx], fine_poi[idx]),
                (0.0, fine_deltaNLL[idx]),
                linestyle="--",
                color=rgb(161, 16, 53),
            )
        plt.plot(
            poi,
            deltaNLL,
            linestyle="-",
            marker=".",
            color=rgb(0, 84, 159),
            label=r"expected (best %s=%.2f)" % (self.poi_label, best_fit_value),
        )
        plt.xlabel(self.poi_label)
        plt.ylabel(r"$-2 \Delta \text{ln}\mathcal{L}$")
        plt.ylim(bottom=0.0)
        plt.legend(loc="best")
        plt.grid()

        self.publish_message("best fit value for {}: {:.2f}".format(self.poi, best_fit_value))
        self.publish_message("1 sigma uncertainties: {}".format(fine_poi[idx1]))
        self.publish_message("2 sigma uncertainties: {}".format(fine_poi[idx2]))

        self.save_plt(plt)


class NLL2DMixin(PlotMixin):

    def plot(self, poi1, poi2, deltaNLL):
        e1 = np.unique(poi1)
        e2 = np.unique(poi2)
        i1 = np.searchsorted(e1, poi1, side="right") - 1
        i2 = np.searchsorted(e2, poi2, side="right") - 1
        d = np.full(e1.shape + e2.shape, np.nan)
        d[i1, i2] = deltaNLL
        plt.figure()
        plt.title(r"\textbf{CMS} \textit{Preliminary}", loc="left")
        plt.title(self.top_right_text, loc="right")
        plt.imshow(
            d,
            norm=matplotlib.colors.LogNorm(),
            aspect="auto",
            origin="lower",
            extent=[e1.min(), e1.max(), e2.min(), e2.max()],
            cmap="viridis",
            interpolation="nearest",
        )
        cmin, cmax = ceil(log(deltaNLL.min(), 10)), ceil(log(deltaNLL.max(), 10))
        plt.colorbar(
            ticks=np.logspace(cmin, cmax, cmax - cmin + 1),
            label=r"$-2 \Delta \text{ln}\mathcal{L}$",
        )
        contours = plt.contour(e1, e2, d, levels=[1.0, 4.0])
        fmt = {l: s for l, s in zip(contours.levels, [r"$1\sigma$", r"$2\sigma$"])}
        plt.clabel(contours, inline=True, fontsize=8, fmt=fmt)
        plt.xlabel(self.poi1_label)
        plt.ylabel(self.poi2_label)
        self.save_plt(plt)


class ViewMixin(object):

    view_cmd = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a command to execute after the task has run to visualize plots in the "
        "terminal, default: empty",
    )

    @staticmethod
    @law.decorator.factory(accept_generator=True)
    def view_output_plots(fn, opts, task, *args, **kwargs):
        def before_call():
            return None

        def call(state):
            return fn(task, *args, **kwargs)

        def after_call(state):
            view_cmd = getattr(task, "view_cmd", None)
            if not view_cmd:
                return

            view_paths = []
            for output in law.util.flatten(task.output()):
                if not getattr(output, "path", None):
                    continue
                if output.path.endswith((".pdf", ".png")):
                    view_paths.append(output.path)

            if "{}" not in view_cmd:
                view_cmd += " {}"
            for path in view_paths:
                task.publish_message("showing {}".format(path))
                law.util.interruptable_popen(
                    view_cmd.format(path), shell=True, executable="/bin/bash"
                )

        return before_call, call, after_call
