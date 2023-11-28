# coding: utf-8

"""
Tasks related to plotting quantities of the physics model.
"""

import copy
import inspect

import luigi
import law

from dhi.tasks.base import PlotTask, view_output_plots
from dhi.tasks.combine import HHModelTask, ParameterValuesTask
from dhi.config import poi_data, colors, cms_postfix
from dhi.util import import_ROOT, linspace, create_tgraph, to_root_latex
from dhi.plots.util import get_y_range, create_model_parameters


colors = colors.root


class PlotSignalEnhancement(HHModelTask, ParameterValuesTask, PlotTask):
    """
    Task that plots the signal enhancement as a function of kappa parameters.
    The HH process (inclusive, ggF, VBF or VHH) can be configured.
    """

    task_namespace = "study"

    DEFAULT_HH_MODULE = "hh_model"
    DEFAULT_HH_MODEL = "model_default_vhh"

    hh_model = copy.copy(HHModelTask.hh_model)
    hh_model._default = "{}.{}".format(DEFAULT_HH_MODULE, DEFAULT_HH_MODEL)
    signal = luigi.ChoiceParameter(
        default="hh",
        choices=["hh", "ggf", "vbf", "vhh"],
        description="the signal mode to plot; should be one of 'hh', 'ggf', 'vbf' or 'vhh'; "
        "default: hh",
    )
    kappas = law.CSVParameter(
        default=("kl",),
        choices=["kl", "kt", "CV", "C2V"],
        description="the kappas to scan; should be any of 'kl', 'kt', 'CV', 'C2V'; default: (kl,)",
    )
    x_min = luigi.FloatParameter(
        default=-1.0,
        description="the lower x-axis limit; default: -1.0",
    )
    x_max = luigi.FloatParameter(
        default=3.0,
        description="the upper x-axis limit; default: 3.0",
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    version = None
    save_hep_data = None

    # signal labels
    s_labels = {
        "hh": "HH",
        "ggf": "ggF",
        "vbf": "VBF",
        "vhh": "VHH",
    }

    # kappa colors
    k_colors = {
        "kl": colors.red_cream,
        "kt": colors.green,
        "CV": colors.purple,
        "C2V": colors.blue_cream,
    }

    def output(self):
        parts = ["signalenhancement", self.signal, self.kappas, [self.x_min, self.x_max]]
        if self.y_log:
            parts.append("log")

        return [self.local_target(name) for name in self.create_plot_names(parts)]

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import plotlib.root as r

        # get the correct cross section function
        module, model = self.load_hh_model()
        if self.signal == "hh":
            get_xsec = model.create_hh_xsec_func()
        else:
            formulae = model.get_formulae()
            name = self.signal + "_formula"
            if name not in formulae:
                raise Exception(
                    "no formula for signal {} in model {}".format(self.signal, model.name),
                )
            get_xsec = getattr(module, "create_{}_xsec_func".format(self.signal))(formulae[name])
        allowed_args = inspect.getargspec(get_xsec).args

        # determine the fixed kappas and their values, per scanned kappa
        fixed_kappas = {
            k: {
                _k: self.parameter_values_dict.get(_k, poi_data[_k].sm_value)
                for _k in self.__class__.kappas._choices
                if _k != k
            }
            for k in self.kappas
        }

        # helper to return a dict containing the fixed kappas and a dynamic one that is scanned
        def get_xsec_kwargs(k, v):
            return {
                _k: _v for _k, _v in [(k, v)] + list(fixed_kappas[k].items())
                if _k in allowed_args
            }

        # helper to compute and return the actual signal enhancement
        def get_enhancement(k, x):
            xsec = get_xsec(**get_xsec_kwargs(k, x))
            xsec_sm = get_xsec(**get_xsec_kwargs(k, poi_data[k].sm_value))
            return (x, xsec / xsec_sm)

        # perform the scan using 200 points for each kappa
        scans = [
            [get_enhancement(k, x) for x in linspace(self.x_min, self.x_max, 200)]
            for k in self.kappas
        ]

        # create graphs, labels, etc
        graphs = []
        labels = []
        for k, scan in zip(self.kappas, scans):
            x = [p[0] for p in scan]
            y = [p[1] for p in scan]
            graphs.append(create_tgraph(len(x), x, y))
            labels.append(to_root_latex(poi_data[k].label))

        # start plotting
        ROOT = import_ROOT()
        r.setup_style()
        canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": self.y_log})
        pad.cd()
        draw_objs = []
        legend_entries = []

        # prepare ranges
        y_min_value = min([0.] + [ROOT.TMath.MinElement(g.GetN(), g.GetY()) for g in graphs])
        y_max_value = max(ROOT.TMath.MaxElement(g.GetN(), g.GetY()) for g in graphs)
        y_min, y_max, _ = get_y_range(
            y_min_value,
            y_max_value,
            self.get_axis_limit("y_min"),
            self.get_axis_limit("y_max"),
            log=self.y_log,
            y_min_log=1e-1,
        )

        # dummy histogram to control axes
        x_title = "#kappa"
        y_title = "Signal enhancement #sigma^{{{0}}} / #sigma^{{{0}}}_{{SM}}".format(
            self.s_labels[self.signal],
        )
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, self.x_min, self.x_max)
        r.setup_hist(h_dummy, pad=pad, props={"Minimum": y_min, "Maximum": y_max, "LineWidth": 0})
        draw_objs.append((h_dummy, "HIST"))

        # dashed line at 1
        if y_min < 1. <= y_max:
            line_sm = ROOT.TLine(self.x_min, 1, self.x_max, 1)
            r.setup_line(line_sm, props={"NDC": False, "LineStyle": 2}, color=12)
            draw_objs.append(line_sm)

        # write graphs
        for k, graph, label in zip(self.kappas, graphs, labels):
            r.setup_graph(graph, props={"LineWidth": 2, "MarkerSize": 0}, color=self.k_colors[k])
            draw_objs.append((graph, "SAME,L"))
            legend_entries.append((graph, label))

        # legend
        legend = r.routines.create_legend(pad=pad, width=160, n=len(legend_entries))
        r.fill_legend(legend, legend_entries)
        draw_objs.append(legend)
        legend_box = r.routines.create_legend_box(
            legend,
            pad,
            "tr",
            props={"LineWidth": 0, "FillColor": colors.white_trans_70},
        )
        draw_objs.insert(-1, legend_box)

        # cms label
        cms_labels = r.routines.create_cms_labels(
            pad=pad,
            postfix=cms_postfix,
            layout="outside_horizontal",
        )
        draw_objs.extend(cms_labels)

        # model parameters
        if self.parameter_values_dict:
            draw_objs.extend(create_model_parameters(self.parameter_values_dict, pad))

        # draw all objects
        r.routines.draw_objects(draw_objs)

        # save
        r.update_canvas(canvas)
        outputs = self.output()
        outputs[0].parent.touch()
        for outp in outputs:
            canvas.SaveAs(outp.path)
