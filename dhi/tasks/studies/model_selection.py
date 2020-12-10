# coding: utf-8

"""
Tasks related to tests of the physics model(s).
"""

import math
from collections import OrderedDict, defaultdict

import luigi
import law

from dhi.tasks.combine import DatacardTask, MultiHHModelTask
from dhi.tasks.inference import CombineDatacards
from dhi.tasks.plotting import view_output_plots, PlotTask
from dhi.datacard_tools import create_datacard_instance
from dhi.config import poi_data
from dhi.util import import_ROOT, create_tgraph, to_root_latex


class PlotMorphedDiscriminant(PlotTask, DatacardTask, MultiHHModelTask):

    task_namespace = "study"

    signal = luigi.ChoiceParameter(
        choices=["ggf", "vbf"],
        description="the signal process to morph; choices: ggf,vbf",
    )
    pois = law.CSVParameter(
        description="the parameters for which the morphing should be done; format e.g. "
        "kl=2.0,kt=1.0",
    )
    bins = law.CSVParameter(
        description="comma-separated list of bins to plot; supports brace expansion",
        brace_expand=True,
    )
    y_log = luigi.BoolParameter(
        default=False,
        significant=False,
        description="apply log scaling to the y-axis; default: False",
    )
    x_min = None
    x_max = None
    y_min = None
    y_max = None

    def __init__(self, *args, **kwargs):
        super(PlotMorphedDiscriminant, self).__init__(*args, **kwargs)

        # store the parsed pois
        self.poi_values = {}
        for p in self.pois:
            name, value = p.split("=", 1)
            self.poi_values[name] = float(value)
        # write back in consistent format
        self.pois = tuple("{}={}".format(*tpl) for tpl in self.poi_values.items())

    def requires(self):
        return [
            CombineDatacards.req(self, hh_model=hh_model)
            for hh_model in self.hh_models
        ]

    def output(self):
        parts = [self.signal] + [p.replace("=", "") for p in self.pois]
        if self.y_log:
            parts.append("log")

        return law.SiblingFileCollection({
            b: self.local_target(self.create_plot_name("morpheddiscr", b, *parts))
            for b in self.bins
        })

    def prepare_morphing_data(self):
        # prepare the config, i.e. bin_name -> model_name -> [(hist, scale_fn), ...]
        data = defaultdict(OrderedDict)
        for hh_model, inp in zip(self.hh_models, self.input()):
            # load the datacard and create a shape builder
            dc, sb = create_datacard_instance(inp.path, create_shape_builder=True)

            # get the model
            model = self._load_hh_model(hh_model)[1]

            # get the model formula
            formula = getattr(model, "{}_formula".format(self.signal))

            # store signal names in order of model samples
            signal_names = []
            for sample in formula.sample_list:
                for signal_name in dc.signals:
                    if signal_name.startswith(sample.label):
                        signal_names.append(signal_name)
                        break
                else:
                    raise Exception("no signal found for sample {} in model {}".format(
                        sample.label, model.name))

            # store shapes
            for bin_name in self.bins:
                for signal_name, scale_fn in zip(signal_names, formula.coeffs):
                    # get the shape histogram
                    h = sb.getShape(bin_name, signal_name)
                    h = h.Clone()
                    h.Sumw2()

                    # store it alongside the scaling function
                    data[bin_name].setdefault(model.name, []).append((h, scale_fn))

        return data

    def create_morphed_shape(self, hists_and_scale_fns, **poi_values):
        # merge additional poi values
        poi_values = law.util.merge_dicts(self.poi_values, poi_values)

        # ensure that C* values are upper case
        for k, v in list(poi_values.items()):
            if k.startswith("c"):
                del poi_values[k]
                poi_values[k.upper()] = v

        morphed_shape = None
        for h, scale_fn in hists_and_scale_fns:
            h = h.Clone()

            # check if all symbols are substituted
            missing_symbols = set(map(str, scale_fn.free_symbols)) - set(poi_values.keys())
            if missing_symbols:
                raise Exception("scaling function misses substitutions for symbol(s) '{}', please "
                    "add them via --pois".format(",".join(missing_symbols)))

            # perform the scaling
            scale = scale_fn.evalf(subs=poi_values)
            h.Scale(scale)

            if morphed_shape is None:
                morphed_shape = h
            else:
                morphed_shape.Add(h)

        return morphed_shape

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        # prepare the output
        output = self.output()
        output.dir.touch()

        # create the morphing data
        morphing_data = self.prepare_morphing_data()

        # create plots per bin
        for bin_name, data in morphing_data.items():
            # prepare morphed shapes
            shapes = OrderedDict()
            for model_name, hists_and_scale_fns in data.items():
                shapes[model_name] = self.create_morphed_shape(hists_and_scale_fns)

            # plot
            self.create_plot(output.targets[bin_name], bin_name, shapes)

    def create_plot(self, output, bin_name, shapes):
        import plotlib.root as r
        ROOT = import_ROOT()

        # prepare histograms and ranges
        labels = [m.replace("_", " ").split("model ", 1)[-1] for m in shapes]
        hists = list(shapes.values())
        x_min = min(h.GetXaxis().GetXmin() for h in hists)
        x_max = max(h.GetXaxis().GetXmax() for h in hists)
        y_min = 1e-3 if self.y_log else 0
        y_max_value = max(h.GetMaximum() for h in hists)
        if self.y_log:
            y_max = y_min * 10**(1.35 * math.log10(y_max_value / y_min))
        else:
            y_max = 1.35 * y_max_value

        # start plotting
        r.setup_style(props={"Palette": ROOT.kRainBow})
        canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": self.y_log})
        pad.cd()
        draw_objs = []
        legend_entries = []

        # dummy histogram to control axes
        x_title = "{} discriminant ({})".format(self.signal, ", ".join(self.pois))
        y_title = "Predicted signal yield"
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
        r.setup_hist(h_dummy, pad=pad, props={"Minimum": y_min, "Maximum": y_max, "LineWidth": 0})
        draw_objs.append((h_dummy, "HIST"))

        # write histograms
        for hist, label in zip(hists, labels):
            r.setup_hist(hist, props={"LineWidth": 1})
            draw_objs.append((hist, "SAME,HIST,E,PLC,PMC"))
            legend_entries.append((hist, label))

        # legend
        legend_cols = min(int(math.ceil(len(legend_entries) / 4.)), 3)
        legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
        legend = r.routines.create_legend(pad=pad, width=legend_cols * 210, height=legend_rows * 30,
            props={"NColumns": legend_cols, "FillStyle": 1001})
        r.fill_legend(legend, legend_entries)
        draw_objs.append(legend)

        # cms label
        cms_labels = r.routines.create_cms_labels(pad=pad)
        draw_objs.extend(cms_labels)

        # bin label
        bin_label = r.routines.create_top_right_label(bin_name, pad=pad)
        draw_objs.append(bin_label)

        # draw all objects
        r.routines.draw_objects(draw_objs)

        # save
        r.update_canvas(canvas)
        canvas.SaveAs(output.path)


class PlotStatErrorScan(PlotMorphedDiscriminant):

    y_log = None

    def __init__(self, *args, **kwargs):
        super(PlotStatErrorScan, self).__init__(*args, **kwargs)

        # map the signal to the main poi
        self.poi = {"ggf": "kl", "vbf": "C2V"}[self.signal]

    def output(self):
        parts = [self.signal] + [p.replace("=", "") for p in self.pois]

        return law.SiblingFileCollection({
            b: self.local_target(self.create_plot_name("morpheddiscr", b, *parts))
            for b in self.bins
        })

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np
        ROOT = import_ROOT()

        # prepare the output
        output = self.output()
        output.dir.touch()

        # create the morphing data
        morphing_data = self.prepare_morphing_data()

        # create plots per bin
        for bin_name, data in morphing_data.items():
            # prepare uncertainty graphs
            graphs = OrderedDict()
            x_min, x_max = poi_data[self.poi].range
            n_points = int(2 * (x_max - x_min) + 1)
            x_values = np.linspace(x_min, x_max, n_points).tolist()

            for model_name, hists_and_scale_fns in data.items():
                # create a morphed shape per point and store the relative integral error
                itg_errors = []
                for x in x_values:
                    h = self.create_morphed_shape(hists_and_scale_fns, **{self.poi: x})
                    err = ROOT.Double()
                    itg = h.IntegralAndError(1, h.GetNbinsX(), err)
                    itg_errors.append((err / itg) if itg else 0.)
                # create and store a graph
                graphs[model_name] = create_tgraph(n_points, x_values, itg_errors)

            # plot
            self.create_plot(output.targets[bin_name], bin_name, graphs)

    def create_plot(self, output, bin_name, graphs):
        import plotlib.root as r
        ROOT = import_ROOT()

        # prepare histograms and ranges
        labels = [m.replace("_", " ").split("model ", 1)[-1] for m in graphs]
        graphs = list(graphs.values())
        x_min, x_max = poi_data[self.poi].range
        y_max_value = max(ROOT.TMath.MaxElement(g.GetN(), g.GetY()) for g in graphs)
        y_max = 1.35 * y_max_value

        # start plotting
        r.setup_style(props={"Palette": ROOT.kRainBow})
        canvas, (pad,) = r.routines.create_canvas()
        pad.cd()
        draw_objs = []
        legend_entries = []

        # dummy histogram to control axes
        x_title = "{} ({})".format(to_root_latex(poi_data[self.poi].label), ", ".join(self.pois))
        y_title = "Relative statistical error of {} discriminant".format(self.signal)
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
        r.setup_hist(h_dummy, pad=pad, props={"Minimum": 0, "Maximum": y_max, "LineWidth": 0})
        draw_objs.append((h_dummy, "HIST"))

        # write graphs
        for graph, label in zip(graphs, labels):
            r.setup_graph(graph, props={"LineWidth": 1, "MarkerStyle": 20, "MarkerSize": 0.5})
            draw_objs.append((graph, "SAME,PL,PLC,PMC"))
            legend_entries.append((graph, label))

        # legend
        legend_cols = min(int(math.ceil(len(legend_entries) / 4.)), 3)
        legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
        legend = r.routines.create_legend(pad=pad, width=legend_cols * 210, height=legend_rows * 30,
            props={"NColumns": legend_cols, "FillStyle": 1001})
        r.fill_legend(legend, legend_entries)
        draw_objs.append(legend)

        # cms label
        cms_labels = r.routines.create_cms_labels(pad=pad)
        draw_objs.extend(cms_labels)

        # bin label
        bin_label = r.routines.create_top_right_label(bin_name, pad=pad)
        draw_objs.append(bin_label)

        # draw all objects
        r.routines.draw_objects(draw_objs)

        # save
        r.update_canvas(canvas)
        canvas.SaveAs(output.path)
