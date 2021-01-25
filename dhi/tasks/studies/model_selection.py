# coding: utf-8

"""
Tasks related to tests of the physics model(s).
"""

import math
from collections import OrderedDict, defaultdict

import luigi
import law

from dhi.tasks.base import PlotTask, view_output_plots
from dhi.tasks.combine import (
    HHModelTask,
    MultiHHModelTask,
    DatacardTask,
    ParameterValuesTask,
    ParameterScanTask,
    CombineDatacards,
)
from dhi.datacard_tools import create_datacard_instance
from dhi.config import poi_data, color_sequence
from dhi.util import import_ROOT, create_tgraph, to_root_latex


class PlotMorphingScales(PlotTask, HHModelTask, ParameterScanTask, ParameterValuesTask):

    task_namespace = "study"

    signal = luigi.ChoiceParameter(
        choices=["ggf", "vbf"],
        description="the signal process to morph; choices: ggf,vbf",
    )
    x_min = None
    x_max = None
    y_min = None
    y_max = None

    force_n_scan_parameters = 1

    @classmethod
    def modify_param_values(cls, params):
        params = ParameterScanTask.modify_param_values(params)
        params = ParameterValuesTask.modify_param_values(params)
        return params

    def get_output_postfix(self, join=True):
        parts = ParameterScanTask.get_output_postfix(self, join=False)
        parts.extend(ParameterValuesTask.get_output_postfix(self, join=False))
        return self.join_postfix(parts) if join else parts

    def output(self):
        name = self.create_plot_name(["morphingfractions", self.signal, self.get_output_postfix()])
        return self.local_target(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import plotlib.root as r

        ROOT = import_ROOT()

        # get the model and the formula corresponding to the signal class
        model = self.load_hh_model()[1]
        formula = getattr(model, "{}_formula".format(self.signal))

        # store scaling functions mapped to samples in the formula
        scale_fns = OrderedDict(zip(formula.sample_list, formula.coeffs))

        # create graphs
        graphs = []
        for sample, scale_fn in scale_fns.items():
            x, y = [], []
            for v in self.get_scan_linspace():
                subs = {self.scan_parameter_names[0]: v[0]}
                subs.update(self.parameter_values_dict)

                # check if all symbols are substituted
                missing_symbols = set(map(str, scale_fn.free_symbols)) - set(subs.keys())
                if missing_symbols:
                    raise Exception(
                        "scaling function misses substitutions for symbol(s) '{}', "
                        "please add them via --parameter-values".format(",".join(missing_symbols))
                    )

                # get the scaling value
                scale = scale_fn.evalf(subs=subs)
                x.append(v[0])
                y.append(scale)
            graphs.append(create_tgraph(len(x), x, y))

        # prepare histograms and ranges
        labels = [sample.label.split("_", 1)[1] for sample in scale_fns]
        x_min, x_max = self.scan_parameters[0][1:3]
        y_min_value = min(ROOT.TMath.MinElement(g.GetN(), g.GetY()) for g in graphs)
        y_max_value = max(ROOT.TMath.MaxElement(g.GetN(), g.GetY()) for g in graphs)
        y_max = y_min_value + 1.35 * (y_max_value - y_min_value)

        # start plotting
        r.setup_style(props={"Palette": ROOT.kRainBow})
        canvas, (pad,) = r.routines.create_canvas()
        pad.cd()
        draw_objs = []
        legend_entries = []

        # dummy histogram to control axes
        x_title = "{} ({})".format(
            to_root_latex(poi_data[self.scan_parameter_names[0]].label),
            self.joined_parameter_values,
        )
        y_title = "Morphing scales"
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
        r.setup_hist(
            h_dummy, pad=pad, props={"Minimum": y_min_value, "Maximum": y_max, "LineWidth": 0}
        )
        draw_objs.append((h_dummy, "HIST"))

        # write graphs
        for graph, label, col in zip(graphs, labels, color_sequence[:len(graphs)]):
            r.setup_graph(graph, props={"LineWidth": 1, "MarkerStyle": 20, "MarkerSize": 0.5},
                color=col)
            draw_objs.append((graph, "SAME,PL"))
            legend_entries.append((graph, label))

        # legend
        legend_cols = min(int(math.ceil(len(legend_entries) / 4.0)), 3)
        legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
        legend = r.routines.create_legend(
            pad=pad,
            width=legend_cols * 280,
            n=legend_rows,
            props={"NColumns": legend_cols, "FillStyle": 1001},
        )
        r.fill_legend(legend, legend_entries)
        draw_objs.append(legend)

        # model label
        model_label = r.routines.create_top_right_label(model.name, pad=pad)
        draw_objs.append(model_label)

        # cms label
        cms_labels = r.routines.create_cms_labels(pad=pad)
        draw_objs.extend(cms_labels)

        # draw all objects
        r.routines.draw_objects(draw_objs)

        # save
        r.update_canvas(canvas)
        output = self.output()
        output.parent.touch()
        canvas.SaveAs(output.path)


class PlotMorphedDiscriminant(PlotTask, DatacardTask, MultiHHModelTask, ParameterValuesTask):

    task_namespace = "study"

    signal = luigi.ChoiceParameter(
        choices=["ggf", "vbf"],
        description="the signal process to morph; choices: ggf,vbf",
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

    def requires(self):
        return [CombineDatacards.req(self, hh_model=hh_model) for hh_model in self.hh_models]

    def output(self):
        parts = []
        if self.y_log:
            parts.append("log")

        return {
            b: self.local_target(
                self.create_plot_name(
                    ["morpheddiscr", self.signal, b, self.get_output_postfix()] + parts
                )
            )
            for b in self.bins
        }

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
                    raise Exception(
                        "no signal found for sample {} in model {}".format(sample.label, model.name)
                    )

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

    def create_morphed_shape(self, hists_and_scale_fns, **parameter_values):
        # merge additional poi values
        parameter_values = law.util.merge_dicts(self.parameter_values_dict, parameter_values)

        # ensure that C* values are upper case
        for k, v in list(parameter_values.items()):
            if k.startswith("c"):
                del parameter_values[k]
                parameter_values[k.upper()] = v

        morphed_shape = None
        for h, scale_fn in hists_and_scale_fns:
            h = h.Clone()

            # check if all symbols are substituted
            missing_symbols = set(map(str, scale_fn.free_symbols)) - set(parameter_values.keys())
            if missing_symbols:
                raise Exception(
                    "scaling function misses substitutions for symbol(s) '{}', please "
                    "add them via --pois".format(",".join(missing_symbols))
                )

            # perform the scaling
            scale = scale_fn.evalf(subs=parameter_values)
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
        list(output.values())[0].parent.touch()

        # create the morphing data
        morphing_data = self.prepare_morphing_data()

        # create plots per bin
        for bin_name, data in morphing_data.items():
            # prepare morphed shapes
            shapes = OrderedDict()
            for model_name, hists_and_scale_fns in data.items():
                shapes[model_name] = self.create_morphed_shape(hists_and_scale_fns)

            # plot
            self.create_plot(output[bin_name], bin_name, shapes)

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
            y_max = y_min * 10 ** (1.35 * math.log10(y_max_value / y_min))
        else:
            y_max = 1.35 * y_max_value

        # start plotting
        r.setup_style(props={"Palette": ROOT.kRainBow})
        canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": self.y_log})
        pad.cd()
        draw_objs = []
        legend_entries = []

        # dummy histogram to control axes
        x_title = "{} discriminant ({})".format(self.signal, self.joined_parameter_values)
        y_title = "Predicted signal yield"
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
        r.setup_hist(h_dummy, pad=pad, props={"Minimum": y_min, "Maximum": y_max, "LineWidth": 0})
        draw_objs.append((h_dummy, "HIST"))

        # write histograms
        for hist, label, col in zip(hists, labels, color_sequence[:len(hists)]):
            r.setup_hist(hist, props={"LineWidth": 1}, color=col, color_flags="lm")
            draw_objs.append((hist, "SAME,HIST,E"))
            legend_entries.append((hist, label))

        # legend
        legend_cols = min(int(math.ceil(len(legend_entries) / 4.0)), 3)
        legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
        legend = r.routines.create_legend(
            pad=pad,
            width=legend_cols * 210,
            n=legend_rows,
            props={"NColumns": legend_cols, "FillStyle": 1001},
        )
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


class PlotStatErrorScan(PlotMorphedDiscriminant, ParameterScanTask):

    y_log = None

    force_n_scan_parameters = 1

    def get_output_postfix(self, join=True):
        parts = ParameterScanTask.get_output_postfix(self, join=False)
        parts.extend(ParameterValuesTask.get_output_postfix(self, join=False))
        return self.join_postfix(parts) if join else parts

    def output(self):
        return {
            b: self.local_target(
                self.create_plot_name(["staterror", self.signal, b, self.get_output_postfix()])
            )
            for b in self.bins
        }

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        ROOT = import_ROOT()

        # prepare the output
        output = self.output()
        list(output.values())[0].parent.touch()

        # create the morphing data
        scan_parameter = self.scan_parameter_names[0]
        morphing_data = self.prepare_morphing_data()

        # create plots per bin
        for bin_name, data in morphing_data.items():
            # prepare uncertainty graphs
            graphs = OrderedDict()
            x_min, x_max = poi_data[scan_parameter].range
            n_points = int(2 * (x_max - x_min) + 1)
            x_values = np.linspace(x_min, x_max, n_points).tolist()

            for model_name, hists_and_scale_fns in data.items():
                # create a morphed shape per point and store the relative integral error
                itg_errors = []
                for x in x_values:
                    h = self.create_morphed_shape(hists_and_scale_fns, **{scan_parameter: x})
                    err = ROOT.Double()
                    itg = h.IntegralAndError(1, h.GetNbinsX(), err)
                    itg_errors.append((err / itg) if itg else 0.0)
                # create and store a graph
                graphs[model_name] = create_tgraph(n_points, x_values, itg_errors)

            # plot
            self.create_plot(output[bin_name], bin_name, graphs)

    def create_plot(self, output, bin_name, graphs):
        import plotlib.root as r

        ROOT = import_ROOT()

        scan_parameter = self.scan_parameter_names[0]

        # prepare histograms and ranges
        labels = [m.replace("_", " ").split("model ", 1)[-1] for m in graphs]
        graphs = list(graphs.values())
        x_min, x_max = poi_data[scan_parameter].range
        y_max_value = max(ROOT.TMath.MaxElement(g.GetN(), g.GetY()) for g in graphs)
        y_max = 1.35 * y_max_value

        # start plotting
        r.setup_style(props={"Palette": ROOT.kRainBow})
        canvas, (pad,) = r.routines.create_canvas()
        pad.cd()
        draw_objs = []
        legend_entries = []

        # dummy histogram to control axes
        x_title = "{} ({})".format(
            to_root_latex(poi_data[scan_parameter].label), self.joined_parameter_values
        )
        y_title = "Relative statistical error of {} discriminant".format(self.signal)
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
        r.setup_hist(h_dummy, pad=pad, props={"Minimum": 0, "Maximum": y_max, "LineWidth": 0})
        draw_objs.append((h_dummy, "HIST"))

        # write graphs
        for graph, label, col in zip(graphs, labels, color_sequence[:len(graphs)]):
            r.setup_graph(graph, props={"LineWidth": 1, "MarkerStyle": 20, "MarkerSize": 0.5},
                color=col)
            draw_objs.append((graph, "SAME,PL"))
            legend_entries.append((graph, label))

        # legend
        legend_cols = min(int(math.ceil(len(legend_entries) / 4.0)), 3)
        legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
        legend = r.routines.create_legend(
            pad=pad,
            width=legend_cols * 210,
            n=legend_rows,
            props={"NColumns": legend_cols, "FillStyle": 1001},
        )
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
