# coding: utf-8

"""
Collection of lightweight, functional helpers to create HEPData entries.
"""


from collections import OrderedDict

import yaml
from scinum import Number, create_hep_data_representer

from dhi.util import DotDict, import_ROOT, prepare_output, create_tgraph
from dhi.plots.util import get_graph_points


#
# setup and general helpers
#

# configure yaml to transparently encode OrderedDict and DotDict instances like normal dicts
map_repr = lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items())
yaml.add_representer(OrderedDict, map_repr)
yaml.add_representer(DotDict, map_repr)

# configure yaml to convert scinum.Number's with uncertainties to hep data value structs
yaml.add_representer(Number, create_hep_data_representer())


class Dumper(yaml.Dumper):
    """
    Custom dumper class ensuring that sequence items are idented.
    """

    def increase_indent(self, *args, **kwargs):
        kwargs["indentless"] = False
        return super(Dumper, self).increase_indent(*args, **kwargs)


def save_hep_data(data, path, **kwargs):
    # default configs
    kwargs.setdefault("indent", 2)

    # forced configs
    kwargs["Dumper"] = Dumper

    # prepare the output
    path = prepare_output(path)

    # dump
    with open(path, "w") as f:
        yaml.dump(data, f, **kwargs)
    print("written HEPData file to {}".format(path))

    return path


#
# structured data creators
#

def create_hist_data(independent_variables=None, dependent_variables=None):
    # create the entry
    data = OrderedDict()

    # add placeholders for in/dependent variables
    data["independent_variables"] = independent_variables or []
    data["dependent_variables"] = dependent_variables or []

    return data


def create_independent_variable(label, unit=None, values=None, parent=None):
    v = OrderedDict()

    # header
    v["header"] = OrderedDict({"name": label})
    if unit is not None:
        v["header"]["units"] = unit

    # values
    v["values"] = values or []

    # add to parent hist data
    if parent is not None:
        parent["independent_variables"].append(v)

    return v


def create_dependent_variable(label, unit=None, qualifiers=None, values=None, parent=None):
    v = OrderedDict()

    # header
    v["header"] = OrderedDict({"name": label})
    if unit is not None:
        v["header"]["units"] = unit

    # qualifiers
    if qualifiers is not None:
        v["qualifiers"] = qualifiers

    # values
    v["values"] = values or []

    # add to parent hist data
    if parent is not None:
        parent["dependent_variables"].append(v)

    return v


def create_qualifier(name, value, unit=None, parent=None):
    q = OrderedDict()

    # name and value
    q["name"] = name
    q["value"] = value

    # unit
    if unit is not None:
        q["units"] = unit

    # add to parent dependent variable
    if parent is not None:
        parent.setdefault("qualifiers", []).append(q)

    return q


def create_range(start, stop, parent=None):
    r = OrderedDict([("low", start), ("high", stop)])

    # add to parent values
    if parent is not None:
        parent.append(r)

    return r


def create_value(value, errors=None, parent=None):
    v = OrderedDict()

    v["value"] = value

    if errors is not None:
        v["errors"] = errors

    # add to parent values
    if parent is not None:
        parent.append(v)

    return v


def create_error(value, label=None, parent=None):
    e = OrderedDict()

    # error values
    if isinstance(value, (list, tuple)) and len(value) == 2:
        e["asymerror"] = OrderedDict([("plus", value[0]), ("minus", value[1])])
    else:
        e["symerror"] = value

    # label
    if label is not None:
        e["label"] = label

    # add to parent value
    if parent is not None:
        parent.setdefault("errors", []).append(e)

    return e


#
# adapters
#

def create_independent_variable_from_x_axis(x_axis, label=None, unit=None, parent=None,
        transform=None):
    # default transform
    if not callable(transform):
        transform = lambda bin_number, low, high: (low, high)

    # default label
    if label is None:
        label = x_axis.GetTitle()

    # extract bin ranges
    values = []
    for b in range(1, x_axis.GetNbins() + 1):
        low, high = x_axis.GetBinLowEdge(b), x_axis.GetBinUpEdge(b)

        # transform
        low, high = transform(b, low, high)

        # add a new value
        values.append(create_range(low, high))

    return create_independent_variable(label, unit=unit, values=values, parent=parent)


def create_dependent_variable_from_hist(hist, label=None, unit=None, qualifiers=None, parent=None,
        error_label=None, rounding_method=None, transform=None):
    # default transform
    if not callable(transform):
        transform = lambda bin_number, value, error: (value, error)

    # default label
    if label is None:
        label = hist.GetTitle()

    # get values
    values = []
    for b in range(1, hist.GetXaxis().GetNbins() + 1):
        v = hist.GetBinContent(b)

        # extract the error
        err = {}
        if error_label:
            err_u = hist.GetBinErrorUp(b)
            err_d = hist.GetBinErrorLow(b)
            err[error_label] = abs(err_u) if err_u == err_d else (err_u, err_d)

        # transform if required, create a number object, and add it
        v, err = transform(b, v, err)
        num = Number(v, err, default_format=rounding_method)
        values.append(num)

    return create_dependent_variable(label, unit=unit, qualifiers=qualifiers, values=values,
        parent=parent)


def create_dependent_variable_from_graph(graph, label=None, unit=None, qualifiers=None, parent=None,
        coord="y", x_values=None, error_label=None, rounding_method=None, transform=None):
    ROOT = import_ROOT()

    # checks
    if coord not in ["x", "y"]:
        raise Exception("coord must be 'x' or 'y', got {}".format(coord))

    # default transform
    if not callable(transform):
        transform = lambda index, x_value, y_value, error: (x_value, y_value, error)

    # default label
    if label is None:
        label = graph.GetTitle()

    # helper to create splines for interpolation
    def make_spline(x, y):
        # delete repeated horizontal endpoints which lead to interpolation failures
        x, y = list(x), list(y)
        if len(x) > 1 and x[0] == x[1]:
            x, y = x[1:], y[1:]
        if len(x) > 1 and x[-1] == x[-2]:
            x, y = x[:-1], y[:-1]
        return ROOT.TSpline3("spline", create_tgraph(len(x), x, y), "", x[0], x[-1])

    # build values dependent on the coordiate to extract
    values = []

    if coord == "x":
        # when x coordinates are requested, just get graph values and optionally obtain errors
        gx, gy = get_graph_points(graph)
        for i, (x, y) in enumerate(zip(gx, gy)):
            x, y = float(x), float(y)

            # extract the error
            err = {}
            if error_label:
                err_u = graph.GetErrorXhigh(i)
                err_d = graph.GetErrorXlow(i)
                err[error_label] = abs(err_u) if err_u == err_d else (err_u, err_d)

            # transform if required, create a number object, and add it
            x, y, err = transform(i, x, y, err)
            num = Number(x, err, default_format=rounding_method)
            values.append(num)

    else:  # coord == "y"
        # when y coordinates are requested, consider custom x values and use interpolation splines
        # for both nominal values and errors
        points = get_graph_points(graph, errors=True)
        gx, gy, errors = points[0], points[1], points[2:]
        has_errors, has_asym_errors = len(errors) > 0, len(errors) > 2
        spline = make_spline(gx, gy)
        if error_label and has_errors:
            if has_asym_errors:
                spline_err_u = make_spline(gx, errors[3])
                spline_err_d = make_spline(gx, errors[2])
            else:
                spline_err = make_spline(gx, errors[1])

        # determine x values to scan
        if x_values is None:
            x_values = gx

        for i, x in enumerate(x_values):
            x = float(x)
            y = spline.Eval(x)

            # extract the error
            err = {}
            if error_label and has_errors:
                if has_asym_errors:
                    err[error_label] = (spline_err_u.Eval(x), spline_err_d.Eval(x))
                else:
                    err[error_label] = spline_err.Eval(x)

            # transform if required, create a number object, and add it
            x, y, err = transform(i, x, y, err)
            num = Number(y, err, default_format=rounding_method)
            values.append(num)

    return create_dependent_variable(label, unit=unit, qualifiers=qualifiers, values=values,
        parent=parent)
