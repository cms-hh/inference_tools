# coding: utf-8

"""
Different helpers and ROOT style configurations to be used with plotlib.
"""

import functools
import collections

from dhi.config import poi_data, br_hh_names
from dhi.util import try_int, to_root_latex


_styles = {}


def _setup_styles():
    global _styles
    if _styles:
        return

    import plotlib.root as r

    # dhi_default
    s = _styles["dhi_default"] = r.styles.copy("default", "dhi_default")
    s.legend_y2 = -15
    s.legend_dy = 32
    s.legend.TextSize = 20
    s.legend.FillStyle = 1


def use_style(style_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import plotlib.root as r

            # setup styles
            _setup_styles()

            # invoke the wrapped function in that style
            with r.styles.use(style_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def draw_model_parameters(model_parameters, pad, grouped=False, x_offset=25, y_offset=40, dy=24,
        props=None):
    import plotlib.root as r
    from plotlib.util import merge_dicts

    # merge properties with defaults
    props = merge_dicts({"TextSize": 20}, props)

    labels = []

    if grouped:
        # group parameters by value
        groups = collections.OrderedDict()
        for p, v in model_parameters.items():
            p_label = poi_data.get(p, {}).get("label", p)
            groups.setdefault(v, []).append(p_label)

        # create labels
        for i, (v, ps) in enumerate(groups.items()):
            label = "{} = {}".format(" = ".join(map(str, ps)), try_int(v))
            label = r.routines.create_top_left_label(label, pad=pad, props=props, x_offset=x_offset,
                y_offset=y_offset + i * dy)
            labels.append(label)

    else:
        # create one label per parameter
        for i, (p, v) in enumerate(model_parameters.items()):
            p_label = poi_data.get(p, {}).get("label", p)
            label = "{} = {}".format(p_label, try_int(v))
            label = r.routines.create_top_left_label(label, pad=pad, props=props, x_offset=x_offset,
                y_offset=y_offset + i * dy)
            labels.append(label)

    return labels


def create_hh_process_label(poi="r", br=None):
    return "{} #rightarrow HH{}{}".format(
        {"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}.get(poi, "pp"),
        {"r_qqhh": "jj"}.get(poi, ""),
        "#scale[0.75]{{ ({})}}".format(to_root_latex(br_hh_names[br])) if br in br_hh_names else "",
    )


def determine_limit_digits(limit, is_xsec=False):
    # TODO: adapt to publication style
    if is_xsec:
        if limit < 10:
            return 2
        elif limit < 200:
            return 1
        else:
            return 0
    else:
        if limit < 10:
            return 2
        elif limit < 300:
            return 1
        else:
            return 0
