# coding: utf-8

"""
Different ROOT style configurations to be used with plotlib.
"""

import functools


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
