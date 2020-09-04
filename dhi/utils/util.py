# coding: utf-8
import law
import luigi


def WIP(cls):
    """
    This decorator overwrites the constructor of any
    class and quits the program once it is instantiated.
    Additionally it removes the task from `law index`.
    It is useful to prohibit the execution and instantiation
    of law.Tasks/luigi.Tasks, which are currently Work in Progress (WIP).
    """

    def _wip(self, *args, **kwargs):
        raise SystemExit(
            "{} is under development. You can not instantiated this class.".format(cls.__name__)
        )

    if issubclass(cls, (law.Task, luigi.Task)):
        # remove from law index
        cls.exclude_index = True
        # also prevent instantiation
        cls.__init__ = _wip
    return cls


def rgb(r, g, b):
    """
    This function norms the rgb color inputs for
    matplotlib in case they are not normalized to 1.
    Otherwise just return inputs as tuple.
    Additionally this method displays the color inline,
    when using the atom package "pigments"!
    """
    return tuple(v if v <= 1.0 else float(v) / 255.0 for v in (r, g, b))


def next_pow2(num):
    k = 1
    while k < num:
        k = k << 1
    return k
