# coding: utf-8

"""
File containing custom hook functions invoked by the inference tools, designed particularly to
make custom adjustments for the run 2 HH combination.

The inference tools do neither depend on the hooks below, nor do they expect particular behavior,
but they rather just provide the mechanism to invoke custom actions to happen to (e.g.) change task
parameters or combine commands. The separation between generic tools and custom hooks is intended to
keep the former clean while the latter is allowed to be a place for messier "solutions".
"""

import os
import math
from decimal import Decimal

import numpy as np
import scipy.interpolate

from dhi.tasks.combine import DatacardTask
from dhi.tasks.limits import UpperLimits
from dhi.util import real_path, get_dcr2_path, round_digits


def _is_r2c_bbbb_boosted_ggf(task):
    """
    Helper that returns *True* in case all cards passed to a :py:class:`tasks.combine.DatacardTask`
    are part of the run 2 combination, pointing to bbbb boosted datacards and contain at least the
    ggf channel. This channel uses a different toy approach and thus requires special treatment.
    """
    if not isinstance(task, DatacardTask):
        return None

    if not task.r2c_bin_names:
        return False

    all_bbbb_boosted = all(name.startswith("bbbb_boosted_") for name in task.r2c_bin_names)
    return all_bbbb_boosted and "bbbb_boosted_ggf" in task.r2c_bin_names


def init_datacard_task(task):
    """
    Hook called by :py:class:`tasks.combine.DatacardTask` to modify task parameters.
    """
    # when all passed datacards are located in the DHI_DATACARDS_RUN2 directory and have bin names
    # (most probably auto assigned as part of resolve_datacards), store these bin names
    task.r2c_bin_names = None
    if task.datacards:
        dcr2_path = get_dcr2_path()
        if dcr2_path:
            in_dcr2 = lambda path: real_path(path).startswith(dcr2_path.rstrip(os.sep) + os.sep)
            split_cards = list(map(task.split_datacard_path, task.datacards))
            paths = [c[0] for c in split_cards]
            bin_names = [c[1] for c in split_cards]
            if all(bin_names) and all(map(in_dcr2, paths)):
                task.r2c_bin_names = bin_names

    # change 1: for blinded fits with bbbb_boosted_ggf cards, disable snapshots
    if _is_r2c_bbbb_boosted_ggf(task) and not getattr(task, "unblinded", True):
        if getattr(task, "use_snapshot", False):
            task.use_snapshot = False


def modify_combine_params(task, params):
    """
    Hook called by :py:class:`tasks.combine.CombineCommandTask` to modify parameters going to be
    added to combine commands. *params* is a list of key-value pairs (in a 2-list) for options with
    values, and single values (in a 1-list) for flags.
    """
    # change 1: for blinded fits with bbbb_boosted_ggf cards, adjust toy arguments
    if _is_r2c_bbbb_boosted_ggf(task) and not getattr(task, "unblinded", True):
        # gather info
        is_limit = isinstance(task, UpperLimits)
        has_toys_freq = False
        has_bypass = False
        remove_params = []
        for i, param in enumerate(params):
            if param[0] == "--toys" and param[1] == "-1" and is_limit:
                remove_params.append(i)
            elif param[0] == "--toysFrequentist":
                has_toys_freq = True
            elif param[0] == "--bypassFrequentistFit":
                has_bypass = True

        # adjust params
        params = [param for i, param in enumerate(params) if i not in remove_params]
        if not has_toys_freq:
            params.append(["--toysFrequentist"])
        if not has_bypass:
            params.append(["--bypassFrequentistFit"])

    return params


_limit_grid_path = os.getenv("DHI_LIMIT_GRID_PATH", "/eos/user/a/acarvalh/HH_fullRun2_fits/misc")
_limit_grid_interps = {}


def _get_limit_grid_interps(poi, scan_name):
    key = (poi, scan_name)
    if key not in _limit_grid_interps:
        grid_file = os.path.join(_limit_grid_path, "limitscan_{}_{}.npz".format(*key))
        if not os.path.exists(grid_file):
            raise Exception("limit grid file {1} not existing in directory {0}".format(
                *os.path.split(grid_file)))

        f = np.load(grid_file)
        data = f["data"]

        def create_interp(y):
            nan_mask = np.isnan(y)
            y = y[~nan_mask]
            x = data[scan_name][~nan_mask]
            return scipy.interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")

        exp_interp = create_interp(data["limit"])
        up_interp = create_interp(np.maximum(data["limit_p2"], data["observed"]))
        down_interp = create_interp(np.minimum(data["limit_m2"], data["observed"]))
        _limit_grid_interps[key] = (exp_interp, up_interp, down_interp)

    return _limit_grid_interps[key]


def define_limit_grid(task, scan_parameter_values, approx_points, debug=False):
    """
    Hook called by :py:class:`tasks.limits.UpperLimit` to define a scan-parameter-dependent grid of
    POI points for parallelizing limit computations. The grid itself depends on the datacards being
    used, so please consider the grid definitions below as volatile information that is very likely
    subject to constant changes.

    *scan_parameter_values* is a tuple defining the point of the parameter scan, which the grid will
    be based on. *approx_points* should be a rough estimate of the grid points to use. The actual
    grid dimension and granularity is stabilized so that its points are not precise to the last
    digit but tend to use common steps (i.e. 1/2, 1/4, 1/10, etc.).
    """
    # some checks
    if task.n_pois != 1:
        raise Exception("define_limit_grid hook only supports 1 POI, got {}".format(task.n_pois))
    poi = task.pois[0]
    if poi not in ("r",):
        raise Exception("define_limit_grid hook only supports POI r, got {}".format(poi))
    if len(scan_parameter_values) != 1:
        raise Exception("define_limit_grid hook only supports 1 scan parameter, got {}".format(
            len(scan_parameter_values)))
    scan_value = scan_parameter_values[0]
    scan_name = task.scan_parameter_names[0]
    if any(v != 1 for v in task.parameter_values_dict.values()):
        raise Exception("define_limit_grid hook does not support tasks with extra parameter "
            "values that are not 1, got {}".format(task.parameter_values))
    if not os.path.exists(_limit_grid_path):
        raise Exception("define_limit_grid hook cannot access {}".format(_limit_grid_path))

    # detect if any combined datacard setup is used by comparing to environment variables that are
    # usually defined in the combination
    used_cards = set(task.resolve_datacards(task.datacards)[2].split("__"))
    cards_vars = ["COMBCARDS", "COMBCARDS_POINTS", "COMBCARDS_VBF", "COMBCARDS_VBF_POINTS"]
    for cards_var in cards_vars:
        value = os.getenv(cards_var)
        if not value:
            continue
        comb_cards = {c.replace("/", "_") for c in value.split(",")}
        if comb_cards == used_cards:
            break
    else:
        raise Exception("define_limit_grid could not detect combined datacard variables {}".format(
            ",".join(cards_vars)))

    if debug:
        print("defining limit grid for datacards '{}' on POI {} for scan over {} at {}".format(
            cards_var, poi, scan_name, scan_value))

    from_grid = None
    if (poi, scan_name) in (("r", "kl"), ("r", "C2V")):
        # load the previous scan, interpolate limit values, extract a viable range and granularity
        exp, up, down = [interp(scan_value) for interp in _get_limit_grid_interps(poi, scan_name)]
        grid_max = (2 * up - exp) if up > exp else (up * 1.25)
        grid_min = (2 * down - exp) if down < exp else (down * 0.75)
        if grid_min >= grid_max:
            raise Exception("define_limit_grid encountered unstable limit interpolation with "
                "grid_min ({}) >= grid_max ({})".format(grid_min, grid_max))

        # when the minimum is close to 0, just start at 0
        if grid_min < 5:
            grid_min = 0.0

        # round edges to two significant digits
        grid_min = round_digits(grid_min, 2, math.floor)
        grid_max = round_digits(grid_max, 2, math.ceil)
        approx_grid_range = grid_max - grid_min

        # find a viable divider for the desired approximate step size
        dividers = [1.0, 1.25, 2.0, 2.5, 3.125, 5.0, 6.25, 10.0]
        # fewer fix points, but less adherence to approx_points:
        # dividers = [1.0, 1.25, 2.0, 2.5, 5.0, 10.0]
        approx_step_size = approx_grid_range / approx_points
        exp = int(math.floor(math.log(approx_step_size, 10)))
        raised_approx_step_size = approx_step_size / 10.0**exp

        # algorithm 1: find the first divider that is larger than the approx step size
        # for div in dividers:
        #     if div >= raised_approx_step_size:
        #         break
        # else:
        #     raise Exception("no valid divider found for grid between {} and {}".format(
        #         grid_min, grid_max))

        # algorithm 2: find the closest divider
        div = dividers[np.argmin(map((lambda d: abs(d - raised_approx_step_size)), dividers))]

        # get the number of steps between points, compute the new grid_range and adjust edges
        step_size = div * 10**exp
        steps = int(math.ceil(approx_grid_range / step_size))
        grid_range = steps * step_size
        grid_max += grid_range - approx_grid_range

        # ensure that grid edges do not have float uncertainties
        safe_float = lambda f: float(Decimal(str(f)).quantize(Decimal(str(step_size))))
        grid_min = safe_float(grid_min)
        grid_max = safe_float(grid_max)

        # build the grid definition
        from_grid = ((poi, grid_min, grid_max, steps + 1),)
    else:
        raise NotImplementedError("define_limit_grid hook does not support POI {} and scan "
            "parameter {}".format(poi, scan_name))

    if debug:
        print("defined {}".format(from_grid))

    return from_grid


def scale_multi_likelihoods(task, data):
    """
    Hook called by :py:class:`tasks.likelihoods.PlotMultipleLikelihoods.run` to adjust dnll2 values
    in place for optional projection studies. *data* is a list of dictionaries with a field "values"
    that refers to a numpy rec array containing columns "<poi>" and "dnll2". Changes to "dnll2"
    should happend in-place, or *data* can be ammended to create additional scans.
    """
    import numpy as np
    from dhi.plots.likelihoods import _preprocess_values, evaluate_likelihood_scan_1d

    # skip the projection when the env variable is not set
    setup = os.getenv("DHI_NLL_PROJECTION", None)
    if not setup:
        return

    projection, r = "3000 fb^{-1}", 3000.0 / 138.0
    print("")
    print(" Run projection ".center(100, "-"))
    print("name    : {}".format(projection))
    print("r       : {}".format(r))
    print("")

    # input checks
    if not task.n_pois == 1:
        raise Exception("scale_multi_likelihoods hook only supports 1 POI, got {}".format(
            task.n_pos))
    poi = task.pois[0]
    if poi not in ["r", "kl", "C2V"]:
        raise Exception("scale_multi_likelihoods hook only supports POIs r,kl,C2V, got {}".format(
            poi))

    # get values for all systematics and stat. only
    # as this point we must anticipate them on positions 0 and 1
    assert len(data) == 3
    values_all = data[0]["values"]
    values_stat = data[1]["values"]
    values_statexp = data[2]["values"]

    # remove run stat+exp
    del data[2:]
    task.datacard_names = task.datacard_names[:2]

    # helper to obtain scan information
    def get_scan(values, name):
        origin = "{}, poi {}, {}".format(projection, poi, name)
        y, x = _preprocess_values(values["dnll2"], (poi, values[poi]), remove_nans=True,
            shift_negative_values=True, min_is_external=True, origin=origin)
        return evaluate_likelihood_scan_1d(x, y, origin=origin, poi_min=1.0)

    # get the full and stat. only scans
    scan_all = get_scan(values_all, "all")
    scan_stat = get_scan(values_stat, "stat")
    scan_statexp = get_scan(values_statexp, "statext")

    # get errors that parametrize the scaling factor
    # when available, use 2 sigma estimators
    sigma_idx = 1 if None not in scan_all["summary"]["uncertainty"][1] else 0
    print("\nbuilding scaling factor from +-{} sigma uncertainties".format(sigma_idx + 1))
    err_all_u, err_all_d = scan_all["summary"]["uncertainty"][sigma_idx]
    err_stat_u, err_stat_d = scan_stat["summary"]["uncertainty"][sigma_idx]
    err_statexp_u, err_statexp_d = scan_statexp["summary"]["uncertainty"][sigma_idx]
    if None in [err_all_u, err_all_d, err_stat_u, err_stat_d, err_statexp_d, err_statexp_u]:
        raise Exception("scale_multi_likelihoods hook encountered missing {} sigma interval".format(
            sigma_idx + 1))

    # build the scaling factors separately for both sides of the curve
    def add_projection(label, scale_stat, scale_exp, scale_thy):
        assert err_all_u > err_statexp_u > err_stat_u
        assert err_all_d <= err_statexp_d <= err_stat_d

        # when the systematics scale is low (e.g. 1.0) one should rather scale the run 2 stat only
        # curve, but when systematics are assumed to be unchanged (e.g. 1.0), scale the full run 2;
        # therefore, use an average between the two based on the overall systematic scale;
        # to compute this, create a weighted mean of the exp and thy scales using errors as weights
        err_thy_u = (err_all_u**2 - err_statexp_u**2)**0.5
        err_thy_d = (err_all_d**2 - err_statexp_d**2)**0.5
        err_exp_u = (err_statexp_u**2 - err_stat_u**2)**0.5
        err_exp_d = (err_statexp_d**2 - err_stat_d**2)**0.5
        # weighted average
        scale_syst_u = (err_thy_u * scale_thy + err_exp_u * scale_exp) / (err_thy_u + err_exp_u)
        scale_syst_d = (err_thy_d * scale_thy + err_exp_d * scale_exp) / (err_thy_d + err_exp_d)

        # for smoothing between them, consider the two values spanning an ellipse and get the point
        # on the circumference with the syst scale (between 0.0 and 1.0) being interpreted as an
        # angle (between 0 and pi/2)
        angle_u = math.pi * 0.5 * min(max(scale_syst_u, 0.0), 1.0)
        angle_d = math.pi * 0.5 * min(max(scale_syst_d, 0.0), 1.0)
        ellipse_avg = lambda x, y, a: ((x * np.cos(a))**2 + (y * np.sin(a))**2)**0.5
        err_avg_u = ellipse_avg(err_stat_u, err_all_u, angle_u)
        err_avg_d = ellipse_avg(err_stat_d, err_all_d, angle_d)

        # build the scaling factors
        e_all = scale_thy**2
        e_stat = scale_stat**2 - scale_exp**2
        e_statexp = scale_exp**2 - scale_thy**2
        s_u = err_avg_u**2 / (
            err_all_u**2 * e_all +
            err_stat_u**2 * e_stat +
            err_statexp_u**2 * e_statexp
        )
        s_d = err_avg_d**2 / (
            err_all_d**2 * e_all +
            err_stat_d**2 * e_stat +
            err_statexp_d**2 * e_statexp
        )

        # build the dnll2 curve averages
        dnll2_all = np.copy(values_all["dnll2"])
        dnll2_stat = np.copy(values_stat["dnll2"])
        dnll2_u = ellipse_avg(dnll2_stat, dnll2_all, angle_u)
        dnll2_d = ellipse_avg(dnll2_stat, dnll2_all, angle_d)

        # create the projection and apply the scaling
        values_projected = np.copy(values_all)
        poi_min = 1.0
        up_mask = values_projected[poi] >= poi_min
        values_projected["dnll2"][up_mask] = dnll2_u[up_mask] * s_u
        values_projected["dnll2"][~up_mask] = dnll2_d[~up_mask] * s_d

        # add the data entry to plot
        data.append({
            "values": values_projected,
            "poi_min": poi_min,
        })
        task.datacard_names += (label,)

    add_projection(
        label="{}, #epsilon_{{exp}} = #epsilon_{{th}} = 1".format(projection),
        scale_stat=r**-0.5,
        scale_exp=1.0,
        scale_thy=1.0,
    )
    add_projection(
        label="{}, #epsilon_{{exp}} = #epsilon_{{th}} = 0.5".format(projection),
        scale_stat=r**-0.5,
        scale_exp=0.5,
        scale_thy=0.5,
    )
    add_projection(
        label="{}, #epsilon_{{exp}} = 1/#sqrt{{R}}, #epsilon_{{th}} = 0.5".format(projection),
        scale_stat=r**-0.5,
        scale_exp=r**-0.5,
        scale_thy=0.5,
    )
    add_projection(
        label="{}, #epsilon_{{exp}} = #epsilon_{{th}} = 0".format(projection),
        scale_stat=r**-0.5,
        scale_exp=0.0,
        scale_thy=0.0,
    )

    print("")
    print(" Finish projection ".center(100, "-"))
    print("")
