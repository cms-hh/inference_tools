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

from dhi.tasks.combine import DatacardTask
from dhi.tasks.limits import UpperLimits
from dhi.util import real_path, get_dcr2_path


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
