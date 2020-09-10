# coding: utf-8

import os
import glob
import itertools

import law
import luigi
import luigi.util

from dhi.tasks.base import CHBase, AnalysisTask, HTCondorWorkflow
from dhi.utils.util import *


class DCBase(CHBase):
    """
    A task that requires datacards in its downstream dependencies, that can have quite longish
    names and are therefore not encoded in the output paths of tasks inheriting from this class.
    Instead, it defines a generic prefix that can be prepended to its outputs, and defines other
    parameters that are significant for the datacard handling.
    """

    input_cards = law.CSVParameter(
        default=os.getenv("DHI_EXAMPLE_CARDS"),
        description="Path to input datacards, comma seperated:"
        "e.g.: '/path/to/card1.txt,/path/to/card2.txt,...'",
    )
    dc_prefix = luigi.Parameter(
        default="", description="prefix to prepend to output file paths, default: ''"
    )
    hh_model = luigi.Parameter(
        default="HHdefault", description="the name of the HH model, default: HHdefault"
    )
    stack_cards = luigi.BoolParameter(
        default=False,
        description="when True, stack histograms and propagate fractional uncertainties instead of "
        "using combineCards.py to extend the number of bins, default: False",
    )

    @classmethod
    def modify_param_values(cls, params):
        """
        Interpret globbing statements in input_cards, expand variables and remove duplicates.
        """
        cards = params.get("input_cards")
        if isinstance(cards, tuple):
            unique_cards = []
            for card in sum((glob.glob(card) for card in cards), []):
                card = os.path.expandvars(os.path.expanduser(card))
                if card not in unique_cards:
                    unique_cards.append(card)
            params["input_cards"] = tuple(unique_cards)
        return params

    def store_parts(self):
        parts = super(DCBase, self).store_parts()
        parts["hh_model"] = self.hh_model
        return parts

    def local_target_dc(self, *path, **kwargs):
        cls = law.LocalFileTarget if not kwargs.pop("dir", False) else law.LocalDirectoryTarget
        store = kwargs.pop("store", self.default_store)
        store = os.path.expandvars(os.path.expanduser(store))
        if path:
            # add the dc_prefix to the last path fragment
            last_parts = path[-1].rsplit(os.sep, 1)
            last_parts[-1] = self.dc_prefix + last_parts[-1]
            last_path = os.sep.join(last_parts)

            # insert a postfix before the file extension when stacking is selected
            postfix = ""
            if self.stack_cards:
                postfix = "_stacked"
            last_path = "{1}{0}{2}".format(postfix, *os.path.splitext(last_path))

            # add the last path fragment back
            path = path[:-1] + (last_path,)

        return cls(self.local_path(*path, store=store), **kwargs)


class CombDatacards(DCBase):
    def output(self):
        outputs = {"datacard": self.local_target_dc("datacard.txt")}
        if self.stack_cards:
            outputs["shapes"] = self.local_target_dc("shapes.root")
        return outputs

    @property
    def cmd(self):
        inputs = " ".join(self.input_cards)
        outputs = self.output()

        if len(self.input_cards) == 1:
            return "cp {} {}".format(inputs, outputs["datacard"].path)
        elif self.stack_cards:
            return "stackCards.py -m {} -i {} -o {} {}".format(
                self.mass,
                inputs,
                outputs["datacard"].path,
                outputs["shapes"].path,
            )
        else:
            return "combineCards.py {datacards} > {out}".format(
                datacards=inputs,
                out=outputs["datacard"].path,
            )


class NLOT2W(DCBase):
    def requires(self):
        return CombDatacards.req(self, mass=self.mass)

    def output(self):
        return self.local_target_dc(
            "workspace_{}.root".format(self.hh_model), store="$DHI_LOCAL_STORE"
        )

    @property
    def cmd(self):
        inputs = self.input()
        return (
            "text2workspace.py {datacard}"
            " -m {mass}"
            " -o {workspace}"
            " -P dhi.utils.models:{model}"
        ).format(
            datacard=inputs["datacard"].path,
            workspace=self.output().path,
            mass=self.mass,
            model=self.hh_model,
        )


class NLOBase1D(DCBase):

    k_pois = ("kl", "kt", "CV", "C2V")
    r_pois = ("r", "r_qqhh", "r_gghh")
    all_pois = k_pois + r_pois

    poi = luigi.ChoiceParameter(default="kl", choices=k_pois)
    poi_range = law.CSVParameter(cls=luigi.IntParameter, default=(-30, 30), min_len=2, max_len=2)

    def __init__(self, *args, **kwargs):
        super(NLOBase1D, self).__init__(*args, **kwargs)

        # store poi infos
        self.other_pois = [p for p in self.all_pois if p != self.poi]
        self.freeze_params = ",".join(self.other_pois)
        self.set_params = ",".join([p + "=1" for p in self.other_pois])

    def store_parts(self):
        parts = super(NLOBase1D, self).store_parts()
        parts["poi"] = "{}_{}_{}".format(self.poi, *self.poi_range)
        return parts


class NLOBase2D(DCBase):

    k_pois = ("kl", "kt", "CV", "C2V")
    r_pois = ("r", "r_qqhh", "r_gghh")
    all_pois = k_pois + r_pois

    poi1 = luigi.ChoiceParameter(default="kl", choices=k_pois)
    poi2 = luigi.ChoiceParameter(default="kt", choices=k_pois)
    poi1_range = law.CSVParameter(cls=luigi.IntParameter, default=(-30, 30), min_len=2, max_len=2)
    poi2_range = law.CSVParameter(cls=luigi.IntParameter, default=(-10, 10), min_len=2, max_len=2)

    def __init__(self, *args, **kwargs):
        super(NLOBase2D, self).__init__(*args, **kwargs)

        # poi's should differ
        assert self.poi1 != self.poi2

        # define params
        self.other_pois = [p for p in self.all_pois if p not in (self.poi1, self.poi2)]
        self.freeze_params = ",".join(self.other_pois)
        self.set_params = ",".join([p + "=1" for p in self.other_pois])

    def store_parts(self):
        parts = super(NLOBase2D, self).store_parts()
        parts["pois"] = "{0}_{2}_{3}__{1}_{4}_{5}".format(
            self.poi1, self.poi2, *(self.poi1_range + self.poi2_range)
        )
        return parts


class NLOLimit(NLOBase1D, HTCondorWorkflow, law.LocalWorkflow):
    def create_branch_map(self):
        return list(range(self.poi_range[0], self.poi_range[1] + 1))

    def output(self):
        return self.local_target_dc("limit_{}.json".format(self.branch_data))

    def requires(self):
        return NLOT2W.req(self)

    def workflow_requires(self):
        reqs = super(NLOLimit, self).workflow_requires()
        reqs["nlolimit"] = self.requires_from_branch()
        return reqs

    @property
    def cmd(self):
        return (
            "combine -M AsymptoticLimits {workspace}"
            " -m {mass} {stable_options} -v 1 --keyword-value {poi}={point}"
            " --run expected --noFitAsimov"
            " --redefineSignalPOIs r --setParameters {set_params},{poi}={point}"
            "&& combineTool.py -M CollectLimits higgsCombineTest.AsymptoticLimits.mH{mass}.{poi}{point}.root"
            " -o {limit}"
        ).format(
            workspace=self.input().path,
            mass=self.mass,
            poi=self.poi,
            point=self.branch_data,
            limit=self.output().path,
            stable_options=self.stable_options,
            set_params=self.set_params,
        )


class NLOScan1D(NLOBase1D, HTCondorWorkflow, law.LocalWorkflow):

    points = luigi.IntParameter(default=200, description="Number of points to scan. Default: 200")

    def create_branch_map(self):
        return list(range(self.points))

    def store_parts(self):
        parts = super(NLOScan1D, self).store_parts()
        parts["points"] = self.points
        return parts

    def requires(self):
        return NLOT2W.req(self)

    def output(self):
        return self.local_target_dc(
            "scan1d_{}.root".format(self.branch_data), store="$DHI_LOCAL_STORE"
        )

    def workflow_requires(self):
        reqs = super(NLOScan1D, self).workflow_requires()
        reqs["nloscan1d"] = self.requires_from_branch()
        return reqs

    @property
    def cmd(self):
        return (
            "combine -M MultiDimFit {workspace}"
            " -m {mass} -t -1 {stable_options}"
            " --algo grid -v 1 --points {points} --robustFit 1 --X-rtd MINIMIZER_analytic"
            " --redefineSignalPOIs {poi}"
            " --setParameterRanges {poi}={range}"
            " --setParameters {set_params}"
            " --freezeParameters {freeze_params}"
            " --firstPoint {point} --lastPoint {point} -n .Test.POINTS.{point}"
            " && mv higgsCombine.Test.POINTS.{point}.MultiDimFit.mH{mass}.root {output}"
        ).format(
            workspace=self.input().path,
            mass=self.mass,
            stable_options=self.stable_options,
            points=self.points,
            poi=self.poi,
            range=",".join(str(r) for r in self.poi_range),
            set_params=self.set_params,
            freeze_params=self.freeze_params,
            point=self.branch_data,
            output=self.output().path,
        )


@luigi.util.inherits(NLOScan1D)
class MergeScans1D(NLOBase1D):
    def requires(self):
        return NLOScan1D.req(self)

    def store_parts(self):
        parts = super(MergeScans1D, self).store_parts()
        parts["points"] = self.points
        return parts

    def output(self):
        return self.local_target("scan1d_merged.npz")

    def run(self):
        from tqdm import tqdm
        import numpy as np
        import uproot

        inputs = [inp.path for inp in self.input()["collection"].targets.values()]
        pb = tqdm(inputs, total=len(inputs), unit="files", desc="Extract fit results")
        deltaNLL, poi = [], []
        for inp in pb:
            f = uproot.open(inp)["limit"]
            deltaNLL.append(np.delete(f["deltaNLL"].array(), 0))
            poi.append(np.delete(f[self.poi].array(), 0))
        deltaNLL = np.concatenate(deltaNLL, axis=None)
        poi = np.concatenate(poi, axis=None)
        self.output().dump(poi=poi, deltaNLL=deltaNLL)


class NLOScan2D(NLOBase2D, HTCondorWorkflow, law.LocalWorkflow):

    points = luigi.IntParameter(default=1024, description="Number of points to scan. Default: 1024")

    def __init__(self, *args, **kwargs):
        super(NLOScan2D, self).__init__(*args, **kwargs)

        if not is_pow2(self.points):
            new_points = next_pow2(self.points)
            self.publish_message(
                "The number of points has to be a number of power 2. Rounding up... old: {} new: {}".format(
                    self.points, new_points
                )
            )
            self.points = new_points

    def create_branch_map(self):
        return list(range(self.points))

    def store_parts(self):
        parts = super(NLOScan2D, self).store_parts()
        parts["points"] = self.points
        return parts

    def requires(self):
        return NLOT2W.req(self)

    def output(self):
        return self.local_target_dc(
            "scan2d_{}.root".format(self.branch_data), store="$DHI_LOCAL_STORE"
        )

    def workflow_requires(self):
        reqs = super(NLOScan2D, self).workflow_requires()
        reqs["nloscan2d"] = self.requires_from_branch()
        return reqs

    @property
    def cmd(self):
        return (
            "combine -M MultiDimFit {workspace}"
            " -m {mass} -t -1 {stable_options}"
            " --algo grid -v 1 --points {points} --robustFit 1 --X-rtd MINIMIZER_analytic"
            " --redefineSignalPOIs {poi1},{poi2}"
            " --setParameterRanges {poi1}={range1}:{poi2}={range2}"
            " --setParameters {set_params}"
            " --freezeParameters {freeze_params}"
            " --firstPoint {point} --lastPoint {point} -n .Test.POINTS.{point}"
            " && mv higgsCombine.Test.POINTS.{point}.MultiDimFit.mH{mass}.root {output}"
        ).format(
            workspace=self.input().path,
            mass=self.mass,
            stable_options=self.stable_options,
            points=self.points,
            poi1=self.poi1,
            poi2=self.poi2,
            range1=",".join(str(r) for r in self.poi1_range),
            range2=",".join(str(r) for r in self.poi2_range),
            set_params=self.set_params,
            freeze_params=self.freeze_params,
            point=self.branch_data,
            output=self.output().path,
        )


@luigi.util.inherits(NLOScan2D)
class MergeScans2D(NLOBase2D):
    def requires(self):
        return NLOScan2D.req(self)

    def store_parts(self):
        parts = super(MergeScans2D, self).store_parts()
        parts["points"] = self.points
        return parts

    def output(self):
        return self.local_target("scan2d_merged.npz")

    def run(self):
        from tqdm import tqdm
        import numpy as np
        import uproot

        inputs = [inp.path for inp in self.input()["collection"].targets.values()]
        pb = tqdm(inputs, total=len(inputs), unit="files", desc="Extract fit results")
        deltaNLL, poi1, poi2 = [], [], []
        for inp in pb:
            f = uproot.open(inp)["limit"]
            deltaNLL.append(np.delete(f["deltaNLL"].array(), 0))
            poi1.append(np.delete(f[self.poi1].array(), 0))
            poi2.append(np.delete(f[self.poi2].array(), 0))
        deltaNLL = np.concatenate(deltaNLL, axis=None)
        poi1 = np.concatenate(poi1, axis=None)
        poi2 = np.concatenate(poi2, axis=None)
        self.output().dump(poi1=poi1, poi2=poi2, deltaNLL=deltaNLL)


class ImpactsPulls(DCBase):

    r = luigi.FloatParameter(default=1.0, description="Inject signal strength (r). Default: 1.")
    r_range = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(0, 30),
        min_len=2,
        max_len=2,
        description="Ranges for signal strength. Default: 0..30",
    )

    params = "--redefineSignalPOIs r --setParameters r_gghh=1,r_qqhh=1,kt=1,kl=1,CV=1,C2V=1"

    def __init__(self, *args, **kwargs):
        super(ImpactsPulls, self).__init__(*args, **kwargs)
        assert self.r >= 0.0
        self.r_low, self.r_high = self.r_range
        assert self.r_low >= 0.0
        assert self.r_high > 0.0
        assert self.r_high > self.r_low

    def requires(self):
        return NLOT2W.req(self)

    def output(self):
        return self.local_target_dc("impacts.json")

    @property
    def cmd(self):
        initial_fit = (
            "combineTool.py -M Impacts -d {workspace}"
            " -m {mass} -t -1 "
            " --cminDefaultMinimizerStrategy 0 --robustFit 1 --X-rtd MINIMIZER_analytic"
            " --expectSignal={r} --setParameterRanges r={r_low},{r_high}"
            " --doInitialFit"
            " --parallel {cores} {params}"
        ).format(
            workspace=self.input().path,
            mass=self.mass,
            r=self.r,
            r_low=self.r_low,
            r_high=self.r_high,
            cores=os.environ["DHI_N_CORES"],
            params=self.params,
        )
        fit = initial_fit.replace("--doInitialFit", "--doFits")
        impacts = (
            "combineTool.py -M Impacts -d {workspace} -m {mass} --output {impacts} {params} --setParameterRanges r={r_low},{r_high}"
        ).format(
            mass=self.mass,
            workspace=self.input().path,
            impacts=self.output().path,
            params=self.params,
            r_low=self.r_low,
            r_high=self.r_high,
        )
        return " && ".join([initial_fit, fit, impacts])
