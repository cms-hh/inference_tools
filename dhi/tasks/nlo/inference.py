# coding: utf-8

"""
NLO inference tasks.
"""

import itertools
import multiprocessing

import law
import luigi

from dhi.tasks.base import CombineCommandTask, HTCondorWorkflow
from dhi.tasks.nlo.base import DatacardBaseTask, POIScanTask1D, POIScanTask2D


class CombineDatacards(DatacardBaseTask, CombineCommandTask):
    def output(self):
        return self.local_target_dc("datacard.txt")

    def build_command(self):
        inputs = " ".join(self.input_cards)
        output = self.output()

        # TODO: copy all shape files to same location as before
        if len(self.input_cards) == 1:
            return "cp {} {}".format(inputs, output.path)
        else:
            return "combineCards.py {inputs} > {output}".format(
                inputs=inputs,
                output=output.path,
            )


class CreateWorkspace(DatacardBaseTask, CombineCommandTask):
    def requires(self):
        return CombineDatacards.req(self)

    def output(self):
        return self.local_target_dc("workspace.root")

    def build_command(self):
        return (
            "text2workspace.py {datacard}"
            " -o {workspace}"
            " -m {self.mass}"
            " -P dhi.models.{self.hh_model}"
        ).format(
            self=self,
            datacard=self.input().path,
            workspace=self.output().path,
        )


class LimitScan(POIScanTask1D, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def requires(self):
        return CreateWorkspace.req(self)

    def workflow_requires(self):
        reqs = super(LimitScan, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def create_branch_map(self):
        import numpy as np

        return np.linspace(self.poi_range[0], self.poi_range[1], self.points).round(7).tolist()

    def output(self):
        return self.local_target_dc("limit__{}_{}.root".format(self.poi, self.branch_data))

    def build_command(self):
        return (
            "combine -M AsymptoticLimits {workspace}"
            " -m {self.mass}"
            " -v 1"
            " --keyword-value {self.poi}={point}"  # TODO: remove this and adjust file to copy
            " --run expected"
            " --noFitAsimov"
            " --redefineSignalPOIs r"  # TODO: shouldn't this be {poi}?
            " --setParameters {set_params},{self.poi}={point}"
            " {self.combine_stable_options}"
            " && "
            "mv higgsCombineTest.AsymptoticLimits.mH{self.mass}.{self.poi}{point}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            point=self.branch_data,
            set_params=self.get_set_parameters(),
        )


class MergeLimitScan(POIScanTask1D):
    def requires(self):
        return LimitScan.req(self)

    def output(self):
        return self.local_target_dc(
            "limits__{}_n{}_{}_{}.npz".format(self.poi, self.points, *self.poi_range)
        )

    def run(self):
        import numpy as np

        records = []
        dtype = [
            (self.poi, np.float32),
            ("limit", np.float32),
            ("limit_p1", np.float32),
            ("limit_m1", np.float32),
            ("limit_p2", np.float32),
            ("limit_m2", np.float32),
        ]
        limit_scan_task = self.requires()
        for branch, inp in self.input()["collection"].targets.items():
            f = inp.load(formatter="uproot")["limit"]
            limits = f.array("limit")
            kl = limit_scan_task.branch_map[branch]
            if len(limits) == 1:
                # only the central limit exists
                # TODO: shouldn't we raise an error when this happens?
                records.append((kl, limits[0], 0.0, 0.0, 0.0, 0.0))
            else:
                # also 1 and 2 sigma variations exist
                records.append((kl, limits[2], limits[3], limits[1], limits[4], limits[0]))

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class LikelihoodScan1D(POIScanTask1D, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def requires(self):
        return CreateWorkspace.req(self)

    def workflow_requires(self):
        reqs = super(LikelihoodScan1D, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def create_branch_map(self):
        import numpy as np

        return np.linspace(self.poi_range[0], self.poi_range[1], self.points).round(7).tolist()

    def output(self):
        return self.local_target_dc("likelihood__{}_{}.root".format(self.poi, self.branch_data))

    def build_command(self):
        corr = 0.5 * (self.poi_range[1] - self.poi_range[0]) / (self.points - 1)
        return (
            "combine -M MultiDimFit {workspace}"
            " -m {self.mass}"
            " -t -1"
            " -v 1"
            " --algo grid"
            " --points {self.points}"
            " --setParameterRanges {self.poi}={start},{stop}"
            " --firstPoint {self.branch}"
            " --lastPoint {self.branch}"
            " --redefineSignalPOIs {self.poi}"
            " --setParameters {self.fixed_params}"
            " --freezeParameters {self.frozen_params}"
            " --robustFit 1"
            " --X-rtd MINIMIZER_analytic"
            " {self.combine_stable_options}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            point=self.branch_data,
            start=self.poi_range[0] - corr,
            stop=self.poi_range[1] + corr,
        )


class MergeLikelihoodScan1D(POIScanTask1D):
    def requires(self):
        return LikelihoodScan1D.req(self)

    def output(self):
        return self.local_target_dc(
            "likelihoods__{}_n{}_{}_{}.npz".format(self.poi, self.points, *self.poi_range)
        )

    def run(self):
        import numpy as np

        records = []
        dtype = [(self.poi, np.float32), ("delta_nll", np.float32)]
        from IPython import embed

        embed()
        for inp in self.input()["collection"].targets.values():
            f = inp.load(formatter="uproot")["limit"]
            records.append((f[self.poi].array()[1], f["deltaNLL"].array()[1]))

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class LikelihoodScan2D(POIScanTask2D, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def requires(self):
        return CreateWorkspace.req(self)

    def workflow_requires(self):
        reqs = super(LikelihoodScan2D, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def create_branch_map(self):
        import numpy as np

        range1 = np.linspace(self.poi1_range[0], self.poi1_range[1], self.points1).round(7).tolist()
        range2 = np.linspace(self.poi2_range[0], self.poi2_range[1], self.points2).round(7).tolist()
        return list(itertools.product(range1, range2))

    def output(self):
        return self.local_target_dc(
            "likelihood__{0}_{2}__{1}_{3}.root".format(self.poi1, self.poi2, *self.branch_data)
        )

    def build_command(self):
        return (
            "combine -M MultiDimFit {workspace}"
            " -m {self.mass}"
            " -t -1"
            " -v 1"
            " --algo grid"
            " --points 1"
            " --setParameterRanges {self.poi1}={point1},{point1}:{self.poi2}={point2},{point2}"
            " --firstPoint 0"
            " --lastPoint 0"
            " --redefineSignalPOIs {self.poi1},{self.poi2}"
            " --setParameters {self.fixed_params}"
            " --freezeParameters {self.frozen_params}"
            " --robustFit 1"
            " --X-rtd MINIMIZER_analytic"
            " {self.combine_stable_options}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            point1=self.branch_data[0],
            point2=self.branch_data[1],
        )


class MergeLikelihoodScan2D(POIScanTask2D):
    def requires(self):
        return LikelihoodScan2D.req(self)

    def output(self):
        return self.local_target_dc(
            "likelihoods__{0}_n{1}_{4}_{5}__{2}_n{3}_{6}_{7}.npz".format(
                self.poi1,
                self.points1,
                self.poi2,
                self.points2,
                *(self.poi1_range + self.poi2_range)
            )
        )

    def run(self):
        import numpy as np

        records = []
        dtype = [
            (self.poi1, np.float32),
            (self.poi2, np.float32),
            ("delta_nll", np.float32),
        ]
        for inp in self.input()["collection"].targets.values():
            f = inp.load(formatter="uproot")["limit"]
            records.append(
                (
                    f[self.poi1].array()[1],
                    f[self.poi2].array()[1],
                    f["deltaNLL"].array()[1],
                )
            )

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class ImpactsPulls(CombineCommandTask):

    r = luigi.FloatParameter(default=1.0, description="injected signal strength; default: 1.0")
    r_range = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(0, 30),
        min_len=2,
        max_len=2,
        description="signal strength range given by two values separate by comma; default: 0,30",
    )

    set_parameters = "--redefineSignalPOIs r --setParameters r_gghh=1,r_qqhh=1,kt=1,kl=1,CV=1,C2V=1"

    run_command_in_tmp = True

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target_dc("impacts.json")

    def build_command(self):
        # TODO: rewrite commands with plain combine
        raise NotImplementedError
        # initial_fit = (
        #     "combineTool.py -M Impacts -d {workspace}"
        #     " -m {mass} -t -1 "
        #     " --cminDefaultMinimizerStrategy 0 --robustFit 1 --X-rtd MINIMIZER_analytic"
        #     " --expectSignal={r} --setParameterRanges r={r_low},{r_high}"
        #     " --doInitialFit"
        #     " --parallel {cores} {params}"
        # ).format(
        #     workspace=self.input().path,
        #     mass=self.mass,
        #     r=self.r,
        #     r_low=self.r_low,
        #     r_high=self.r_high,
        #     cores=multiprocessing.cpu_count(),
        #     params=self.params,
        # )
        # fit = initial_fit.replace("--doInitialFit", "--doFits")
        # impacts = (
        #     "combineTool.py -M Impacts -d {workspace} -m {mass} --output {impacts} {params} --setParameterRanges r={r_low},{r_high}"
        # ).format(
        #     mass=self.mass,
        #     workspace=self.input().path,
        #     impacts=self.output().path,
        #     params=self.params,
        #     r_low=self.r_low,
        #     r_high=self.r_high,
        # )
        # return " && ".join([initial_fit, fit, impacts])
