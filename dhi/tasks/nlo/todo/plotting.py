# coding: utf-8

import re

import law
import luigi

from dhi.tasks.base import AnalysisTask
from dhi.tasks.nlo.inference import (
    DCBase,
    NLOBase1D,
    NLOBase2D,
    NLOLimit,
    MergeScans1D,
    MergeScans2D,
    ImpactsPulls,
)
from dhi.tasks.nlo.mixins import (
    PlotMixin,
    ScanMixin,
    LabelsMixin,
    NLL1DMixin,
    NLL2DMixin,
    ViewMixin,
)
from dhi.util import rgb


class PlotScan(ScanMixin, LabelsMixin, ViewMixin, NLOBase1D):
    def requires(self):
        return NLOLimit.req(self)

    def output(self):
        return self.local_target_dc("scan.pdf")

    @ViewMixin.view_output_plots
    def run(self):
        import numpy as np

        self.output().parent.touch()
        inputs = {
            int(re.findall(r"-?\d+", k.basename)[0]): k
            for k in self.input()["collection"].targets.values()
        }

        data = [
            [kl] + [data[i] for i in range(-2, 3)]
            for kl, data in (
                (key, {int(k[3:]): v for k, v in inp.load()["125.0"].items()})
                for key, inp in sorted(inputs.items(), key=lambda x: x)
            )
        ]
        arr = np.array(data)
        self.plot(arr=arr)


class PlotNLL1D(NLL1DMixin, LabelsMixin, ViewMixin, NLOBase1D):
    def requires(self):
        return MergeScans1D.req(self)

    def output(self):
        return self.local_target_dc("nll.pdf")

    @ViewMixin.view_output_plots
    def run(self):
        import numpy as np

        inp = self.input().load()
        poi = inp["poi"]
        deltaNLL = 2 * inp["deltaNLL"]
        self.plot(poi=poi, deltaNLL=deltaNLL)


class PlotNLL2D(NLL2DMixin, LabelsMixin, ViewMixin, NLOBase2D):
    def requires(self):
        return MergeScans2D.req(self)

    def output(self):
        return self.local_target_dc("nll.pdf")

    @ViewMixin.view_output_plots
    def run(self):
        import numpy as np

        inp = self.input().load()
        poi1 = inp["poi1"]
        poi2 = inp["poi2"]
        deltaNLL = 2 * inp["deltaNLL"]
        self.plot(poi1=poi1, poi2=poi2, deltaNLL=deltaNLL)


class PlotImpacts(ViewMixin, DCBase):
    def requires(self):
        return ImpactsPulls.req(self)

    def output(self):
        return self.local_target_dc("impacts.pdf")

    @property
    def cmd(self):
        return "plotImpacts.py -i {impacts} --output {out} ".format(
            impacts=self.input().path, out=self.output().basename.split(".")[0]
        )

    @ViewMixin.view_output_plots
    def run(self):
        super(PlotImpacts, self).run()


class TestPlots(ViewMixin, AnalysisTask, law.WrapperTask):
    def requires(self):
        return [
            PlotScan.req(self),
            PlotNLL1D.req(self),
            PlotNLL2D.req(self),
        ]
