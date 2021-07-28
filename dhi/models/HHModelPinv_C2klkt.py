###################################
# Author      : L. Cadamuro (UF)
# Date        : 22/04/2020
# Brief       : code that implements the HH model in combine for a C2/kt/kl scan
# Additions by: Marcel Rieger, Fabio Monti, Torben Lange
# structure of the code :
# xxHHSample  -> defines the interface to the user, that will pass the xs and the coupling setups
# xxHHFormula -> implements the matrix component representation, that calculates the symbolic scalings
# HHModel     -> implements the interfaces to combine
###################################


from collections import OrderedDict, defaultdict

import sympy

from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel
from HBRscaler import HBRscaler


####################

class GGFHHSample:
    def __init__(self, val_kl, val_kt, val_C2, val_xs, label):
        self.val_kl  = val_kl
        self.val_kt  = val_kt
        self.val_C2  = val_C2
        self.val_xs  = val_xs
        self.label   = label

####################

#1D formula in case one is interested in a pure C2 scan, kt/kl need to be fixed to one
class GGFHHFormula_1D:
    def __init__(self, sample_list):
        self.sample_list = sample_list
        self.build_matrix()
        self.calculatecoeffients()

    def build_matrix(self):
        """ create the matrix M in this object """

        # the code will combine as many samples as passed to the input
        # into a matrix with 3 columns and Nsamples rows
        nrows = len(self.sample_list)
        ncols = 3
        M_tofill = [[None]*ncols for i in range(nrows)]

        for isample, sample in enumerate(self.sample_list):

            ## implement the 3 scalings - box, triangle, interf
            M_tofill[isample][0] = 1
            M_tofill[isample][1] = sample.val_C2**2
            M_tofill[isample][2] = sample.val_C2

        # print M_tofill
        self.M = sympy.Matrix(M_tofill)

    def calculatecoeffients(self):
        """ create the function sigma and the six coefficients in this object """

        try: self.M
        except AttributeError: self.build_matrix()

        # ##############################################
        kl, kt, C2, box, tri, interf = sympy.symbols('kl kt C2 box tri interf')
        samples_symb = OrderedDict() # order is essential -> OrderedDict
        Nsamples     = self.M.shape[0] #num rows
        for i in range(Nsamples):
            sname = 's%i' % i
            samples_symb[sname] = sympy.Symbol(sname)

        ### the vector of couplings
        c = sympy.Matrix([
            [1]         ,
            [C2**2] ,
            [C2]    ,
        ])

        ### the vector of components
        v = sympy.Matrix([
            [box]   ,
            [tri]   ,
            [interf],
        ])

        ### the vector of samples (i.e. cross sections)
        symb_list = [[sam] for sam in samples_symb.values()]
        s = sympy.Matrix(symb_list)

        ####
        Minv = self.M.pinv()
        self.coeffs = c.transpose() * Minv  # coeffs * s is the sigma, accessing per component gives each sample scaling
        self.sigma = self.coeffs * s

#########################################

# 3d formula, at least 6 samples in the kl/kt/C2 space
class GGFHHFormula_3D:

    def __init__(self, sample_list):
        self.sample_list = sample_list
        self.build_matrix()
        self.calculatecoeffients()

    def build_matrix(self):
        """ create the matrix M in this object """

        # the code will combine as many samples as passed to the input
        # into a matrix with 3 columns and Nsamples rows
        nrows = len(self.sample_list)
        ncols = 6
        M_tofill = [[None]*ncols for i in range(nrows)]

        for isample, sample in enumerate(self.sample_list):
            ## implement the 3 scalings - box, triangle, interf
            M_tofill[isample][0] = sample.val_kt**2 * sample.val_kl**2
            M_tofill[isample][1] = sample.val_kt**4
            M_tofill[isample][2] = sample.val_C2**2
            M_tofill[isample][3] = sample.val_C2 * sample.val_kl * sample.val_kt
            M_tofill[isample][4] = sample.val_C2 * sample.val_kt**2
            M_tofill[isample][5] = sample.val_kl * sample.val_kt**3

        # print M_tofill
        self.M = sympy.Matrix(M_tofill)

    def calculatecoeffients(self):
        """ create the function sigma and the six coefficients in this object """

        try: self.M
        except AttributeError: self.build_matrix()

        # ##############################################
        kl, kt, C2, box, tria, cross, itc, ibc, itb = sympy.symbols('kl kt C2 box tria cross itc ibc itb')
        samples_symb = OrderedDict() # order is essential -> OrderedDict
        Nsamples     = self.M.shape[0] #num rows
        for i in range(Nsamples):
            sname = 's%i' % i
            samples_symb[sname] = sympy.Symbol(sname)

        ### the vector of couplings
        c = sympy.Matrix([
            [kt**2 * kl**2],
            [kt**4],
            [C2**2],
            [C2 * kl * kt],
            [C2 * kt**2],
            [kl * kt**3],
        ])

        ### the vector of components
        v = sympy.Matrix([
            [tria],
            [box],
            [cross],
            [itc],
            [ibc],
            [itb],
        ])

        ### the vector of samples (i.e. cross sections)
        symb_list = [[sam] for sam in samples_symb.values()]
        s = sympy.Matrix(symb_list)

        ####
        Minv = self.M.pinv()
        self.coeffs = c.transpose() * Minv  # coeffs * s is the sigma, accessing per component gives each sample scaling
        self.sigma = self.coeffs * s

#########################################


no_value = object()


class HHModel(PhysicsModel):
    """
    Models the HH production as linear sum of the input components for GGF >= 3 samples for a C2
    scan or >=6 samples for a C2/kt/kl scan. The following physics options are supported:

    - doNNLOscaling (bool)   : Convert ggF HH yields (that are given in NLO by convention) to NNLO.
    - doBRscaling (bool)     : Enable scaling Higgs branching ratios with model parameters.
    - doHscaling (bool)      : Enable scaling single Higgs cross sections with model parameters.
    - doklDependentUnc (bool): Add a theory uncertainty on ggF HH production that depends on model
                               parameters.
    - do3D (bool)            : Configure the model to only do a 3D scan in kl/kt/C2.
    - doProfileX (string)    : Either "flat" to enable the profiling of kappa parameter X with a
      X in {kl,kt,C2}          flat prior, or "gauss,FLOAT" (or "gauss,-FLOAT/+FLOAT") to use a
                               gaussian (asymmetric) prior. In any case, X will be profiled and is
                               hence removed from the list of POIs.

    A string encoded boolean flag is interpreted as *True* when it is either ``"yes"``, ``"true"``
    or ``1`` (case-insensitive).
    """

    R_POIS = ["r", "r_gghh"]
    K_POIS = ["kl", "kt", "C2"]

    def __init__(self, ggf_sample_list, name, do3D=True):
        PhysicsModel.__init__(self)

        self.name = name
        self.klUncName = "THU_HH"

        # names and values of physics options
        self.hh_options = {
            "doNNLOscaling": {"value": True, "is_flag": True},
            "doBRscaling": {"value": True, "is_flag": True},
            "doHscaling": {"value": True, "is_flag": True},
            "doklDependentUnc": {"value": True, "is_flag": True},
            "do3D": {"value": do3D, "is_flag": True},
            "doProfilekl": {"value": None, "is_flag": False},
            "doProfilekt": {"value": None, "is_flag": False},
            "doProfileC2": {"value": None, "is_flag": False},
        }

        # check the sample list and same the formula
        self.check_validity_ggf(ggf_sample_list)
        self.ggf_formula = (GGFHHFormula_3D if do3D else GGFHHFormula_1D)(ggf_sample_list)
        # TODO: the ggf_formula depends in do3D, which is only final after physics options were set,
        #       so we might need to set it lazily

        self.scalingMap = defaultdict(list)

    def set_opt(self, name, value):
        """
        Sets the value of a physics option named *name*, previoulsy registered with
        :py:meth:`register_opt`, to *value*.
        """
        self.hh_options[name]["value"] = value

    def opt(self, name, default=no_value):
        """
        Helper to get the value of a physics option defined by *name* with an optional *default*
        value that is returned when no option with that *name* is registered.
        """
        if name in self.hh_options or default == no_value:
            return self.hh_options[name]["value"]
        else:
            return default

    def setPhysicsOptions(self, options):
        """
        Hook called by the super class to parse physics options received externally, e.g. via
        ``--physics-option`` or ``--PO``.
        """
        # split by "=" and check one by one
        pairs = [opt.split("=", 1) for opt in options if "=" in opt]
        for name, value in pairs:
            if name not in self.hh_options:
                print("[WARNING] unknown physics option '{}'".format(name))
                continue

            opt = self.hh_options[name]
            if opt["is_flag"]:
                # boolean flag
                value = value.lower() in ["yes", "true", "1"]
            else:
                # string value, catch special cases
                value = None if value.lower() in ["", "none"] else value

                opt["value"] = value
                print("[INFO] using model option {} = {}".format(name, value))

        # set all options also as attributes for backwards compatibility
        for name, opt in self.hh_options.items():
            setattr(self, name, opt["value"])

        # check that profile makes sense
        if not self.do3D:
            if self.doProfilekl:
                raise RuntimeError("kl profiling only possible with 3D scan - enable do3D and add at least 6 samples!")
            if self.doProfilekt:
                raise RuntimeError("kt profiling only possible with 3D scan - enable do3D and add at least 6 samples!")

    def check_validity_ggf(self, ggf_sample_list):
        min_samples = 6 if self.opt("do3D") else 3
        if len(ggf_sample_list) < min_samples:
            raise RuntimeError("%s : malformed GGF input sample list - expect at least %d samples" % (self.name, min_samples))
        if not isinstance(ggf_sample_list, list) and not isinstance(ggf_sample_list, tuple):
            raise RuntimeError("%s : malformed GGF input sample list - expect list or tuple" % self.name)
        for s in ggf_sample_list:
            if not isinstance(s, GGFHHSample):
                raise RuntimeError("%s : malformed GGF input sample list - each element must be a GGFHHSample" % self.name)

    def doParametersOfInterest(self):
        ## the model is built with:
        ## r x [GGF]
        ## GGF = r_GGF x [sum samples(kl, kt, C2)]

        # add rate POIs and freeze r_* by default
        self.modelBuilder.doVar("r[1,-20,20]")
        self.modelBuilder.doVar("r_gghh[1,-20,20]")
        self.modelBuilder.out.var("r_gghh").setConstant(True)
        pois = ["r", "r_gghh"]

        # define kappa parameters, SM vlaues and their uniform ranges
        kappas = OrderedDict([
            ("kl", (1, -30, 30)),
            ("kt", (1, -10, 10)),
            ("C2", (0, -5, 5)),
            ("CV", (1, -10, 10)),  # required for CV dependent BR scaling, not a POI
        ])

        # add them
        for name, (sm_value, start, stop) in kappas.items():
            # define the variable
            self.modelBuilder.doVar("{}[{},{},{}]".format(name, sm_value, start, stop))

            # only make it a POI when it is not profile
            do_profile = name == "kl" and bool(self.doProfilekl)
            do_profile |= name == "kt" and bool(self.doProfilekt)
            do_profile |= name == "C2" and bool(self.doProfileC2)
            if not do_profile:
                self.modelBuilder.out.var(name).setConstant(True)
                pois.append(name)

        # define the POI group
        self.modelBuilder.doSet("POI", ",".join(pois))
        print("using POIs {}".format(",".join(pois)))

        # set or redefine the MH variable on which some of the BRs depend
        if not self.options.mass:
            raise Exception("invalid mass value '{}', please provide a valid value using the "
                "--mass option".format(self.options.mass))
        if self.modelBuilder.out.var("MH"):
            self.modelBuilder.out.var("MH").removeRange()
            self.modelBuilder.out.var("MH").setVal(self.options.mass)
        else:
            self.modelBuilder.doVar("MH[%f]" % self.options.mass)
        self.modelBuilder.out.var("MH").setConstant(True)

        # add objects for kl dependent theory uncertainties
        if self.doklDependentUnc:
            self.makeklDepTheoUncertainties()

        # create cross section scaling functions
        self.create_scalings()

    def preProcessNuisances(self, nuisances):
        ''' this method is executed before nuisances are processed'''
        if self.doklDependentUnc:
            nuisances.append((self.klUncName, False, "param", ["0", "1"], []))

        # enable profiling of kappas with a configurable prior
        for name in ["kl", "kt", "C2"]:
            value = getattr(self, "doProfile" + name)
            if not value:
                continue

            # get the prior and add it
            prior, value = value.split(",", 1) if "," in value else (value, None)
            if prior == "flat":
                self.modelBuilder.DC.flatParamNuisances[name] = True
                print("adding flat prior for parameter {}".format(name))
            elif prior == "gauss":
                nuisances.append((name, False, "param", ["1", value, "[-7,7]"], []))
                print("adding gaussian prior for parameter {} with width {}".format(name, value))
            else:
                raise Exception("unknown prior '{}' for parameter {}".format(prior, name))

    def makeInterpolation(self, nameout, nameHi, nameLo, x):
        # as in https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/102x/interface/ProcessNormalization.h
        ## maybe we can try to reuse that should be fast
        d = {"name": nameout, "hi": nameHi, "lo": nameLo, 'x': x}

        d['logKhi']   = "log({hi})".format(**d)
        d['logKlo']   = "-log({lo})".format(**d)
        d['avg']      = "0.5*({logKhi} + {logKlo})".format(**d)
        d['halfdiff'] = "0.5*({logKhi} - {logKlo})".format(**d)
        d["twox"]     = "2*{x}".format(**d)
        d["twox2"]    = "({twox})*({twox})".format(**d)
        d['alpha']    = '0.125 * {twox} * ({twox2} * ({twox2} - 10.) + 15.)'.format(**d)

        d['retCent']  = "{avg}+{alpha}*{halfdiff}".format(**d)
        d['retLow']   = d['logKlo']
        d['retHigh']  = d['logKhi']
        d['retFull']  = "{x} <= -0.5 ? ({retLow}) : {x} >= 0.5 ? ({retHigh}) : ({retCent})".format(**d)

        d['ret'] = 'expr::{name}("exp({retFull})",{{{hi},{lo},{x}}})'.format(**d)

        # print "[DEBUG]","[makeInterpolation]","going to build: ",d['ret']
        self.modelBuilder.factory_(d['ret'])

    def makeklDepTheoUncertainties(self):
        ''' Construct and import uncertanties on the workspace'''
        #upper_unc[kl] = Max[72.0744-51.7362*kl+11.3712*kl2, 70.9286-51.5708*kl+11.4497*kl2] in fb.
        #lower_unc[kl] = Min[66.0621-46.7458*kl+10.1673*kl2, 66.7581-47.721*kl+10.4535*kl2] in fb.
        # if not self.doklDependentUnc: return

        self.modelBuilder.doVar("%s[-7,7]" % self.klUncName)

        self.modelBuilder.factory_('expr::%s_kappaHi("max(76.6075 - 56.4818 * @0 + 12.6350 *@0 * @0, 75.4617 - 56.3164 * @0 + 12.7135 * @0 * @0) / (70.3874 - 50.4111 * @0 + 11.0595 * @0 * @0)", kl)' % self.klUncName)
        self.modelBuilder.factory_('expr::%s_kappaLo("min(57.6809 - 42.9905 * @0 + 9.58474 * @0 * @0, 58.3769 - 43.9657 * @0 + 9.87094 * @0 * @0)  / (70.3874-50.4111 * @0 + 11.0595 * @0 * @0)", kl)' % self.klUncName)

        self.makeInterpolation("%s_kappa" % self.klUncName, "%s_kappaHi" % self.klUncName, "%s_kappaLo" % self.klUncName, self.klUncName)

        ## make scaling
        self.modelBuilder.factory_("expr::scaling_{name}(\"pow(@0,@1)\",{name}_kappa,{name})".format(name=self.klUncName))

    def create_scalings(self):
        """
        Create the functions that scale the >= 6 components of vbf and the >= 3 components of ggf,
        as well as the single Higgs and BR scalings.
        """
        self.HBRscal = HBRscaler(self.modelBuilder, self.doBRscaling, self.doHscaling)
        self.f_r_ggf_names = [] # the RooFormulae that scale the components (GGF)

        def pow_to_mul_string(expr):
            """ Convert integer powers in an expression to Muls, like a**2 => a*a. Returns a string """
            pows = list(expr.atoms(sympy.Pow))
            if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):
                raise ValueError("A power contains a non-integer exponent")
            s = str(expr)
            repl = zip(pows, (sympy.Mul(*([b] * e), evaluate=False) for b, e in (i.as_base_exp() for i in pows)))
            for fr, to in repl:
                s = s.replace(str(fr), str(to))
            return s

        ### loop on the GGF scalings
        sampleList = self.ggf_formula.sample_list
        for i, s in enumerate(sampleList):
            # f_name = 'f_ggfhhscale_sample_{0}'.format(i)
            f_name = 'f_ggfhhscale_sample_{0}'.format(s.label)
            f_expr = self.ggf_formula.coeffs[i] # the function that multiplies each sample
            kl = sympy.symbols('kl')
            #NLO xsec formula
            f_NLO_xsec = '62.5339 - 44.3231*kl + 9.6340*kl*kl'
            #NNLO xsec formula https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGHH#Latest_recommendations_for_gluon
            f_NNLO_xsec = '70.3874 - 50.4111*kl + 11.0595*kl*kl'

            # print f_expr
            # for ROOFit, this will convert expressions as a**2 to a*a
            s_expr = pow_to_mul_string(f_expr)

            couplings_in_expr = []
            if 'kl'  in s_expr: couplings_in_expr.append('kl')
            if 'kt'  in s_expr: couplings_in_expr.append('kt')
            if 'C2'  in s_expr: couplings_in_expr.append('C2')

            # no constant expressions are expected
            if len(couplings_in_expr) == 0:
                raise RuntimeError('GGF HH : scaling expression has no coefficients')

            for idx, ce in enumerate(couplings_in_expr):
                # print '..replacing', ce
                symb = '@{}'.format(idx)
                s_expr = s_expr.replace(ce, symb)

            # embed the scaling due to the xs uncertainty
            if self.doklDependentUnc:
                s_expr = 'scaling_{name} * ({expr})'.format(name=self.klUncName, expr=s_expr)
                couplings_in_expr.append('scaling_{name}'.format(name=self.klUncName))

            if self.doNNLOscaling:
                #print str(f_expr)
                if('kl' not in str(f_expr)): couplings_in_expr.append('kl')

                for idx, ce in enumerate(couplings_in_expr):
                    symb = '@{}'.format(idx)
                    f_NLO_xsec = f_NLO_xsec.replace(ce, symb)
                    f_NNLO_xsec = f_NNLO_xsec.replace(ce, symb)

                arglist = ','.join(couplings_in_expr)
                #this will scale NNLO_xsec
                exprname = 'expr::{}("({}) / (1.115 * ({}) / ({}))" , {})'.format(f_name, s_expr, f_NLO_xsec, f_NNLO_xsec, arglist)
                self.modelBuilder.factory_(exprname) # the function that scales each VBF sample
                #self.modelBuilder.out.function(f_name).Print("")

            else:
                arglist = ','.join(couplings_in_expr)
                exprname = 'expr::{}(\"{}\" , {})'.format(f_name, s_expr, arglist)
                self.modelBuilder.factory_(exprname) # the function that scales each VBF sample
            f_prod_name = f_name + '_r'
            prodname = 'prod::{}(r,{})'.format(f_prod_name, f_name)
            self.modelBuilder.factory_(prodname)  ## the function that scales this production mode
            # self.modelBuilder.out.function(f_prod_name).Print("") ## will just print out the values

            self.f_r_ggf_names.append(f_prod_name) #bookkeep the scaling that has been created

    def getYieldScale(self, bin, process):
        def find_hh_matches(samples, kind):
            # get the matching hh sample index
            matching_indices = []
            for i, sample in enumerate(samples):
                if process.startswith(sample.label):
                    matching_indices.append(i)

            # complain when there is more than one hit
            if len(matching_indices) > 1:
                raise Exception("found {} matches for {} signal process {} in bin {}".format(
                    len(matching_indices), kind, process, bin))

            # return the index when there is only one hit
            if len(matching_indices) == 1:
                return matching_indices[0]

            # otherwise, return nothing
            return None

        # ggf match?
        isample = find_hh_matches(self.ggf_formula.sample_list, "GGF")
        if isample is not None:
            self.scalingMap[process].append((isample, "GGF"))
            scaling = self.f_r_ggf_names[isample]
            # when the BR scaling is enabled, try to extract the decays from the process name
            if self.doBRscaling:
                scaling = self.HBRscal.buildXSBRScalingHH(scaling, bin, process) or scaling
            return scaling

        # complain when the process is a signal but no sample matched
        if self.DC.isSignal[process]:
            raise Exception("HH process {} did not match any GGF or VBF samples in bin {}".format(
                process, bin))

        # single H match?
        if self.doHscaling:
            scaling = self.HBRscal.findSingleHMatch(bin, process)
            # when the BR scaling is enabled, try to extract the decay from the process name
            if scaling and self.doBRscaling:
                scaling = self.HBRscal.buildXSBRScalingH(scaling, bin, process) or scaling
            return scaling or 1.

        # at this point we are dealing with a background process that is also not single-H-scaled,
        # so it is safe to return 1 since any misconfiguration should have been raised already
        return 1.

    def get_formulae(self):
        return {
            "ggf_formula": self.ggf_formula,
        }

    def done(self):
        super(HHModel, self).done()

        # get the labels of ggF samples and store a flag to check if they were matched
        matches_ggf = OrderedDict((s.label, []) for s in self.ggf_formula.sample_list)
        # go through the scaling map and match to samples
        for prefix, matches in [("ggHH", matches_ggf)]:
            for sample_name in self.scalingMap:
                if not sample_name.startswith(prefix + "_"):
                    continue
                for sample_label in matches:
                    if sample_name.startswith(sample_label):
                        matches[sample_label].append(sample_name)
                        break

        # print matches
        matches = list(matches_ggf.items())
        max_len = max(len(label) for label, _ in matches)
        print("Matching signal samples: (sample -> processes)")
        for label, names in matches:
            print("  {}{} -> {}".format(label, " " * (max_len - len(label)), ", ".join(names)))

        # complain about samples that were not matched by any process
        unmatched_ggf_samples = [label for label, names in matches_ggf.items() if not names]
        msg = []
        n_samples = len(self.ggf_formula.sample_list)
        if len(unmatched_ggf_samples) not in [0, n_samples]:
            msg.append("{} ggF signal sample(s) were not matched by any process: {}".format(
                len(unmatched_ggf_samples), ", ".join(unmatched_ggf_samples)))
        if msg:
            raise Exception("\n".join(msg))


# ggf samples with keys (kl, kt, c2)
ggf_samples = OrderedDict([
    ((0,    1, 0),    GGFHHSample(0,    1,    0,    val_xs=0.069725, label="ggHH_kl_0p00_kt_1p00_c2_0p00")),
    ((1,    1, 0),    GGFHHSample(1,    1,    0,    val_xs=0.031047, label="ggHH_kl_1p00_kt_1p00_c2_0p00")),
    ((2.45, 1, 0),    GGFHHSample(2.45, 1,    0,    val_xs=0.013124, label="ggHH_kl_2p45_kt_1p00_c2_0p00")),
    ((5,    1, 0),    GGFHHSample(5,    1,    0,    val_xs=0.091172, label="ggHH_kl_5p00_kt_1p00_c2_0p00")),
    ((0,    1, 1),    GGFHHSample(0,    1,    1,    val_xs=0.155508, label="ggHH_kl_0p00_kt_1p00_c2_1p00")),
    ((1,    1, 0.1),  GGFHHSample(1,    1,    0.1,  val_xs=0.015720, label="ggHH_kl_1p00_kt_1p00_c2_0p10")),
    ((1,    1, 0.35), GGFHHSample(1,    1,    0.35, val_xs=0.011103, label="ggHH_kl_1p00_kt_1p00_c2_0p35")),
    ((1,    1, 3),    GGFHHSample(1,    1,    3,    val_xs=2.923567, label="ggHH_kl_1p00_kt_1p00_c2_3p00")),
    ((1,    1, -2),   GGFHHSample(1,    1,    -2,   val_xs=1.956196, label="ggHH_kl_1p00_kt_1p00_c2_m2p00")),
])


def _get_hh_samples(samples, keys):
    all_keys = list(samples.keys())
    return [
        samples[key if isinstance(key, tuple) else all_keys[key]]
        for key in keys
    ]


def get_ggf_samples(keys):
    """
    Returns a list of :py:class:`GGFHHSample` instances that are mapped to certain *keys* in the
    ordered *ggf_samples* dictionary above. A key can either be a tuple of (kl, kt) values as used
    in the dict, or a numeric index. Example:

    .. code-block:: python

        get_ggf_samples([(2.45, 1), 3])
        # -> [GGFHHSample:ggHH_kl_2p45_kt_1, GGFHHSample:ggHH_kl_5_kt_1]
    """
    return _get_hh_samples(ggf_samples, keys)


def create_model(name, ggf_keys, **kwargs):
    # create the return the model
    return HHModel(ggf_sample_list=get_ggf_samples(ggf_keys), name=name, **kwargs)


model_default = create_model("model_default", ggf_keys=[
    (0, 1, 0), (1, 1, 0), (2.45, 1, 0), (0, 1, 1), (1, 1, 0.35), (1, 1, 3),
])

model_default_1d = create_model("model_default_1d", ggf_keys=[
    (1, 1, 0), (1, 1, 0.35), (1, 1, 3),
], do3D=False)


def create_ggf_xsec_func(formula=None):
    """
    Creates and returns a function that can be used to calculate numeric ggF cross section values in
    pb given an appropriate *formula*, which defaults to *model_default.ggf_formula*. The returned
    function has the signature ``(kl=1.0, kt=1.0, nnlo=True, unc=None)``. When *nnlo* is *False*,
    the constant k-factor is still applied. Otherwise, the returned value is in full
    next-to-next-to-leading order. In this case, *unc* can be set to eiher "up" or "down" to return
    the up / down varied cross section instead where the uncertainty is composed of a *kl* dependent
    scale + mtop uncertainty and an independent PDF uncertainty of 3%.

    Example:

    .. code-block:: python

        get_ggf_xsec = create_ggf_xsec_func()

        print(get_ggf_xsec(kl=2.))
        # -> 0.013803...

        print(get_ggf_xsec(kl=2., nnlo=False))
        # -> 0.013852...

        print(get_ggf_xsec(kl=2., unc="up"))
        # -> 0.014305...

    Formulas are taken from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWGHH?rev=70.
    """
    if formula is None:
        formula = model_default.ggf_formula

    # create the lambdify'ed evaluation function
    n_samples = len(formula.sample_list)
    symbol_names = ["kl", "kt", "C2"] + list(map("s{}".format, range(n_samples)))
    xsec_func = sympy.lambdify(sympy.symbols(symbol_names), formula.sigma)

    # nlo-to-nnlo scaling functions in case nnlo is set
    xsec_nlo = lambda kl: 0.001 * 1.115 * (62.5339 - 44.3231 * kl + 9.6340 * kl**2.)
    xsec_nnlo = lambda kl: 0.001 * (70.3874 - 50.4111 * kl + 11.0595 * kl**2.)
    nlo2nnlo = lambda xsec, kl: xsec * xsec_nnlo(kl) / xsec_nlo(kl)

    # scale+mtop uncertainty in case unc is set
    xsec_nnlo_scale_up = lambda kl: 0.001 * max(
        76.6075 - 56.4818 * kl + 12.635 * kl**2,
        75.4617 - 56.3164 * kl + 12.7135 * kl**2,
    )
    xsec_nnlo_scale_down = lambda kl: 0.001 * min(
        57.6809 - 42.9905 * kl + 9.58474 * kl**2,
        58.3769 - 43.9657 * kl + 9.87094 * kl**2,
    )

    def apply_uncertainty_nnlo(kl, xsec_nom, unc):
        # note on kt: in the twiki linked above, uncertainties on the ggF production cross section
        # are quoted for different kl values but otherwise fully SM parameters, esp. kt=1;
        # however, the nominal cross section *xsec_nom* might be subject to a different kt value
        # and thus, the following implementation assumes that the relative uncertainties according
        # to the SM recommendation are preserved; for instance, if the the scale uncertainty for
        # kl=2,kt=1 would be 10%, then the code below will assume an uncertainty for kl=2,kt!=1 of
        # 10% as well

        # compute the relative, signed scale+mtop uncertainty
        if unc.lower() not in ("up", "down"):
            raise ValueError("unc must be 'up' or 'down', got '{}'".format(unc))
        scale_func = {"up": xsec_nnlo_scale_up, "down": xsec_nnlo_scale_down}[unc.lower()]
        xsec_nom_kt1 = xsec_func(kl, 1, 0, *(sample.val_xs for sample in formula.sample_list))[0, 0]
        xsec_unc = (scale_func(kl) - xsec_nom_kt1) / xsec_nom_kt1

        # combine with flat 3% PDF uncertainty, preserving the sign
        unc_sign = 1 if xsec_unc > 0 else -1
        xsec_unc = unc_sign * (xsec_unc**2. + 0.03**2.)**0.5

        # compute the shifted absolute value
        xsec = xsec_nom * (1. + xsec_unc)

        return xsec

    # wrap into another function to apply defaults and nlo-to-nnlo scaling
    def wrapper(kl=1., kt=1., C2=0., nnlo=True, unc=None):
        xsec = xsec_func(kl, kt, C2, *(sample.val_xs for sample in formula.sample_list))[0, 0]

        # nnlo scaling?
        if nnlo:
            xsec = nlo2nnlo(xsec, kl)

        # apply uncertainties?
        if unc:
            if not nnlo:
                raise NotImplementedError("NLO ggF cross section uncertainties are not implemented")
            xsec = apply_uncertainty_nnlo(kl, xsec, unc)

        return xsec

    return wrapper


def create_hh_xsec_func(ggf_formula=None, vbf_formula=None):
    """
    Creates and returns a function that can be used to calculate numeric HH cross section values in
    pb given the appropriate *ggf_formula* object, which defaults to *model_default.ggf_formula*.
    The returned function has the signature ``(kl=1.0, kt=1.0, C2=1.0, nnlo=True, unc=None)``.

    The *nnlo* setting only affects the ggF component of the cross section. When *nnlo* is *False*,
    the constant k-factor of the ggf calculation is still applied. Otherwise, the returned value is
    in full next-to-next-to-leading order for ggF. *unc* can be set to eiher "up" or "down" to
    return the up / down varied cross section instead where the uncertainty is composed of a *kl*
    dependent scale + mtio uncertainty and an independent PDF uncertainty of 3% for ggF, and a scale
    and pdf+alpha_s uncertainty for VBF. The uncertainties of the ggF and VBF processes are treated
    as uncorrelated.

    Example:

    .. code-block:: python

        get_hh_xsec = create_hh_xsec_func()

        print(get_hh_xsec(kl=2.))
        # -> 0.015226...

        print(get_hh_xsec(kl=2., nnlo=False))
        # -> 0.015275...

        print(get_hh_xsec(kl=2., unc="up"))
        # -> 0.015702...
    """
    # get the particular ggf wrapper
    get_ggf_xsec = create_ggf_xsec_func(ggf_formula)

    # create a combined wrapper with the merged signature
    def wrapper(kl=1., kt=1., C2=0., nnlo=True, unc=None):
        ggf_xsec = get_ggf_xsec(kl=kl, kt=kt, C2=C2, nnlo=nnlo)
        xsec = ggf_xsec

        # apply uncertainties?
        if unc:
            if unc.lower() not in ("up", "down"):
                raise ValueError("unc must be 'up' or 'down', got '{}'".format(unc))

            # ggf uncertainty
            ggf_unc = get_ggf_xsec(kl=kl, kt=kt, C2=C2, nnlo=nnlo, unc=unc) - ggf_xsec
            unc = (1 if unc.lower() == "up" else -1) * (ggf_unc**2.)**0.5
            xsec += unc

        return xsec

    return wrapper


#: Default ggF cross section getter using the formula of the *model_default* model.
get_ggf_xsec = create_ggf_xsec_func(model_default.ggf_formula)

#: Default combined cross section getter using the formulas of the *model_default* model.
get_hh_xsec = create_hh_xsec_func(model_default.ggf_formula)
