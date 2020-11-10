###################################
# Author : L. Cadamuro (UF)
# Date   : 22/04/2020
# Brief  : code that implements the HH model in combine
# structure of the code :
# xxHHSample  -> defines the interface to the user, that will pass the xs and the coupling setups
# xxHHFormula -> implements the matrix component representation, that calculates the symbolic scalings
# HHModel     -> implements the interfaces to combine
###################################


from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from sympy import *
from numpy import matrix
from numpy import linalg
from sympy import Matrix
from collections import OrderedDict
import functools


class VBFHHSample:
    def __init__(self, val_CV, val_C2V, val_kl, val_xs, label):
        self.val_CV  = val_CV
        self.val_C2V = val_C2V
        self.val_kl  = val_kl
        self.val_xs  = val_xs
        self.label   = label

####################

class GGFHHSample:
    def __init__(self, val_kl, val_kt, val_xs, label):
        self.val_kl  = val_kl
        self.val_kt  = val_kt
        self.val_xs  = val_xs
        self.label   = label

####################

class GGFHHFormula:
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
            M_tofill[isample][0] = sample.val_kt**4
            M_tofill[isample][1] = sample.val_kt**2 * sample.val_kl**2
            M_tofill[isample][2] = sample.val_kt**3 * sample.val_kl

        # print M_tofill
        self.M = Matrix(M_tofill)

    def calculatecoeffients(self):
        """ create the function sigma and the six coefficients in this object """

        try: self.M
        except AttributeError: self.build_matrix()

        # ##############################################
        kl, kt, box, tri, interf = symbols('kl kt box tri interf')
        samples_symb = OrderedDict() # order is essential -> OrderedDict
        Nsamples     = self.M.shape[0] #num rows
        for i in range(Nsamples):
            sname = 's%i' % i
            samples_symb[sname] = Symbol(sname)

        ### the vector of couplings
        c = Matrix([
            [kt**4]         ,
            [kt**2 * kl**2] ,
            [kt**3 * kl]    ,
        ])

        ### the vector of components
        v = Matrix([
            [box]   ,
            [tri]   ,
            [interf],
        ])

        ### the vector of samples (i.e. cross sections)
        symb_list = [[sam] for sam in samples_symb.values()]
        s = Matrix(symb_list)

        ####
        Minv   = self.M.pinv()
        self.coeffs = c.transpose() * Minv # coeffs * s is the sigma, accessing per component gives each sample scaling
        self.sigma  = self.coeffs*s

#########################################

class VBFHHFormula:
    def __init__(self, sample_list):
        self.sample_list = sample_list
        self.build_matrix()
        self.calculatecoeffients()

    def build_matrix(self):
        """ create the matrix M in this object """

        # the code will combine as many samples as passed to the input
        # into a matrix with 6 columns and Nsamples rows
        nrows = len(self.sample_list)
        ncols = 6
        M_tofill = [[None]*ncols for i in range(nrows)]

        for isample, sample in enumerate(self.sample_list):

            # implement the 3 scalings - box, triangle, interf
            M_tofill[isample][0] = sample.val_CV**2 * sample.val_kl**2
            M_tofill[isample][1] = sample.val_CV**4
            M_tofill[isample][2] = sample.val_C2V**2
            M_tofill[isample][3] = sample.val_CV**3 * sample.val_kl
            M_tofill[isample][4] = sample.val_CV    * sample.val_C2V    * sample.val_kl
            M_tofill[isample][5] = sample.val_CV**2 * sample.val_C2V

        # print M_tofill
        self.M = Matrix(M_tofill)


    def calculatecoeffients(self):
        """ create the function sigma and the six coefficients in this object """

        try: self.M
        except AttributeError: self.build_matrix()

        ##############################################
        CV, C2V, kl, a, b, c, iab, iac, ibc = symbols('CV C2V kl a b c iab iac ibc')
        samples_symb = OrderedDict() # order is essential -> OrderedDict
        Nsamples     = self.M.shape[0] #num rows
        for i in range(Nsamples):
            sname = 's%i' % i
            samples_symb[sname] = Symbol(sname)

        ### the vector of couplings
        c = Matrix([
            [CV**2 * kl**2] ,
            [CV**4]         ,
            [C2V**2]        ,
            [CV**3 * kl]    ,
            [CV * C2V * kl] ,
            [CV**2 * C2V]
        ])

        ### the vector of components
        v = Matrix([
            [a]   ,
            [b]   ,
            [c]   ,
            [iab] ,
            [iac] ,
            [ibc]
        ])

        ### the vector of samples (i.e. cross sections)
        symb_list = [[sam] for sam in samples_symb.values()]
        s = Matrix(symb_list)

        ####
        Minv   = self.M.pinv()
        self.coeffs = c.transpose() * Minv # coeffs * s is the sigma, accessing per component gives each sample scaling
        self.sigma  = self.coeffs*s

#########################################


class HHModel(PhysicsModel):
    """ Models the HH production as linear sum of the input components for VBF (>= 6) and GGF (>= 3) """

    def __init__(self, ggf_sample_list, vbf_sample_list, name):
        PhysicsModel.__init__(self)
        self.doNNLOscaling = True
        self.name = name

        self.check_validity_ggf(ggf_sample_list)
        self.check_validity_vbf(vbf_sample_list)

        self.ggf_formula = GGFHHFormula(ggf_sample_list)
        self.vbf_formula = VBFHHFormula(vbf_sample_list)

        #self.dump_inputs()

    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("doNNLOscaling="):
                self.doNNLOscaling = (po.replace("doNNLOscaling=","") in [ "yes", "1", "Yes", "True", "true" ])
            #print "[DEBUG]","[setPhysicsOptions]","Uncertainties are set to be",self.doNNLOscaling


    def check_validity_ggf(self, ggf_sample_list):
        if len(ggf_sample_list) < 3:
            raise RuntimeError("%s : malformed GGF input sample list - expect at least 3 samples" % self.name)
        if not isinstance(ggf_sample_list, list) and not isinstance(ggf_sample_list, tuple):
            raise RuntimeError("%s : malformed GGF input sample list - expect list or tuple" % self.name)
        for s in ggf_sample_list:
            if not isinstance(s, GGFHHSample):
                raise RuntimeError("%s : malformed GGF input sample list - each element must be a GGFHHSample" % self.name)

    def check_validity_vbf(self, vbf_sample_list):
        if len(vbf_sample_list) < 6:
            raise RuntimeError("%s : malformed VBF input sample list - expect at least 6 samples" % self.name)
        if not isinstance(vbf_sample_list, list) and not isinstance(vbf_sample_list, tuple):
            raise RuntimeError("%s : malformed VBF input sample list - expect list or tuple" % self.name)
        for s in vbf_sample_list:
            if not isinstance(s, VBFHHSample):
                raise RuntimeError("%s : malformed VBF input sample list - each element must be a VBFHHSample" % self.name)


    def dump_inputs(self):
        print "[INFO]  HH model : " , self.name
        print "......  GGF configuration"
        for i,s in enumerate(self.ggf_formula.sample_list):
            print "        {0:<3} ... kl : {1:<3}, kt : {2:<3}, xs : {3:<3.8f} pb, label : {4}".format(i, s.val_kl, s.val_kt, s.val_xs, s.label)
        print "......  VBF configuration"
        for i,s in enumerate(self.vbf_formula.sample_list):
            print "        {0:<3} ... CV : {1:<3}, C2V : {2:<3}, kl : {3:<3}, xs : {4:<3.8f} pb, label : {5}".format(i, s.val_CV, s.val_C2V, s.val_kl, s.val_xs, s.label)


    def doParametersOfInterest(self):

        ## the model is built with:
        ## r x [GGF + VBF]
        ## GGF = r_GGF x [sum samples(kl, kt)]
        ## VBF = r_VBF x [sum samples(kl, CV, C2V)]

        POIs = "r,r_gghh,r_qqhh,CV,C2V,kl,kt"

        self.modelBuilder.doVar("r[1,-20,20]")
        self.modelBuilder.doVar("r_gghh[1,-20,20]")
        self.modelBuilder.doVar("r_qqhh[1,-20,20]")
        self.modelBuilder.doVar("CV[1,-10,10]")
        self.modelBuilder.doVar("C2V[1,-10,10]")
        self.modelBuilder.doVar("kl[1,-30,30]")
        self.modelBuilder.doVar("kt[1,-10,10]")

        self.modelBuilder.doSet("POI",POIs)

        self.modelBuilder.out.var("r_gghh") .setConstant(True)
        self.modelBuilder.out.var("r_qqhh") .setConstant(True)
        self.modelBuilder.out.var("CV")     .setConstant(True)
        self.modelBuilder.out.var("C2V")    .setConstant(True)
        self.modelBuilder.out.var("kl")     .setConstant(True)
        self.modelBuilder.out.var("kt")     .setConstant(True)

        self.create_scalings()

    def create_scalings(self):
        """ create the functions that scale the >= 6 components of vbf and the >= 3 components of ggf """

        self.f_r_vbf_names = [] # the RooFormulae that scale the components (VBF)
        self.f_r_ggf_names = [] # the RooFormulae that scale the components (GGF)

        def pow_to_mul_string(expr):
            """ Convert integer powers in an expression to Muls, like a**2 => a*a. Returns a string """
            pows = list(expr.atoms(Pow))
            if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):
                raise ValueError("A power contains a non-integer exponent")
            s = str(expr)
            repl = zip(pows, (Mul(*[b]*e,evaluate=False) for b,e in (i.as_base_exp() for i in pows)))
            for fr, to in repl:
                s = s.replace(str(fr), str(to))
            return s

        ### loop on the GGF scalings
        for i, s in enumerate(self.ggf_formula.sample_list):
            # f_name = 'f_ggfhhscale_sample_{0}'.format(i)
            f_name = 'f_ggfhhscale_sample_{0}'.format(s.label)
            f_expr = self.ggf_formula.coeffs[i] # the function that multiplies each sample

            kl = symbols('kl')
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

            # no constant expressions are expected
            if len(couplings_in_expr) == 0:
                raise RuntimeError('GGF HH : scaling expression has no coefficients')

            for idx, ce in enumerate(couplings_in_expr):
                # print '..replacing', ce
                symb = '@{}'.format(idx)
                s_expr = s_expr.replace(ce, symb)

            if(self.doNNLOscaling):
                #print str(f_expr)
                if('kl' not in str(f_expr)): couplings_in_expr.append('kl')

                for idx, ce in enumerate(couplings_in_expr):
                    symb = '@{}'.format(idx)
                    f_NLO_xsec = f_NLO_xsec.replace(ce, symb)
                    f_NNLO_xsec = f_NNLO_xsec.replace(ce, symb)

                arglist = ','.join(couplings_in_expr)
                #this will scale NNLO_xsec
                exprname = 'expr::{}(\"({}) / (1.115 * ({}) / ({}))\" , {})'.format(f_name, s_expr, f_NLO_xsec, f_NNLO_xsec, arglist)
                #print exprname
                self.modelBuilder.factory_(exprname) # the function that scales each VBF sample
                #self.modelBuilder.out.function(f_name).Print("")

            else:
                arglist = ','.join(couplings_in_expr)
                exprname = 'expr::{}(\"{}\" , {})'.format(f_name, s_expr, arglist)
                #print exprname
                self.modelBuilder.factory_(exprname) # the function that scales each VBF sample

            f_prod_name_pmode = f_name + '_r_gghh'
            prodname_pmode = 'prod::{}(r_gghh,{})'.format(f_prod_name_pmode, f_name)
            self.modelBuilder.factory_(prodname_pmode)  ## the function that scales this production mode
            # self.modelBuilder.out.function(f_prod_name).Print("") ## will just print out the values

            f_prod_name = f_prod_name_pmode + '_r'
            prodname = 'prod::{}(r,{})'.format(f_prod_name, f_prod_name_pmode)
            self.modelBuilder.factory_(prodname)  ## the function that scales this production mode
            # self.modelBuilder.out.function(f_prod_name).Print("") ## will just print out the values

            self.f_r_ggf_names.append(f_prod_name) #bookkeep the scaling that has been created

        ### loop on the VBF scalings
        for i, s in enumerate(self.vbf_formula.sample_list):
            # f_name = 'f_vbfhhscale_sample_{0}'.format(i)
            f_name = 'f_vbfhhscale_sample_{0}'.format(s.label)
            f_expr = self.vbf_formula.coeffs[i] # the function that multiplies each sample

            # print f_expr
            # for ROOFit, this will convert expressions as a**2 to a*a
            s_expr = pow_to_mul_string(f_expr)

            couplings_in_expr = []
            if 'CV'  in s_expr: couplings_in_expr.append('CV')
            if 'C2V' in s_expr: couplings_in_expr.append('C2V')
            if 'kl'  in s_expr: couplings_in_expr.append('kl')

            # no constant expressions are expected
            if len(couplings_in_expr) == 0:
                raise RuntimeError('VBF HH : scaling expression has no coefficients')

            for idx, ce in enumerate(couplings_in_expr):
                # print '..replacing', ce
                symb = '@{}'.format(idx)
                s_expr = s_expr.replace(ce, symb)

            arglist = ','.join(couplings_in_expr)
            exprname = 'expr::{}(\"{}\" , {})'.format(f_name, s_expr, arglist)
            # print exprname
            self.modelBuilder.factory_(exprname) # the function that scales each VBF sample

            f_prod_name_pmode = f_name + '_r_qqhh'
            prodname_pmode = 'prod::{}(r_qqhh,{})'.format(f_prod_name_pmode, f_name)
            self.modelBuilder.factory_(prodname_pmode)  ## the function that scales this production mode
            # self.modelBuilder.out.function(f_prod_name_pmode).Print("") ## will just print out the values

            f_prod_name = f_prod_name_pmode + '_r'
            prodname = 'prod::{}(r,{})'.format(f_prod_name, f_prod_name_pmode)
            self.modelBuilder.factory_(prodname)  ## the function that scales this production mode
            # self.modelBuilder.out.function(f_prod_name).Print("") ## will just print out the values

            self.f_r_vbf_names.append(f_prod_name) #bookkeep the scaling that has been created

    def getYieldScale(self,bin,process):

        ## my control to verify for a unique association between process <-> scaling function
        try:
            self.scalingMap
        except AttributeError:
            self.scalingMap = {}

        if not self.DC.isSignal[process]: return 1
        # match the process name in the datacard to the input sample of the calculation
        # this is the only point where the two things must be matched

        if not process in self.scalingMap:
            self.scalingMap[process] = []

        imatched_ggf = []
        imatched_vbf = []

        for isample, sample in enumerate(self.ggf_formula.sample_list):
            if process.startswith(sample.label):
                # print self.name, ": {:>40}  ===> {:>40}".format(process, sample.label)
                imatched_ggf.append(isample)

        for isample, sample in enumerate(self.vbf_formula.sample_list):
            if process.startswith(sample.label):
                # print self.name, ": {:>40}  ===> {:>40}".format(process, sample.label)
                imatched_vbf.append(isample)

        ## this checks that a process finds a unique scaling
        if len(imatched_ggf) + len(imatched_vbf) != 1:
            print "[ERROR] : in HH model named", self.name, "there are", len(imatched_ggf), "GGF name matches and", len(imatched_vbf), "VBF name matches"
            # raise RuntimeError('HHModel : could not uniquely match the process %s to the expected sample list' % process)

        if len(imatched_ggf) == 1:
            isample = imatched_ggf[0]
            self.scalingMap[process].append((isample, 'GGF'))
            return self.f_r_ggf_names[isample]

        if len(imatched_vbf) == 1:
            isample = imatched_vbf[0]
            self.scalingMap[process].append((isample, 'VBF'))
            return self.f_r_vbf_names[isample]

        return 1 ## for signal processes that are not ggHH or qqHH -> to implement here the single H scaling
        # raise RuntimeError('HHModel : fatal error in getYieldScale - this should never happen')


#########################################

# ggf samples with keys (kl, kt), ordered by kl
ggf_samples = OrderedDict([
    ((0, 1),    GGFHHSample(0,    1, val_xs=0.06007, label="ggHH_kl_0_kt_1")),
    ((1, 1),    GGFHHSample(1,    1, val_xs=0.02675, label="ggHH_kl_1_kt_1")),
    ((2.45, 1), GGFHHSample(2.45, 1, val_xs=0.01133, label="ggHH_kl_2p45_kt_1")),
    ((5, 1),    GGFHHSample(5,    1, val_xs=0.07903, label="ggHH_kl_5_kt_1")),

])

# vbf samples with keys (CV, C2V, kl), SM point first, then ordered by kl, then C2V, then CV
vbf_samples = OrderedDict([
    ((1,   1, 1), VBFHHSample(1,   1, 1, val_xs=0.00054 / 0.3364, label="qqHH_CV_1_C2V_1_kl_1")),
    ((1,   1, 0), VBFHHSample(1,   1, 0, val_xs=0.00145 / 0.3364, label="qqHH_CV_1_C2V_1_kl_0")),
    ((1,   1, 2), VBFHHSample(1,   1, 2, val_xs=0.00044 / 0.3364, label="qqHH_CV_1_C2V_1_kl_2")),
    ((1,   0, 1), VBFHHSample(1,   0, 1, val_xs=0.00880 / 0.3364, label="qqHH_CV_1_C2V_0_kl_1")),
    ((1,   2, 1), VBFHHSample(1,   2, 1, val_xs=0.00472 / 0.3364, label="qqHH_CV_1_C2V_2_kl_1")),
    ((0.5, 1, 1), VBFHHSample(0.5, 1, 1, val_xs=0.00353 / 0.3364, label="qqHH_CV_0p5_C2V_1_kl_1")),
    ((1.5, 1, 1), VBFHHSample(1.5, 1, 1, val_xs=0.02149 / 0.3364, label="qqHH_CV_1p5_C2V_1_kl_1")),
])


def get_ggf_samples(keys):
    """
    Returns a list of :py:class:`GGFHHSample` instances mapped to certain *keys* in the ordered
    *ggf_samples* dictionary above. A key can either be a tuple of (kl, kt) values as used in dict,
    or an index. Example:

    .. code-block:: python

        get_ggf_samples([2, (5, 1)])
        # -> [GGFHHSample:ggHH_kl_2p45_kt_1, GGFHHSample:ggHH_kl_5_kt_1]
    """
    keys = [
        (key if isinstance(key, tuple) else ggf_samples.keys()[key])
        for key in keys
    ]
    return [ggf_samples[key] for key in keys]


def get_vbf_samples(keys):
    """
    Returns a list of :py:class:`VBFHHSample` instances mapped to certain *keys* in the ordered
    *vbf_samples* dictionary above. A key can either be a tuple of (CV, C2V, kl) values as used in
    dict, or an index. Example:

    .. code-block:: python

        get_vbf_samples([2, (1, 2, 1)])
        # -> [VBFHHSample:qqHH_CV_1_C2V_1_kl_2, VBFHHSample:qqHH_CV_1_C2V_2_kl_1]
    """
    keys = [
        (key if isinstance(key, tuple) else vbf_samples.keys()[key])
        for key in keys
    ]
    return [vbf_samples[key] for key in keys]


def create_model(name, skip_ggf=None, skip_vbf=None):
    """
    Returns a new :py:class:`HHModel` instance named *name*. Its ggf and vbf sample lists are
    using all availabe samples except those defined in *skip_ggf* and *skip_vbf*, respectively,
    through :py:func:`get_ggf_samples` and :py:func:`get_vbf_samples`. The order of passed keys to
    be skipped does not matter.
    """
    # get all use sample keys
    ggf_keys = list(ggf_samples.keys())
    vbf_keys = list(vbf_samples.keys())
    for key in skip_ggf or []:
        ggf_keys.remove(key)
    for key in skip_vbf or []:
        vbf_keys.remove(key)

    # create the return the model
    return HHModel(
        ggf_sample_list=get_ggf_samples(ggf_keys),
        vbf_sample_list=get_vbf_samples(vbf_keys),
        name=name,
    )


# standard of models
model_all     = create_model("model_all")
model_default = create_model("model_default", skip_ggf=[(0, 1)], skip_vbf=[(1, 0, 1)])
model_bbgg    = create_model("model_bbgg",    skip_ggf=[(0, 1)], skip_vbf=[(0.5, 1, 1)])

# ggf test models
model_no_ggf_kl0    = create_model("model_no_ggf_kl0",    skip_ggf=[(0, 1)],    skip_vbf=[(1, 0, 1)])
model_no_ggf_kl1    = create_model("model_no_ggf_kl1",    skip_ggf=[(1, 1)],    skip_vbf=[(1, 0, 1)])
model_no_ggf_kl2p45 = create_model("model_no_ggf_kl2p45", skip_ggf=[(2.45, 1)], skip_vbf=[(1, 0, 1)])
model_no_ggf_kl5    = create_model("model_no_ggf_kl5",    skip_ggf=[(5, 1)],    skip_vbf=[(1, 0, 1)])

# vbf test models
model_no_vbf_sm    = create_model("model_no_vbf_sm",    skip_ggf=[(0, 1)], skip_vbf=[(1, 1, 1)])
model_no_vbf_kl0   = create_model("model_no_vbf_kl0",   skip_ggf=[(0, 1)], skip_vbf=[(1, 1, 0)])
model_no_vbf_kl2   = create_model("model_no_vbf_kl2",   skip_ggf=[(0, 1)], skip_vbf=[(1, 1, 2)])
model_no_vbf_C2V0  = create_model("model_no_vbf_C2V0",  skip_ggf=[(0, 1)], skip_vbf=[(1, 0, 1)])
model_no_vbf_C2V2  = create_model("model_no_vbf_C2V2",  skip_ggf=[(0, 1)], skip_vbf=[(1, 2, 1)])
model_no_vbf_CV0p5 = create_model("model_no_vbf_CV0p5", skip_ggf=[(0, 1)], skip_vbf=[(0.5, 1, 1)])
model_no_vbf_CV1p5 = create_model("model_no_vbf_CV1p5", skip_ggf=[(0, 1)], skip_vbf=[(1.5, 1, 1)])

# legacy objects for development
GGF_sample_list = [
    # GGFHHSample(0,    1, val_xs=0.06007, label="ggHH_kl_0_kt_1"),
    GGFHHSample(1,    1, val_xs=0.02675, label="ggHH_kl_1_kt_1"),
    GGFHHSample(2.45, 1, val_xs=0.01133, label="ggHH_kl_2p45_kt_1"),
    GGFHHSample(5,    1, val_xs=0.07903, label="ggHH_kl_5_kt_1"),
]
VBF_sample_list = [
    VBFHHSample(1,   1, 1, val_xs=0.00054 / 0.3364, label="qqHH_CV_1_C2V_1_kl_1"),
    VBFHHSample(1,   1, 0, val_xs=0.00145 / 0.3364, label="qqHH_CV_1_C2V_1_kl_0"),
    VBFHHSample(1,   1, 2, val_xs=0.00044 / 0.3364, label="qqHH_CV_1_C2V_1_kl_2"),
    # VBFHHSample(1,   0, 1, val_xs=0.00880 / 0.3364, label="qqHH_CV_1_C2V_0_kl_1"),
    VBFHHSample(1,   2, 1, val_xs=0.00472 / 0.3364, label="qqHH_CV_1_C2V_2_kl_1"),
    VBFHHSample(0.5, 1, 1, val_xs=0.00353 / 0.3364, label="qqHH_CV_0p5_C2V_1_kl_1"),
    VBFHHSample(1.5, 1, 1, val_xs=0.02149 / 0.3364, label="qqHH_CV_1p5_C2V_1_kl_1"),
]
HHdefault = HHModel(
    ggf_sample_list=GGF_sample_list,
    vbf_sample_list=VBF_sample_list,
    name="HHdefault",
)


def create_ggf_xsec_func(formula=None, nnlo=True):
    """
    Creates and returns a function that can be used to calculate numeric ggF cross section values
    given an appropriate *formula*, which defaults to *HHdefault.ggf_formula*. The returned function
    has the signature ``(kl=1.0, kt=1.0)``. When *nnlo* is *True*, a k-factor for the conversion
    from NLO to NNLO is applied. Example:

    .. code-block:: python

        get_ggf_xec = create_ggf_xsec_func()

        print(get_ggf_xec(2.))
        # -> 0.0133045... (or similar)
    """
    if formula is None:
        formula = HHdefault.ggf_formula

    # create the lambdify'ed evaluation function
    func = lambdify(symbols("kl kt s0 s1 s2"), formula.sigma)

    # scaling functions in case nnlo is set
    scale_nlo = lambda kl: 62.5339 - 44.3231 * kl + 9.6340 * kl**2.
    scale_nnlo = lambda kl: 70.3874 - 50.4111 * kl + 11.0595 * kl**2.
    nlo2nnlo = lambda xsec, kl: xsec * scale_nnlo(kl) / (1.115 * scale_nlo(kl))

    # wrap into another function to apply defaults and nlo-to-nnlo scaling
    @functools.wraps(func)
    def wrapper(kl=1., kt=1.):
        xsec = func(kl, kt, *(formula.sample_list[i].val_xs for i in range(3)))[0, 0]

        if nnlo:
            xsec = nlo2nnlo(xsec, kl)

        return xsec

    return wrapper


def create_vbf_xsec_func(formula=None):
    """
    Creates and returns a function that can be used to calculate numeric VBF cross section values
    given an appropriate *formula*, which defaults to *HHdefault.vbf_formula*. The returned function
    has the signature ``(c2v=1.0, cv=1.0, kl=1.0)``.

    .. code-block:: python

        get_vbf_xsec = create_vbf_xsec_func()

        print(get_vbf_xsec(2.))
        # -> 0.0156445... (or similar)
    """
    if formula is None:
        formula = HHdefault.vbf_formula

    # create the lambdify'ed evaluation function
    func = lambdify(symbols("C2V CV kl s0 s1 s2 s3 s4 s5"), formula.sigma)

    # wrap into another function to apply defaults
    @functools.wraps(func)
    def wrapper(c2v=1., cv=1., kl=1.):
        xsec = func(c2v, cv, kl, *(formula.sample_list[i].val_xs for i in range(6)))[0, 0]
        return xsec

    return wrapper


#: Default function for getting ggF cross sections using the formula of the HHdefault model.
get_ggf_xsec = create_ggf_xsec_func()

#: Default function for getting VBF cross sections using the formula of the HHdefault model.
get_vbf_xsec = create_vbf_xsec_func()


# debug
#bbZZ_model = False

#if(bbZZ_model):

#    GGF_sample_list_bbZZ = [
#        GGFHHSample(1,1,   val_xs = 0.02675, label = 'ggHH_hbbhzz_process'  ),
#        GGFHHSample(2,1,   val_xs = 0.01238, label = 'klambda2_process'  ),
        #GGFHHSample(0,1,   val_xs = 0.06007, label = 'klambda0_process'  ),
#        GGFHHSample(5,1,   val_xs = 0.07903, label = 'klambda5_process'  ),
        # GGFHHSample(2.45,1,   val_xs = 0.01133, label = 'ggHH_kl_2p45_kt_1'  ),
#    ]

#    HHbbZZ4l = HHModel(
#        ggf_sample_list = GGF_sample_list_bbZZ,
#        vbf_sample_list = VBF_sample_list,
#        name            = 'HH_bbZZ_default'
#    )

# f = GGFHHFormula(GGF_sample_list)
# f = VBFHHFormula(VBF_sample_list)
