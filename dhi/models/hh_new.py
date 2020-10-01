###################################
# Author : L. Cadamuro (UF)
# Date   : 11/03/2020
# Brief  : code that implements the HH model in combine
# structure of the code :
# xxHHSample  -> defines the interface to the user, that will pass the xs and the coupling setups
# xxHHFormula -> implements the matrix component representation, that calculates the symbolic scalings
# HHModel     -> implements the interfaces to combine
###################################

import os
from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import SM_HIGG_DECAYS
from HiggsAnalysis.CombinedLimit.PhysicsModel import SM_HIGG_PROD
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
from sympy import *
from numpy import matrix
from numpy import linalg
from sympy import Matrix

processSource_dict= {
    "ttH":["tth"],
    "ggH":["ggh","ggF"],
    "qqH":["qqh","VBF"],
    "VH":["vh"],
    "ZH":["zh"],
    "WH":["wh"],
    "tHq":["thq","THQ"],
    "tHW":["thw","THW"],
    "ggHH":["gghh"],
    "qqHH":["qqhh"],
}


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

class VBFHHFormula:
    def __init__(self, sample_list):
        self.sample_list = sample_list
        self.build_matrix()
        self.calculatecoeffients()

    def build_matrix(self):
        """ create the matrix M in this object """

        if len(self.sample_list) != 6:
            print "[ERROR] : expecting 6 samples in input"
            raise RuntimeError("malformed vbf input sample list")
        M_tofill = [
            [None,None,None,None,None,None],
            [None,None,None,None,None,None],
            [None,None,None,None,None,None],
            [None,None,None,None,None,None],
            [None,None,None,None,None,None],
            [None,None,None,None,None,None]
        ]

        for isample, sample in enumerate(self.sample_list):
            # print isample, " CV, C2V, kl = ", sample.val_CV, sample.val_C2V, sample.val_kl

            ## implement the 6 scalings
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
        CV, C2V, kl, a, b, c, iab, iac, ibc, s1, s2, s3, s4, s5, s6 = symbols('CV C2V kl a b c iab iac ibc s1 s2 s3 s4 s5 s6')
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
        s = Matrix([
            [s1] ,
            [s2] ,
            [s3] ,
            [s4] ,
            [s5] ,
            [s6]
        ])
        ####
        Minv   = self.M.inv()
        self.coeffs = c.transpose() * Minv # coeffs * s is the sigma, accessing per component gives each sample scaling
        self.sigma  = self.coeffs*s

####################

class GGFHHFormula:
    def __init__(self, sample_list):
        self.sample_list = sample_list
        self.build_matrix()
        self.calculatecoeffients()

    def build_matrix(self):
        """ create the matrix M in this object """

        if len(self.sample_list) != 3:
            print "[ERROR] : expecting 3 samples in input"
            raise RuntimeError("malformed ggf input sample list")
        M_tofill = [
            [None,None,None],
            [None,None,None],
            [None,None,None],
        ]

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

        ##############################################
        kl, kt, box, tri, interf, s1, s2, s3 = symbols('kl kt box tri interf s1 s2 s3')

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
        s = Matrix([
            [s1] ,
            [s2] ,
            [s3]
        ])
        ####
        Minv   = self.M.inv()
        self.coeffs = c.transpose() * Minv # coeffs * s is the sigma, accessing per component gives each sample scaling
        self.sigma  = self.coeffs*s

####################

class HHModel(PhysicsModel):
    """ Models the HH production as linear sum of 6 components (VBF) and 3 components (GGF) """

    #scaleBR=True to scale for branching ratio dependence
    #setBR1 to overwrite the single Higgs decay mode and one of the double Higgs decay modes
    #setBR2 to overwrite the second decay mode of double Higgs
    def __init__(self, ggf_sample_list, vbf_sample_list, name, setBR1="", setBR2=""):
        PhysicsModel.__init__(self)

        self.name            = name

        self.check_validity_ggf(ggf_sample_list)
        self.check_validity_vbf(vbf_sample_list)

        self.ggf_formula = GGFHHFormula(ggf_sample_list)
        self.vbf_formula = VBFHHFormula(vbf_sample_list)

        self.setBR1 = setBR1
        self.setBR2 = setBR2
        self.scaleBR = True

        # include branching ratio uncertainties in the systematics
        # if only one single HH channel is considered, it is suggested to
        # include the specific BR uncertainty(s) directly in the datacard.
        # This should be re-evaluated for a HH combination
        self.doBRU = False

        self.dump_inputs()
        self.f_r_singleH_names=[]

    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("BRU="):
                self.doBRU = (po.replace("BRU=","") in [ "yes", "1", "Yes", "True", "true" ])
            if po.startswith("scaleBR="):
                self.scaleBR = (po.replace("BRU=","") in [ "no", "0", "No", "False", "false" ])

        print "BR uncertainties in partial widths: %s " % self.doBRU


    def check_validity_ggf(self, ggf_sample_list):
        if len(ggf_sample_list) != 3:
            raise RuntimeError("%s : malformed GGF input sample list - expect 3 samples" % self.name)
        if not isinstance(ggf_sample_list, list) and not isinstance(ggf_sample_list, tuple):
            raise RuntimeError("%s : malformed GGF input sample list - expect list or tuple" % self.name)
        for s in ggf_sample_list:
            if not isinstance(s, GGFHHSample):
                raise RuntimeError("%s : malformed GGF input sample list - each element must be a GGFHHSample" % self.name)

    def check_validity_vbf(self, vbf_sample_list):
        if len(vbf_sample_list) != 6:
            raise RuntimeError("%s : malformed VBF input sample list - expect 6 samples" % self.name)
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

        self.modelBuilder.doVar("r[1,0,10]")
        self.modelBuilder.doVar("r_gghh[1,0,10]")
        self.modelBuilder.doVar("r_qqhh[1,0,10]")
        self.modelBuilder.doVar("CV[1,-10,10]")
        self.modelBuilder.doVar("C2V[1,-10,10]")
        self.modelBuilder.doVar("kl[1,-30,30]")
        self.modelBuilder.doVar("kt[1,-10,10]")

        self.modelBuilder.doSet("POI",POIs)
        self.SMH = SMHiggsBuilder(self.modelBuilder,
            datadir=os.path.expandvars("$DHI_SOFTWARE/HiggsAnalysis/CombinedLimit/data/lhc-hxswg"))

        self.modelBuilder.out.var("r_gghh") .setConstant(True)
        self.modelBuilder.out.var("r_qqhh") .setConstant(True)
        self.modelBuilder.out.var("CV")     .setConstant(True)
        self.modelBuilder.out.var("C2V")    .setConstant(True)
        self.modelBuilder.out.var("kl")     .setConstant(True)
        self.modelBuilder.out.var("kt")     .setConstant(True)

        #I need to build MH variables because the BR are tabulated as a function of MH
        # the mass setting must be provided as input, i.e. '-m 125'
        if self.modelBuilder.out.var("MH"):
            self.modelBuilder.out.var("MH").setVal(self.options.mass)
            self.modelBuilder.out.var("MH").setConstant(True)
        else:
            self.modelBuilder.doVar("MH[%g]" % self.options.mass)

        self.create_scalings()

    def create_scalings(self):
        """ create the functions that scale the six components of vbf and the 3 components of ggf """

        ######################################################################
        #create Higgs BR scalings
        for d in SM_HIGG_DECAYS + [ "hss" ]:
            self.SMH.makeBR(d)

        # BR uncertainties
        if self.doBRU:
            self.SMH.makePartialWidthUncertainties()
        else:
            for d in SM_HIGG_DECAYS:
                self.modelBuilder.factory_('HiggsDecayWidth_UncertaintyScaling_%s[1.0]' % d)

        # fix to have all BRs add up to unity
        self.modelBuilder.factory_("sum::c7_SMBRs(%s)" %  (",".join("SM_BR_"+X for X in "hzz hww htt hmm hcc hbb hss hgluglu hgg hzg".split())))
        self.modelBuilder.out.function("c7_SMBRs").Print("")

        # get VBF, tHq, tHW, ggZH cross section and resolved loops
        self.SMH.makeScaling('qqH', CW='CV', CZ='CV')
        self.SMH.makeScaling("tHq", CW='CV', Ctop="kt")
        self.SMH.makeScaling("tHW", CW='CV', Ctop="kt")
        self.SMH.makeScaling("ggZH",CZ='CV', Ctop="kt",Cb="1")
        self.SMH.makeScaling('ggH', Cb='1', Ctop='kt', Cc="1")
        self.SMH.makeScaling('hgluglu', Cb='1', Ctop='kt')
        self.SMH.makeScaling('hgg', Cb='1', Ctop='kt', CW='CV', Ctau='1')
        self.SMH.makeScaling('hzg', Cb='1', Ctop='kt', CW='CV', Ctau='1')


        cGammap = {"hgg":0.49e-2,"hzz":0.83e-2,"hww":0.73e-2,"hgluglu":0.66e-2,"htt":0,"hbb":0,"hcc":0,"hmm":0}

        # First we need to create the terms that account for the self-coupling --> Just scale partial width first - https://arxiv.org/abs/1709.08649 Eq 22.
        # probably a better way to code this since the partial width expressions are being repeated when we write the BR
        for dec in cGammap.keys():
            valC1 = cGammap[dec]
            self.modelBuilder.factory_('expr::kl_scalBR_%s("(@0-1)*%g",kl)' % (dec,valC1))

        # next make the partial widths, also including the kappas -> we want to include the term from the normal kappas and the one from the self-coupling
        self.modelBuilder.factory_('expr::kVktkl_Gscal_Z("(@0*@0+@3)*@1*@2", CV, SM_BR_hzz, HiggsDecayWidth_UncertaintyScaling_hzz, kl_scalBR_hzz)')
        self.modelBuilder.factory_('expr::kVktkl_Gscal_W("(@0*@0+@3)*@1*@2", CV, SM_BR_hww, HiggsDecayWidth_UncertaintyScaling_hww, kl_scalBR_hww)')
        self.modelBuilder.factory_('expr::kVktkl_Gscal_tau("(1+@4)*@0*@2 + (1+@5)*@1*@3", SM_BR_htt, SM_BR_hmm, HiggsDecayWidth_UncertaintyScaling_htt, HiggsDecayWidth_UncertaintyScaling_hmm,kl_scalBR_htt, kl_scalBR_hmm)')
        self.modelBuilder.factory_('expr::kVktkl_Gscal_top("(1+@2)*@0*@1", SM_BR_hcc, HiggsDecayWidth_UncertaintyScaling_hcc, kl_scalBR_hcc)')
        self.modelBuilder.factory_('expr::kVktkl_Gscal_bottom("(1+@3) * (@0*@2+@1)", SM_BR_hbb, SM_BR_hss, HiggsDecayWidth_UncertaintyScaling_hbb, kl_scalBR_hbb)')
        self.modelBuilder.factory_('expr::kVktkl_Gscal_gluon("  (@0+@3)  * @1 * @2", Scaling_hgluglu, SM_BR_hgluglu, HiggsDecayWidth_UncertaintyScaling_hgluglu, kl_scalBR_hgluglu)')
        self.modelBuilder.factory_('expr::kVktkl_Gscal_gamma("(@0+@6)*@1*@4 + @2*@3*@5",  Scaling_hgg, SM_BR_hgg, Scaling_hzg, SM_BR_hzg, HiggsDecayWidth_UncertaintyScaling_hgg, HiggsDecayWidth_UncertaintyScaling_hzg, kl_scalBR_hgg)') # no kl dependance on H->zg known yet ?
        # fix to have all BRs add up to unity
        self.modelBuilder.factory_("sum::kVktkl_SMBRs(%s)" %  (",".join("SM_BR_"+X for X in "hzz hww htt hmm hcc hbb hss hgluglu hgg hzg".split())))
        self.modelBuilder.out.function("kVktkl_SMBRs").Print("")

        ## total witdh, normalized to the SM one (just the sum over the partial widths/SM total BR)
        self.modelBuilder.factory_('expr::kVktkl_Gscal_tot("(@0+@1+@2+@3+@4+@5+@6)/@7", kVktkl_Gscal_Z, kVktkl_Gscal_W, kVktkl_Gscal_tau, kVktkl_Gscal_top, kVktkl_Gscal_bottom, kVktkl_Gscal_gluon, kVktkl_Gscal_gamma, kVktkl_SMBRs)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM)
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hww("(@0*@0+@3)*@2/@1", CV, kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hww, kl_scalBR_hww)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hzz("(@0*@0+@3)*@2/@1", CV, kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hzz, kl_scalBR_hzz)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_htt("(1+@2)*@1/@0", kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_htt, kl_scalBR_htt)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hmm("(1+@2)*@1/@0", kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hmm, kl_scalBR_hmm)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hbb("(1+@2)*@1/@0", kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hbb, kl_scalBR_hbb)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hcc("(1+@2)*@1/@0", kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hcc, kl_scalBR_hcc)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hgg("(@0+@3)*@2/@1", Scaling_hgg, kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hgg,kl_scalBR_hgg)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hzg("@0*@2/@1", Scaling_hzg, kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hzg)')
        self.modelBuilder.factory_('expr::kVktkl_BRscal_hgluglu("(@0+@3)*@2/@1", Scaling_hgluglu, kVktkl_Gscal_tot, HiggsDecayWidth_UncertaintyScaling_hgluglu, kl_scalBR_hgluglu)')

        ######################################################################
        #create single Higgs production scalings
        energy="13TeV"
        cXSmap_13  = {"ggH":0.66e-2,"qqH":0.64e-2,"WH":1.03e-2,"ZH":1.19e-2,"ttH":3.51e-2,"VH":(0.5*(1.03e-2+1.19e-2))}
        EWKmap_13  = {"ggH":1.049,"qqH":0.932,"WH":0.93,"ZH":0.947,"ttH":1.014,"VH":(0.5*(0.93+0.947))}
        dZH = -1.536e-3

        for production in SM_HIGG_PROD:

            if production in [ "ggZH", "tHq", "tHW"]:
                self.f_r_singleH_names.append("Scaling_%s_%s"%(production,energy))

            elif production in [ "ggH", "qqH" ]:
                EWK = EWKmap_13[production]
                self.modelBuilder.factory_("expr::kVktkl_XSscal_%s_%s(\"(@1+(@0-1)*%g/%g)/((1-(@0*@0-1)*%g))\",kl,Scaling_%s_%s)"\
                                           %(production,energy,cXSmap_13[production],EWK,dZH,production,energy))
                self.f_r_singleH_names.append("kVktkl_XSscal_%s_%s" %(production,energy) )

            elif production in [ "ZH", "WH", "VH"]:
                EWK = EWKmap_13[production]
                self.modelBuilder.factory_("expr::kVktkl_XSscal_%s_%s(\"(@1*@1+(@0-1)*%g/%g)/((1-(@0*@0-1)*%g))\",kl,CV)"\
                                           %(production,energy,cXSmap_13[production],EWK,dZH))
                self.f_r_singleH_names.append("kVktkl_XSscal_%s_%s" %(production,energy) )

            elif production == "ttH":
                EWK = EWKmap_13[production]
                self.modelBuilder.factory_("expr::kVktkl_XSscal_%s_%s(\"(@1*@1+(@0-1)*%g/%g)/((1-(@0*@0-1)*%g))\",kl,kt)"\
                                           %(production,energy,cXSmap_13[production],EWK,dZH))
                self.f_r_singleH_names.append("kVktkl_XSscal_%s_%s" %(production,energy) )


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
            f_name = 'f_ggfhhscale_sample_{0}'.format(i)
            f_expr = self.ggf_formula.coeffs[i] # the function that multiplies each sample

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

            arglist = ','.join(couplings_in_expr)
            exprname = 'expr::{}(\"{}\" , {})'.format(f_name, s_expr, arglist)
            # print exprname
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
            f_name = 'f_vbfhhscale_sample_{0}'.format(i)
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

        if not self.DC.isSignal[process]: return 1
        processSource = process.split("_")[0]

        # It might happen that I use a different process naming than the default one
        # In that case I will convert it to the default naming
        if not processSource in processSource_dict:
            for defaultname, equivalentnames in processSource_dict.items():
                for equivalentname in equivalentnames:
                    if processSource==equivalentname:
                        print "[WARNING]: process name \"%s\" is not the default name but I found it equivalent to \"%s\""%(processSource,defaultname)
                        print "           --> if this is not the case fix the naming!"
                        processSource=defaultname


        decaySource = []
        if self.setBR1!="":
            decaySource.append(self.setBR1)
        else:
            decaySource.append(process.split("_")[-1])

        #in case of HH I have to consider an additional Higgs decay
        if processSource == "ggHH" or processSource == "qqHH":
            if self.setBR2!="":
                decaySource.append(self.setBR2)
            else:
                decaySource.append(process.split("_")[-2])

        if not (processSource == "ggHH" or processSource == "qqHH"):
            if(processSource == "bbH"): return 1 #neglect bbH dependence on kl
            for XS_scaling_name in self.f_r_singleH_names:
                if processSource in XS_scaling_name:
                    if not self.modelBuilder.out.function("kVktkl_BRscal_"+decaySource[0]):
                        raise RuntimeError, "Decay mode %s not supported" % decaySource[0]
                    if not self.scaleBR:
                        return XS_scaling_name
                    else:
                        BR_scaling_name =   "kVktkl_BRscal_"+decaySource[0]
                        XSBR_scaling_name = "kVktkl_XSBRscal_%s_%s" % (processSource,decaySource[0])
                        if not self.modelBuilder.out.function(XSBR_scaling_name):
                            self.modelBuilder.factory_("expr::%s(\"0.+@0*(@0>0.)*@1\",%s,%s)"%(XSBR_scaling_name,XS_scaling_name,BR_scaling_name))#I used a trick to avoid negative XS reweights
                        return XSBR_scaling_name

        #if I am here I have a double H process or an unsopported process
        ## my control to verify for a unique association between process <-> scaling function
        try:
            self.scalingMap
        except AttributeError:
            self.scalingMap = {}

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
            raise RuntimeError('HHModel : could not uniquely match the process %s to the expected sample list' % process)

        if len(imatched_ggf) == 1:
            isample = imatched_ggf[0]
            self.scalingMap[process].append((isample, 'GGF'))
            XS_scaling_name = self.f_r_ggf_names[isample]
            if not self.scaleBR:
                return XS_scaling_name
            else:
                BR1_scaling_name = "kVktkl_BRscal_"+decaySource[0]
                BR2_scaling_name = "kVktkl_BRscal_"+decaySource[1]
                if not self.modelBuilder.out.function(BR1_scaling_name):
                    raise RuntimeError, "Decay mode %s not supported" %BR1_scaling_name
                if not self.modelBuilder.out.function(BR2_scaling_name):
                    raise RuntimeError, "Decay mode %s not supported" %BR2_scaling_name
                XSBR_scaling_name = "%s_%s_%s"%(XS_scaling_name,decaySource[0],decaySource[1])
                if not self.modelBuilder.out.function(XSBR_scaling_name):
                    self.modelBuilder.factory_('expr::%s("@0*@1*@2",%s,%s,%s)'%(XSBR_scaling_name,XS_scaling_name,BR1_scaling_name,BR2_scaling_name))
                return XSBR_scaling_name

        if len(imatched_vbf) == 1:
            isample = imatched_vbf[0]
            self.scalingMap[process].append((isample, 'VBF'))
            XS_scaling_name = self.f_r_vbf_names[isample]
            if not self.scaleBR:
                return XS_scaling_name
            else:
                BR1_scaling_name = "kVktkl_BRscal_"+decaySource[0]
                BR2_scaling_name = "kVktkl_BRscal_"+decaySource[1]
                if not self.modelBuilder.out.function(BR1_scaling_name):
                    raise RuntimeError, "Decay mode %s not supported" %BR1_scaling_name
                if not self.modelBuilder.out.function(BR2_scaling_name):
                    raise RuntimeError, "Decay mode %s not supported" %BR2_scaling_name
                XSBR_scaling_name = "%s_%s_%s"%(XS_scaling_name,decaySource[0],decaySource[1])
                if not self.modelBuilder.out.function(XSBR_scaling_name):
                    self.modelBuilder.factory_('expr::%s("@0*@1*@2",%s,%s,%s)'%(XSBR_scaling_name,XS_scaling_name,BR1_scaling_name,BR2_scaling_name))
                return XSBR_scaling_name

        raise RuntimeError('HHModel : fatal error in getYieldScale - this should never happen')


    def done(self):
        ## this checks that a scaling has been attached to a unique process
        scalings = {}
        for k, i in self.scalingMap.items(): ## key -> process, item -> [(isample, 'type')]
            samples = list(set(i)) # remove duplicates
            for s in samples:
                if not s in scalings:
                    scalings[s] = []
                scalings[s].append(k)

        for key, val in scalings.items():
            if len(val) > 1:
                print "[WARNING] : in HH model named", self.name, "there is a double assignment of a scaling : ", key, " ==> ", val
                #raise RuntimeError('HHModel : coudl not uniquely match the scaling to the process')

        ## now check that, if a VBF/GGF scaling exists, there are actually 6/3 samples in the card
        n_VBF = 0
        n_GGF = 0
        for k, i in self.scalingMap.items():
            # the step above ensured me that the list contains a single element -> i[0]
            if i[0][1] == "GGF":
                n_GGF += 1
            elif i[0][1] == "VBF":
                n_VBF += 1
            else:
                raise RuntimeError("HHModel : unrecognised type %s - should never happen" % i[0][1])

        if n_GGF > 0 and n_GGF < 3:
            raise RuntimeError("HHModel : you did not pass all the 3 samples needed to build the GGF HH model")

        if n_VBF > 0 and n_VBF < 6:
            raise RuntimeError("HHModel : you did not pass all the 6 samples needed to build the VBF HH model")

########################################################
# definition of the model inputs

## NOTE : the scaling functions are not sensitive to global scalings of the xs of each sample
## so by convention use val_xs of the samples *without* the decay BR

## NOTE 2 : the xs *must* correspond to the generator one

# VBF : val_CV, val_C2V, val_kl

VBF_sample_list = [
    VBFHHSample(1,1,1,   val_xs = 0.00054/(0.3364), label = 'qqHH_CV_1_C2V_1_kl_1'  ),
    VBFHHSample(1,2,1,   val_xs = 0.00472/(0.3364), label = 'qqHH_CV_1_C2V_2_kl_1'  ),
    VBFHHSample(1,1,2,   val_xs = 0.00044/(0.3364), label = 'qqHH_CV_1_C2V_1_kl_2'  ),
    VBFHHSample(1,1,0,   val_xs = 0.00145/(0.3364), label = 'qqHH_CV_1_C2V_1_kl_0'  ),
    VBFHHSample(0.5,1,1, val_xs = 0.00353/(0.3364), label = 'qqHH_CV_0p5_C2V_1_kl_1'),
    VBFHHSample(1.5,1,1, val_xs = 0.02149/(0.3364), label = 'qqHH_CV_1p5_C2V_1_kl_1')
]

# VBF : val_kl, val_kt
GGF_sample_list = [
    GGFHHSample(1,1,   val_xs = 0.02675, label = 'ggHH_kl_1_kt_1'  ),
    GGFHHSample(0,1,   val_xs = 0.06007, label = 'ggHH_kl_0_kt_1'  ),
    GGFHHSample(5,1,   val_xs = 0.07903, label = 'ggHH_kl_5_kt_1'  )
    #GGFHHSample(2.45,1,   val_xs = 0.01133, label = 'ggHH_kl_2p45_kt_1'  )
]

HHdefault = HHModel(
    ggf_sample_list = GGF_sample_list,
    vbf_sample_list = VBF_sample_list,
    name            = 'HHdefault',
    setBR1       = "",#setBR1       = "hgg",
    setBR2       = "" #setBR2       = "hbb"
)

# g = GGFHHFormula(GGF_sample_list)
# print g.sigma
# print g.sigma.evalf(subs = {
#     kl : 2.45,
#     kt : 1.0,
#     s1 : s.sample_list[0].val_xs,
#     s2 : s.sample_list[1].val_xs,
#     s3 : s.sample_list[2].val_xs,
# })
