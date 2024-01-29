# coding: utf-8

"""
Custom HH physics model implementing gluon gluon fusion (ggf / gghh), vector boson fusion
(vbf / qqhh) and V boson associated production (vhh) modes.

Authors:
  - Aravind Sugunan
  - Kajari Mazumdar

Developments to come:

- implement r_hhh
- add kt in description (might need mode samples)
- retest the basis stability
- integration with HH model, can be already with the kl-kt-c2 one for definitiveness 
  - be sure that c3 = kl without any factors floating
- integration H BR scaling 
- integration with single H scaling (placeholder for when we add kt to modelling to HHH part)

"""

from sympy import *
from numpy import matrix
from numpy import linalg
from sympy import Matrix
from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from collections import OrderedDict, defaultdict

class HHHSample:
    def __init__(self, val_c3, val_d4, val_xs, label):
        self.val_c3  = val_c3
        self.val_d4  = val_d4
        self.val_xs  = val_xs
        self.label   = label

####################
class HHHFormula:
    def __init__(self, sample_list):
        self.sample_list = sample_list
        self.build_matrix()
        self.sigmaEval=None
        self.calculatecoeffients()

    def build_matrix(self):
        """ create the matrix M in this object """

        if len(self.sample_list) != 9:
            print ("[ERROR] : expecting 9 samples in input")
            raise RuntimeError("malformed vbf input sample list")
        M_tofill = [
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None],
            [None,None,None,None,None,None,None,None,None]
        ]
        
        self.xSections={}
        for isample, sample in enumerate(self.sample_list):
            print(isample, " c3, d4  = ", sample.val_c3, sample.val_d4, sample.val_xs)

            ## implement the 9 scalings
            M_tofill[isample][0] = sample.val_c3**4                             
            M_tofill[isample][1] = sample.val_c3**3
            M_tofill[isample][2] = sample.val_c3**2
            M_tofill[isample][3] = sample.val_c3**2 * sample.val_d4
            M_tofill[isample][4] = sample.val_c3    
            M_tofill[isample][5] = sample.val_c3 * sample.val_d4
            M_tofill[isample][6] = sample.val_d4**2
            M_tofill[isample][7] = sample.val_d4
            M_tofill[isample][8] = 1.0

            self.xSections['s'+str(isample+1)]=sample.val_xs


        #print (M_tofill)
        #print("Input Matrix defined as  :\n")
        #for row in  M_tofill:
        #    print(end="   ")
        #    for item in row:
        #        print("{0:0.3f}".format(item), " | ",end="")
        #    print()
 
        self.M = Matrix(M_tofill)
        #print ("\n Determinant of the matrix : " , det(self.M) )

    def calculatecoeffients(self):
        """ create the function sigma and the nine coefficients in this object """

        try: self.M
        except AttributeError: self.build_matrix()
        ##############################################    
        c3, d4, A, B, C, D, E, F, G, H, I, s1, s2, s3, s4, s5, s6, s7, s8, s9 = symbols('c3, d4, A, B, C, D, E, F, G, H, I, s1, s2, s3, s4, s5, s6, s7, s8, s9')
        ### the vector of couplings
        ### the vector of couplings
        c = Matrix([
            [c3**4 ] ,
            [c3**3] ,
            [c3**2] ,
            [c3**2 * d4] ,
            [c3] ,
            [c3 * d4] ,
            [d4**2] ,
            [d4] ,
            [1] ,
        ])
        ### the vector of components
        v = Matrix([
            [A] ,
            [B] ,
            [C] ,
            [D] ,
            [E] ,
            [F] ,
            [G] ,
            [H] ,
            [I] 
        ])
        ### the vector of samples (i.e. cross sections)
        s = Matrix([
            [s1] ,
            [s2] ,
            [s3] ,
            [s4] ,
            [s5] ,
            [s6] ,
            [s7] ,
            [s8] ,
            [s9] 
        ])
        ####    
        Minv   = self.M.inv()
        self.coeffs = c.transpose() * Minv # coeffs * s is the sigma, accessing per component gives each sample scaling
        self.sigma  = self.coeffs*s
        substitutions=[]
        for ky in self.xSections:
            substitutions.append((ky,self.xSections[ky]))
        self.sigmaEval=self.sigma.subs(substitutions)

    def evaluateSigma(self,params):
        substitutions=[]
        if self.sigmaEval==None:
            print("Evaliate the matrix first !! ")
            return -1.0
        for ky in params:
            substitutions.append((ky,params[ky]))
        return self.sigmaEval.subs(substitutions)[0]

####################


class HHHModel(PhysicsModel):
    """ Models the HHH production as linear sum of 9 components (GGF) """
    def __init__(self, ggHHH_sample_list , name):
        PhysicsModel.__init__(self)
        self.name            = name
        self.check_validity_ggf(ggHHH_sample_list)
        self.ggHHH_formula = HHHFormula(ggHHH_sample_list)
        self.dump_inputs()
        self.hh_options = OrderedDict() # placeholder
        self.reset_pois = None #placeholder

    def check_validity_ggf( self, ggf_sample_list ):
        if len(ggf_sample_list) != 9:
            raise RuntimeError("%s : malformed GGHHH input sample list - expect 9 samples" % self.name)
        if not isinstance(ggf_sample_list, list) and not isinstance(ggf_sample_list, tuple):
            raise RuntimeError("%s : malformed GGHHH input sample list - expect list or tuple" % self.name)
        for s in ggf_sample_list:
            if not isinstance(s, HHHSample ):
                raise RuntimeError("%s : malformed GGF input sample list - each element must be a HHHSample" % self.name)

    def dump_inputs(self):
        print ("[INFO]  HHH model : " , self.name)
        print ("......  ggHHH configuration")
        for i,s in enumerate(self.ggHHH_formula.sample_list):
            print ("        {0:<3} ... kl : {1:<3}, kt : {2:<3}, xs : {3:<3.8f} pb, label : {4}".format(i, s.val_c3, s.val_d4, s.val_xs, s.label))

    def doParametersOfInterest(self):
        
        ## the model is built with:
        ## GGF = r_GGF x [sum samples(kl, kt)] 
        
        
        POIs = "r,c3,d4"
        #self.modelBuilder.doVar("r[1,0,10000]")
        self.modelBuilder.doVar("r[0.001,0,10000.0]")
        #self.modelBuilder.doVar("r_hhh[1,0,10000]")
        self.modelBuilder.doVar("c3[0.0,-10000,10000]")
        self.modelBuilder.doVar("d4[0.0,-10000,10000]")
        
        self.modelBuilder.doSet("POI",POIs)

        #self.modelBuilder.out.var("r_gghh") .setConstant(True)
        #self.modelBuilder.out.var("r_qqhh") .setConstant(True)
        #self.modelBuilder.out.var("CV")     .setConstant(True)
        #self.modelBuilder.out.var("C2V")    .setConstant(True)
        #self.modelBuilder.out.var("kl")     .setConstant(True)
        #self.modelBuilder.out.var("kt")     .setConstant(True)
        
        #self.modelBuilder.out.var("r_hhh")  .setConstant(True)
        self.modelBuilder.out.var("c3")     .setConstant(True)
        self.modelBuilder.out.var("d4")     .setConstant(True)
        

        self.create_scalings()

    def create_scalings(self):
        """ create the functions that scale the six components of vbf and the 3 components of ggf """
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
        for i, s in enumerate(self.ggHHH_formula.sample_list):
            f_name = 'f_ggfhhhscale_sample_{0}'.format(i)
            f_expr = self.ggHHH_formula.coeffs[i] # the function that multiplies each sample

            # print f_expr
            # for ROOFit, this will convert expressions as a**2 to a*a
            s_expr = pow_to_mul_string(f_expr)
            couplings_in_expr = []
            if 'c3'  in s_expr: couplings_in_expr.append('c3')
            if 'd4'  in s_expr: couplings_in_expr.append('d4')

            # no constant expressions are expected
            if len(couplings_in_expr) == 0:
                raise RuntimeError('GGF HH : scaling expression has no coefficients')
            
            for idx, ce in enumerate(couplings_in_expr):
                # print '..replacing', ce
                symb = '@{}'.format(idx)
                s_expr = s_expr.replace(ce, symb)

            arglist = ','.join(couplings_in_expr)
            exprname = 'expr::{}(\"{}\" , {})'.format(f_name, s_expr, arglist)
            print ("Expression name  : " , exprname)
            self.modelBuilder.factory_(exprname) # the function that scales each sample
            
            #f_prod_name_pmode = f_name + '_r_hhh'
            #prodname = 'prod::{}(r_hhh,{})'.format(f_prod_name_pmode, f_name)
            #self.modelBuilder.factory_(prodname)  ## the function that scales this production mode
            #self.modelBuilder.out.function(f_prod_name_pmode).Print("") ## will just print out the values

            #f_prod_name = f_prod_name_pmode + '_r'
            f_prod_name = f_name + '_r'
            prodname = 'prod::{}(r,{})'.format(f_prod_name, f_name)
            self.modelBuilder.factory_(prodname)  ## the function that scales this production mode
            self.modelBuilder.out.function(f_prod_name).Print("") ## will just print out the values

            #self.f_r_ggf_names.append(f_prod_name) #bookkeep the scaling that has been created
            self.f_r_ggf_names.append(f_prod_name) #bookkeep the scaling that has been created


    def getYieldScale(self,bin,process):
        #print "got process/bin : ", process," / ",bin
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

        for isample, sample in enumerate(self.ggHHH_formula.sample_list):
            if process.startswith(sample.label):
                #print self.name, ": {:>20}  ===> {:>20}".format(process, sample.label)
                imatched_ggf.append(isample)

        ## this checks that a process finds a unique scaling
        if len(imatched_ggf) != 1:
            print ("[ERROR] : in HHH model named", self.name, "there are", len(imatched_ggf), "GGF name matches")
            raise RuntimeError('HHHModel : could not uniquely match the process %s to the expected sample list' % process)

        if len(imatched_ggf) == 1:
            isample = imatched_ggf[0]
            self.scalingMap[process].append((isample, 'GGF'))
            #print isample , self.f_r_ggf_names[isample]
            return self.f_r_ggf_names[isample]
        raise RuntimeError('HHHModel : fatal error in getYieldScale - this should never happen')

    def done(self):
        print ('the function done is not used for the moment, have to be updated not to fail for Run II combination')
        ## this checks that a scaling has been attached to a unique process
 #       scalings = {}
 #       for k, i in self.scalingMap.items(): ## key -> process, item -> [(isample, 'type')]
 #           samples = list(set(i)) # remove duplicates
 #           for s in samples:
 #               if not s in scalings:
 #                   scalings[s] = []
 #               scalings[s].append(k)
#
#        #for key, val in scalings.items():
#        #    if len(val) > 1:
#        #        print "[ERROR] : in HH model named", self.name, "there is a double assignment of a scaling : ", key, " ==> ", val
#        #        raise RuntimeError('HHModel : coudl not uniquely match the scaling to the process')
#
#        ## now check that, if a VBF/GGF scaling exists, there are actually 6/3 samples in the card
#        n_VBF = 0
#        n_GGF = 0
#        for k, i in self.scalingMap.items():
#            # the step above ensured me that the list contains a single element -> i[0]
#            if i[0][1] == "GGF":
#                n_GGF += 1
#            elif i[0][1] == "VBF":
#                n_VBF += 1
#            else:
#                raise RuntimeError("HHModel : unrecognised type %s - should never happen" % i[0][1])
#
#        if n_GGF > 0 and n_GGF != 3:
#            raise RuntimeError("HHModel : you did not pass all the 3 samples needed to build the GGF HH model")
#        
#        if n_VBF > 0 and n_VBF != 6:
#            raise RuntimeError("HHModel : you did not pass all the 6 samples needed to build the VBF HH model")


####################
# set of all cross sections

HHH_List=[]
br_ratio=0.0023098822656
#HHH_List.append(HHHSample( 0,0,val_xs=3.274e-05*br_ratio *1e3*2.22 , label='c3_0_d4_0' ) )
#HHH_List.append(HHHSample( 0,99,val_xs=5.243e-03*br_ratio *1e3*2.22 , label='c3_0_d4_99' ) )
#HHH_List.append(HHHSample( 0,-1,val_xs=3.624e-05*br_ratio *1e3*2.22 , label='c3_0_d4_m1' ) )
#HHH_List.append(HHHSample( 19,19,val_xs=1.318e-01*br_ratio *1e3*2.22 , label='c3_19_d4_19' ) )
#HHH_List.append(HHHSample( 1,0,val_xs=2.567e-05*br_ratio *1e3*2.22 , label='c3_1_d4_0' ) )
#HHH_List.append(HHHSample( 1,2,val_xs=1.415e-05*br_ratio *1e3*2.22 , label='c3_1_d4_2' ) )
#HHH_List.append(HHHSample( 2,-1,val_xs=5.110e-05*br_ratio *1e3*2.22 , label='c3_2_d4_m1' ) )
#HHH_List.append(HHHSample( 4,9,val_xs=2.182e-04*br_ratio *1e3*2.22 , label='c3_4_d4_9' ) )
#HHH_List.append(HHHSample( -1,0,val_xs=1.004e-04*br_ratio *1e3*2.22 , label='c3_m1_d4_0' ) )
#HHH_List.append(HHHSample( -1,-1,val_xs=9.674e-05*br_ratio *1e3*2.22 , label='c3_m1_d4_m1' ) )
#HHH_List.append(HHHSample( -1.5,-0.5,val_xs=1.723e-04*br_ratio *1e3*2.22 , label='c3_m1p5_d4_m0p5' ) )

HHH_List.append(HHHSample( 0   ,0  ,  val_xs=3.274e-05*br_ratio*1e3*2.22,      label='c3_0_d4_0' ) )
HHH_List.append(HHHSample( 0   ,99 ,  val_xs=5.243e-03*br_ratio*1e3*2.22,      label='c3_0_d4_99' ) )
HHH_List.append(HHHSample( 0   ,-1 ,  val_xs=3.624e-05*br_ratio*1e3*2.22,      label='c3_0_d4_m1' ) )
HHH_List.append(HHHSample( 19  ,19 ,  val_xs=1.318e-01*br_ratio*1e3*2.22,      label='c3_19_d4_19' ) )
HHH_List.append(HHHSample( 1   ,0  ,  val_xs=2.567e-05*br_ratio*1e3*2.22,      label='c3_1_d4_0' ) )
HHH_List.append(HHHSample( 4   ,9  ,  val_xs=2.182e-04*br_ratio*1e3*2.22,      label='c3_4_d4_9' ) )
HHH_List.append(HHHSample( -1  ,0  ,  val_xs=1.004e-04*br_ratio*1e3*2.22,      label='c3_m1_d4_0' ) )
HHH_List.append(HHHSample( -1  ,-1 ,  val_xs=9.674e-05*br_ratio*1e3*2.22,       label='c3_m1_d4_m1' ) )
HHH_List.append(HHHSample( -1.5,-0.5, val_xs=1.723e-04*br_ratio*1e3*2.22,     label='c3_m1p5_d4_m0p5' ) )


print("Making the Object with ",len(HHH_List)," samples \n") 
gHere = HHHFormula(HHH_List)

#print("Crossection as a function of {sigma_i} {c3,d4} ! ")
#for sig in gHere.sigma:
#    print(sig)
#
#print("Scaling funtion to be applied to the sample {i} for a given {c3,d4} ! ")
#for i,sig in enumerate(gHere.coeffs):
#    print (i , "   :  ", sig)
#    s=str(sig)
#    print("   > for (0,0) " ,eval(s.replace('c3','0.0').replace('d4','0.0')))
#
#print("   Validation !  =====================  ")
#for sample in HHH_List:
#    print ("{ ",sample.label,"  | ",sample.val_c3 , "  |  " ,sample.val_d4) 
#    evals=gHere.evaluateSigma({'c3':sample.val_c3 , 'd4':sample.val_d4 })
#    print ("   > exact : { ",sample.val_xs," } | eval : { ",evals," } | ratio : { ",evals/sample.val_xs," } )")
#print("              !  =====================  ")




#print("Formula =  ",gHere.sigma[0])
########################################################
# definition of the model inputs
## NOTE : the scaling functions are not sensitive to global scalings of the xs of each sample
## so by convention use val_xs of the samples *without* the decay BR
## NOTE 2 : the xs *must* correspond to the generator one



model_default = HHHModel(
    ggHHH_sample_list = HHH_List,
    name            = 'model_default'
)

#if __name__=="__main__":
#    gHere = HHHFormula(HHH_List)
#        
#    for st in gHere.sigmaEval:
#        print(st)
#
#    for i,st in enumerate(gHere.coeffs):
#        print(i,st)
#
