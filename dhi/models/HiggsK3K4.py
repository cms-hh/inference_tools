### This is the base python class to study the Higgs width

from __future__ import absolute_import, print_function

from HiggsAnalysis.CombinedLimit.PhysicsModel import *


class HiggsK3K4(PhysicsModel):
    def __init__(self):
        self.verbose = False
        self.pois = {}
        self.poiMap = []
    
    def setModelBuilder(self, modelBuilder):
        PhysicsModel.setModelBuilder(self,modelBuilder)
        self.modelBuilder.doModelBOnly = False

    def getYieldScale(self,bin,process):
        if process == "GluGluToHHHTo6B_SM": 
            print("Will scale signal")
            return "get_k3_k4_func"
        else: return 1

    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
       
        self.modelBuilder.doVar("k3[1,-100,100]")
        print("Creating k3 POI")
        self.modelBuilder.doVar("k4[1,-1000,1000]")
        print("Creating k4 POI")

        self.modelBuilder.factory_('expr::get_k3_k4_func("(1-0.921*(@0-1)-0.091*(@1-1)+0.86*(@0-1)*(@0-1)-0.168*(@0-1)*(@1-1)+0.0171*(@1-1)*(@1-1)-0.258*(@0-1)*(@0-1)*(@0-1)+0.0491*(@0-1)*(@0-1)*(@1-1)+0.0413*(@0-1)*(@0-1)*(@0-1)*(@0-1))",k3,k4)')
        print("Getting parametrisation")

        self.modelBuilder.doSet("POI", 'k3,k4')





higgsk3k4 = HiggsK3K4()