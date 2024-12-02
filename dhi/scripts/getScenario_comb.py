#!/usr/bin/env python
info = {
    'S2': {
      'scale_args': [
          #from 4b boosted
            r"""--X-nuisance-group-function 'pLeptonID' 'expr::scaleLeptonID("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pJmsAK8J' 'expr::scaleJmsAK8J("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pJmrAK8J' 'expr::scaleJmrAK8J("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pPnetAK8J' 'expr::scalePnetAK8J("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
          #from 4b
            r"""--X-nuisance-group-function 'pTrigEff' 'expr::scaleTrigEff("1/sqrt(@0)",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pResJBreg' 'expr::scaleResJBreg("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
          #from bbtt,bbgg
            r"""--X-nuisance-group-function 'pBTag' '1.0'""",
            r"""--X-nuisance-group-function 'pBTagStat' 'expr::scaleBTagStat("1/sqrt(@0)",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pEleID' 'expr::scaleEleID("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pMuonID' 'expr::scaleMuonID("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pLepFakes' 'expr::scaleLepFakes("max(0.4,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pPhotonID' 'expr::scalePhotonID("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pTauID' '1.0'""",
            r"""--X-nuisance-group-function 'pScaleJ' 'expr::scaleScaleJ("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pScaleJAbs' 'expr::scaleScaleJAbs("max(0.3,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pScaleJFlav' 'expr::scaleScaleJFlav("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pScaleJPileup' '1.0'""",
            r"""--X-nuisance-group-function 'pScaleJRel' 'expr::scaleScaleJRel("max(0.2,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pScaleJTime' 'expr::scaleScaleJTime("1/sqrt(@0)",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pScaleJMethod' 'expr::scaleScaleJMethod("1/sqrt(@0)",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pResJ' 'expr::scaleResJ("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pScaleMet' 'expr::scaleScaleMet("max(0.5,1/sqrt(@0))",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'pLumi' '0.6'""",
            r"""--X-nuisance-group-function 'pOther' 'expr::scaleOther("1/sqrt(@0)",lumiscale[1])'""",
            r"""--X-nuisance-group-function 'sigTheory' '0.5'""",
            r"""--X-nuisance-group-function 'bkgTheory' '0.5'"""
      ]
    },
    'S2ESU': {
      'scale_args': [
          # theory
          r"""--X-nuisance-group-function 'sigTheory' '0.5'""",
          r"""--X-nuisance-group-function 'bkgTheory' '0.5'"""
          # lumi
          r"""--X-nuisance-group-function 'pLumi' '0.6'""",
          # b(b)-tagging
          r"""--X-nuisance-group-function 'pPnetAK8J' '0.5'""",
          r"""--X-nuisance-group-function 'pBTagHF' '0.5'""",
          r"""--X-nuisance-group-function 'pBTagLF' '1.0'""",
          # JES and JER
          r"""--X-nuisance-group-function 'pScaleJ' '1.0'""",
          r"""--X-nuisance-group-function 'pScaleJAbs' '1.0'""",
          r"""--X-nuisance-group-function 'pScaleJFlav' '1.0'""",
          r"""--X-nuisance-group-function 'pScaleJPileup' '1.0'""",
          r"""--X-nuisance-group-function 'pScaleJRel' '1.0'""",
          r"""--X-nuisance-group-function 'pScaleJTime' '1.0'""",
          r"""--X-nuisance-group-function 'pScaleJMethod' '1.0'""",
          r"""--X-nuisance-group-function 'pResJ' '1.0'""",
          r"""--X-nuisance-group-function 'pScaleMet' '1.0'""",
          # Photon efficiency
          r"""--X-nuisance-group-function 'pPhotonID' '0.8'""",
          r"""--X-nuisance-group-function 'pHggTrigger' '0.8'""",
          r"""--X-nuisance-group-function 'pHggPreselections' '0.8'""",
          # Photon energy scale and resolution
          r"""--X-nuisance-group-function 'pResPhoton' '1.0'""",
          # tau ID
          r"""--X-nuisance-group-function 'pTauID' '1.0'""",
          # regressions for jet energy and mass - Not in ATLAS
          r"""--X-nuisance-group-function 'pJmsAK8J' '0.5'""",
          r"""--X-nuisance-group-function 'pJmrAK8J' '0.5'""",
          r"""--X-nuisance-group-function 'pResJBreg' '0.5'""",
          
      ]
    },
}

def GetOpts(scenario):
    return ' '.join(info[scenario]['scale_args'])

if __name__ == "__main__":
    import sys

    if sys.argv[2] == '-o':
        print GetOpts(sys.argv[1])

