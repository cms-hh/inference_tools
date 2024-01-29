# HH physics models

- `hh_model.py`: The default HH physics model covering ggF, VBF and VHH production. Self-contained file including all necessary objects.
- `hh_model_C2klkt.py`: Extension of the HH physics model to include C2 in the modeling of ggF production (in addition to kl and kt). Requires `hh_model.py`.
- `hh_model_C2klkt_EFT.py`: Extension of the C2 HH model to recast HH results in various UV complete models. Includes also SH (STSX) and BR scaling for modifications in all SH kappas. Requires `hh_model_C2klkt.py` and `h_hh_model_kWkZ.py`
- `h_hh_model_kWkZ.py` Extension of the HH model to include modifications of SH (also STSX) cross sections and branching fractions as a function of all SH kappas in the kappa framework. Slightly modified (split kV in kW and kZ) version from the H+HH combination (HIG-23-006).
- `hh_model_boosted.py`: Extension of the HH physics model to include signal processes dedicated to boosted topologies for ggF and VBF production. Requires `hh_model.py`.
- `HBRscaler.py`: Initial implementation of the scaling of Higgs branching ratios and single Higgs backgrounds with kappa parameters. Fully included in `hh_model.py`.
