# HH physics models

- `hh_model.py`: The default HH physics model covering ggF, VBF and VHH production. Self-contained file including all necessary objects.
- `hh_model_C2klkt.py`: Extension of the HH physics model to include C2 in the modeling of ggF production (in addition to kl and kt). Requires `hh_model.py`.
- `hh_model_boosted.py`: Extension of the HH physics model to include signal processes dedicated to boosted topologies for ggF and VBF production. Requires `hh_model.py`.
- `HBRscaler.py`: Initial implementation of the scaling of Higgs branching ratios and single Higgs backgrounds with kappa parameters. Fully included in `hh_model.py`.
