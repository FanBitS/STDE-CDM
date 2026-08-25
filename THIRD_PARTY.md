# Third-party provenance

This release combines newly authored STDE-CDM code with retained or
independently reproduced research components.

## Normalizing-flow components

`src/models/` and the Joint UMNN integration originate from the normalizing
flow implementation distributed with Jonathan Dumas et al., *A Deep Generative
Model for Probabilistic Energy Forecasting in Power Systems: Normalizing
Flows*. The retained BSD 2-Clause terms are reproduced in
`LICENSES/UPSTREAM_GENERATIVE_MODELS_BSD-2-Clause.txt`.

## CLDM implementation

`src/cldm/` and the Joint CLDM baseline are independent implementations based
on the published method description of *Short-Term Wind Power Scenario
Generation Based on Conditional Latent Diffusion Models*. They are not an
official upstream code release and do not contain source code supplied by that
paper's authors.

## FICA dispatch optimizer

`fica_dispatch_optimizer/` contains retained FICA and EIFICA optimization code
used by the downstream experiment. Its original documentation and GNU GPL 3.0
notice are preserved as `UPSTREAM_README.md` and `UPSTREAM_LICENSE`. Gurobi is
proprietary software and is not distributed with this repository.

## GEFCom2014 data

The aligned wind data in `data/` originates from the public GEFCom2014 wind
forecasting dataset. This repository does not assert ownership of the original
competition data. The official dataset record is linked from `data/README.md`.

## Project-wide terms

No project-wide license has yet been selected for newly authored STDE-CDM
components. The notices above continue to govern their respective retained
components.
