# Data

`wind_data_all_zone.csv` is the aligned GEFCom2014 wind dataset used for the
joint multisite experiments. The loader in `stde_cdm.joint_data` constructs
the learning, validation, and locked TEST splits.

This repository does not assert ownership of the underlying competition data.
The source is the public
[GEFCom2014 wind forecasting dataset](https://ieee-pes-data-sharing.org/datasets/detail/0e87366e-2e91-4024-b658-43f6b22faa69).
Users should review the official dataset record and applicable terms before
redistributing the data separately.

`generated/wind_UMNN_M_1_z0-1-2-3-4_d0_n6000.npz` is the compact reference
pool used to instantiate the FICA system and regenerate the dispatch figure.
Large model-by-day candidate and validation pools are not stored here;
they can be regenerated from the saved checkpoints and sampling seeds.
