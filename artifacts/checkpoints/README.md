# Frozen checkpoints

This directory contains the final checkpoints for all ten reported training
seeds, numbered 0 through 9, for each evaluated model family:

- `joint_wgan_gp_seedN.pt`
- `joint_vae_seedN.pt`
- `joint_umnn_seedN.pt`
- `joint_ddpm_seedN.pt`
- `joint_cldm_seedN.pt`
- `stde_spatiotemporal_seedN.pt`

Here, `N` ranges from 0 to 9. There are therefore 60 final joint-model
checkpoints. They are the weights consumed by the ten-seed evaluation and DM
test scripts.

`single_site_cldm/` contains the five compact sitewise CLDM checkpoints used
only by the validation-calibration audit. They are not additional STDE-CDM
experts and are not used by the locked TEST comparison.

New training outputs are written to `outputs/checkpoints/` and are ignored by
Git by default. The packaged files contain final model states and the
normalization/configuration metadata required for evaluation; intermediate
best-component files and optimizer states are not needed for inference.

The ten UMNN checkpoints dominate the binary size. Each file remains below
GitHub's 100 MB per-file limit, but users should expect a larger initial clone.
