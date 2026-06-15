### MRI_QSM

This repository contains two separate workflows:

1. **QSM2016 algorithm-test workflow**
   - Entry: `main_qsm_pipeline.m`
   - Purpose: compare TKD / CFL2 / iLSQR / MEDI / xQSM etc. on QSM2016-style `.mat` data.

2. **Real-subject DICOM WH-QSM-only workflow**
   - Entry: `MRI_QSM_dicom_adapter/run_whqsm_comparison.m`
   - Purpose: process two real DICOM subjects with the validated WH-QSM algorithm only.
   - Current version: v4, DICOM multi-echo phase fitting + SEPIA/FANSI weak-harmonic QSM.

For real subject data, the fixed project root is:

```text
D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge
```

Use the one-click fixed-path entry from that root:

```matlab
cd D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge
RUN_REALDATA_WHQSM_ONECLICK
```

Or double-click `RUN_WHQSM_ONECLICK.bat` if MATLAB is on Windows PATH.

All fixed paths are centralised in:

```text
MRI_QSM_dicom_adapter/whqsm_local_paths.m
```

The real-data workflow does **not** run TKD/CFL2/iLSQR/MEDI/xQSM and does **not** generate unregistered voxel-wise subtraction maps.

See `MRI_QSM_dicom_adapter/README.md` for details.
