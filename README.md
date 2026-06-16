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

## χ-separation 遇到 `onnxmex.mexw64 无效` 怎么办

如果运行磁化率分离时报 MATLAB `importNetworkFromONNX` / `onnxmex.mexw64 无效:
找不到指定的程序`，这是 MATLAB ONNX 支持包/VC++ 损坏（与模型无关）。本仓库提供
**绕过方案**：用 Python(onnxruntime) 直接加载同一批 `.onnx` 推理。

```matlab
cd D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge
RUN_CHISEP_ONNXRUNTIME          % 一键：诊断 + 冒烟测试 + 真实数据
```

开关与配置在 `MRI_QSM_dicom_adapter/whqsm_local_paths.m`
（`P.useOnnxRuntimeChiSep = true`，默认开启）。详见
`docs/CHISEP_ONNXRUNTIME_BRIDGE.md`。
