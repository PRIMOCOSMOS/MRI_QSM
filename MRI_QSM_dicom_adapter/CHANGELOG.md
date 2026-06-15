# Changelog - MRI_QSM dicom_adapter

## v4 (2026-06-14) — WH-QSM-only real subject pipeline

### 目标变化
- 真实被试流程只跑 **WH-QSM**。
- 不再在真实被试数据上重复 TKD / CFL2 / iLSQR / MEDI / xQSM 等旧算法测试流程。
- 直接调用下层 SEPIA `QSMMacroIOWrapper` + FANSI weak-harmonic 接口。

### 一键运行
- 新增 `whqsm_local_paths.m`：集中固化 `projectRoot = D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge`、`dataRoot`、`sepiaRoot`、日志目录和运行选项。
- 新增 `RUN_WHQSM_ONECLICK.m`：在 adapter 目录一键运行。
- 新增根目录 `RUN_REALDATA_WHQSM_ONECLICK.m`：在项目根目录一键运行。
- 新增根目录 `RUN_WHQSM_ONECLICK.bat`：可选 Windows 双击入口。

### v4 hotfix
- 修复第二个被试在 phase fitting 时可能报 `输入参数太多` 的问题：SEPIA/FANSI 运行后可能把第三方 `unwrap.m` 加到 path 前面，导致 `unwrap(phase,[],4)` 被错误解析。现在改为本地 `unwrap_echo_phase_local`，不再依赖 path 上的 `unwrap`。
- 增强可视化：新增每个被试的 native-space QC panel，包括 magnitude、mask、input field ppm、WH-QSM 三平面；继续禁止未配准 voxel-wise subtraction。
- 新增 SNU-LIST Chi-Separation Toolbox v1.2.1 适配：`snu_chisep_v121_adapter.m` 调用 `chi_sepnet_general`，使用 GRE-only R2* 路线，输出 `chi_para_ppm.nii` / `chi_dia_ppm.nii` / QC 图。
- DICOM loader 增加 4-echo magnitude `R2star_Hz` 拟合，为磁化率分离提供 feature map。

### 关键修正
- 新增 `modules/mod_whqsm_reconstruction.m`
  - 公开的 WH-QSM-only 模块。
  - 使用 `cfg.sepiaRoot`，去除旧版硬编码 SEPIA 路径问题。
  - 将 DICOM field map 以 **Hz** 写入 SEPIA NIfTI。
  - 将实际 `TE`、`delta_TE`、`B0`、`B0_dir`、`voxel_size` 写入 SEPIA header。
  - WH-QSM 失败时保留 debug 信息，不 silent fallback 到其他算法。

- 重写 `run_whqsm_comparison.m` v4
  - fail-fast 检查 SEPIA / DICOM / NIfTI I/O。
  - 每个被试只调用 `dicom_loader_subject` + `mod_whqsm_reconstruction`。
  - 输出 `chi_normal.mat`、`chi_elderly.mat`、`whqsm_*_complete.mat`。

- 重写 `dicom_loader_subject.m` v4
  - 支持两回波/多回波 phase fitting。
  - 不再只取最后一个 echo。
  - Siemens phase internal units → radians → fieldmap_Hz。
  - 保存 `fieldmap_Hz`、`local_field_ppm`、`echo_times_ms`、`delta_TE`、`B0`、`phase_fit_method`。
  - 修复 qsm2016_format 旧版只保存 `phs_tissue.mat` 且变量名为 `data` 的问题。

- 重写 `compare_subjects.m` v4
  - 不再生成未配准前提下不严谨的 `elderly - normal` voxel-wise subtraction。
  - 输出 side-by-side QC、histogram、描述性 summary CSV。

- 重写 `discover_subjects.m` v5
  - DICOM 发现支持 `.dcm/.dicom/.IMA/.001` 与 DICM magic。
  - 修复多 pattern `contains` 潜在非标量逻辑问题。

---

## v3 (2026-06-13)
- `phs_tissue` 单位由 rad 修正为 ppm。
- 增加 B0 自动检测。
- 增加 `run_whqsm_comparison.m` 预检查。

## v2 (2026-06-13)
- 删除独立手写 WH-QSM。
- 改为复用原库 `mod_dipole_inversion.m → inversion_whqsm_stable` 调用 SEPIA/FANSI。

## v1 (2026-06-13)
- 初版 DICOM 扫描、被试发现、DICOM 加载、对比可视化。
