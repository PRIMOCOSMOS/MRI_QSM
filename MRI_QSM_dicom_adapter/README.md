# MRI_QSM DICOM Adapter — WH-QSM-only real subject pipeline (v4)

本目录现在只服务一个目标：**把两个真实 DICOM 被试稳定跑 WH-QSM**。旧的 TKD / CFL2 / iLSQR / MEDI / xQSM 对比属于算法测试 Pipeline，不再在真实被试流程中重复运行。

## 一键运行（推荐）

所有固化路径集中在：

```matlab
MRI_QSM_dicom_adapter/whqsm_local_paths.m
```

默认已经固化为：

```matlab
P.projectRoot = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge';
P.dataRoot    = fullfile(P.projectRoot, 'data_course');
P.sepiaRoot   = 'D:\MRI_PRO\MRILAB_X\sepia';
```

运行时只需要：

```matlab
cd MRI_QSM/MRI_QSM_dicom_adapter
RUN_WHQSM_ONECLICK
```

或者在项目根目录 `D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge` 运行：

```matlab
RUN_REALDATA_WHQSM_ONECLICK
```

如果 MATLAB 已加入 Windows PATH，也可以双击项目根目录下的：

```text
RUN_WHQSM_ONECLICK.bat
```

如路径变化，只修改 `whqsm_local_paths.m`，不要修改 Pipeline 主体代码。

## 参数化运行（备用）

```matlab
cd MRI_QSM/MRI_QSM_dicom_adapter
setup
run_whqsm_comparison('D:\path\data_course', 'D:\path\MRI_QSM', ...
    'sepia_root', 'D:\path\sepia')
```

调试时保留 SEPIA 中间 NIfTI：

```matlab
run_whqsm_comparison(..., 'keep_sepia_work', true)
```

## v4 Pipeline

```text
run_whqsm_comparison
  ├─ preflight
  │   ├─ 检查 dicominfo/dicomread/niftiwrite/niftiread
  │   ├─ 检查 mod_whqsm_reconstruction.m
  │   └─ 检查 SEPIA/QSMMacroIOWrapper
  ├─ discover_subjects
  │   └─ 找到第一个 NORMAL + 第一个 ELDERLY
  ├─ NORMAL subject
  │   ├─ dicom_loader_subject
  │   │   ├─ Phase/Magnitude/T1 序列识别
  │   │   ├─ 两回波/多回波 phase(TE) 拟合 → fieldmap_Hz
  │   │   ├─ fieldmap_Hz → ppm compatibility field
  │   │   ├─ 自动 mask
  │   │   └─ 保存 data_full.mat + 单独变量
  │   └─ mod_whqsm_reconstruction
  │       └─ SEPIA QSMMacroIOWrapper + FANSI(isWeakHarmonic=true)
  ├─ ELDERLY subject
  │   └─ 同上
  └─ compare_subjects
      ├─ native-space side-by-side QC
      ├─ histogram
      └─ subject_summary.csv
```

## 当前数据目录结构适配

已针对你的实际目录结构加入强先验：

```text
8_t1_mprage_sag_p2_iso      -> T1 结构像，不允许作为 WH-QSM magnitude
14_Mag_Images / 15_Mag_Images -> 原始 SWI magnitude 候选
16_Pha_Images               -> 原始 SWI phase
17/18_mIP_Images(SW)        -> 投影后处理图，排除
19/20_SWI_Images            -> SWI 后处理图，排除
```

选择逻辑是：先选 `16_Pha_Images` 这类 phase，再从 `14/15_Mag_Images` 中选择与 phase 几何一致的 magnitude；绝不再把 `8_t1_mprage...` 误当作 WH-QSM magnitude。

## 两个/多回波如何处理？

旧版只取最后一个 echo。v4 已改为：

1. 读取每个 echo 的 DICOM `EchoTime`。
2. 将 Siemens phase internal unit 转成 radians：
   - 若数据已在 `[-pi, pi]` 附近：直接使用。
   - 若是常见 Siemens 12-bit 缩放后 `[-4096, 4094]`：使用 `phase_rad = phase_scaled * pi / 4096`。
3. 沿 echo 维度 `unwrap`。
4. 对每个 voxel 拟合：

```text
phase_rad(TE) = intercept + slope * TE
fieldmap_Hz = slope / (2*pi)
fieldmap_ppm = fieldmap_Hz / (gamma_MHz_per_T * B0_T)
```

如果只有一个 echo，则显式 fallback：

```text
fieldmap_Hz = phase_rad / (2*pi*TE)
```

所有 echo 参数会写入：

```matlab
data.echo_times_ms
data.echo_times_sec
data.delta_TE
data.B0
data.B0_dir
data.phase_fit_method
```

这些参数会继续传给 SEPIA header。

## WH-QSM 调用方式

`mod_whqsm_reconstruction.m` 直接调用下层接口：

```matlab
QSMMacroIOWrapper(input, output_basename, mask_filename, algorParam)
```

关键参数：

```matlab
algorParam.qsm.method = 'FANSI';
algorParam.qsm.isWeakHarmonic = true;
algorParam.qsm.constraint = 'TV';
algorParam.qsm.lambda = 5e-4;
algorParam.qsm.tol = 1e-4;
algorParam.qsm.maxiter = 100;
algorParam.qsm.beta = 150;
```

传给 SEPIA 的 local field 单位为 **Hz**，不是 ppm：

```text
localField_Hz = data.fieldmap_Hz
```

SEPIA header 内保存实际 DICOM 参数：

```matlab
TE
delta_TE
B0 / b0
B0_dir / b0dir
CF = B0 * gamma
matrix_size / voxel_size
```

## 磁化率分离 / Susceptibility source separation

当前版本已适配你获得的 SNU-LIST Chi-Separation Toolbox：

```matlab
P.chiSepRoot = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\Chisep_Toolbox_v1.2.1';
P.chiSepAdapterFunction = 'snu_chisep_v121_adapter';
```

Pipeline 会从 4 echo magnitude 拟合 `R2star_Hz`，并在 WH-QSM 后调用：

```matlab
chi_sepnet_general(home_directory, local_field_hz, R2star_Hz, mask, Dr, B0_dir, CF, voxel_size, matrix_size, have_r2map=false)
```

默认使用 GRE-only `R2*` 路线。SNU 输入 local field 默认采用 `forward_from_whqsm`，即由 WH-QSM 总 χ 正演得到较干净的 tissue field，避免把未显式 BFR 的 DICOM fieldmap 直接送入 χ-sepnet。如需改用原始 DICOM phase fitting field，可在 `whqsm_local_paths.m` 中设置：

```matlab
P.snuLocalFieldMode = 'measured';
```

每个被试会输出：

```text
results/susceptibility_separation/
├── chisep_inputs.mat
├── input_chi_total_ppm.nii
├── input_R2star_Hz.nii
├── input_localField_Hz.nii
├── input_mask.nii
├── snu_chisep_v121_inputs.mat
├── snu_chisep_v121_raw_outputs.mat
├── susceptibility_separation_results.mat
├── chi_para_ppm.nii
├── chi_dia_ppm.nii
├── chi_dia_abs_ppm.nii
└── susceptibility_separation_qc.png
```

## 输出结构

```text
data_course/
└── _qsm_comparison_results/
    ├── normal_<subject>/
    │   ├── qsm2016_format/
    │   │   ├── data_full.mat
    │   │   ├── fieldmap_Hz.mat
    │   │   ├── local_field_ppm.mat
    │   │   ├── phs_tissue.mat
    │   │   ├── magn.mat
    │   │   ├── msk.mat
    │   │   ├── spatial_res.mat
    │   │   └── dicom_whqsm_metadata.mat
    │   ├── results/
    │   │   ├── whqsm_result.mat
    │   │   └── WHQSM_chi.nii
    │   ├── chi_normal.mat
    │   └── whqsm_normal_complete.mat
    ├── elderly_<subject>/
    │   └── 同上
    └── comparison/
        ├── qc_normal_native_space.png
        ├── qc_elderly_native_space.png
        ├── compare_3view.png
        ├── compare_histogram.png
        ├── subject_summary.csv
        └── all_results.mat
```

## 注意

- v4 不做跨被试配准，也不做 voxel-wise subtraction。
- `compare_subjects` 只输出 native-space QC 与描述性统计，不把差异图作为结论。
- WH-QSM 必须依赖 SEPIA + FANSI + `QSMMacroIOWrapper`；找不到 SEPIA 时 Pipeline 会 fail fast，不再 silently fallback 到其他算法。
- 若有多个 NORMAL / ELDERLY，本入口只取各组第一个被试。
