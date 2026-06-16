# 场图制备（背景场去除 BFR）—— 修复"苍白球偏低/输入场异常"

## 问题与诊断（已被代码对比 + 文献双重证实）

**现象**：真实被试 WH-QSM 的输入场图深部全平、只剩血管/边缘，苍白球 χ 只有 0.0x，
远低于 COSMOS 金标准（~0.10-0.18 ppm）。

**根因**：真实 DICOM pipeline **缺少 QSM 共识规定的"背景场去除(BFR)"必需步骤**。
- DICOM loader 算出的 `fieldmap_Hz` 是**总场(total field)**（且仅做了回波维解缠）；
- 之前直接把总场当局部场喂给 WH-QSM 反演；
- Challenge 数据之所以正常，是因为它的 `phs_tissue` **已经是别人做完空间解缠 + LBV
  背景场去除的组织场**。两条 pipeline 输入口径不一致，不是反演 bug。

## 文献依据

1. **ISMRM QSM 共识 (2024)**：标准流程为
   `相位解缠+回波合并 → 脑mask → 背景场去除(SHARP类/PDF) → 稀疏正则偶极反演`；
   背景场必须在 brain mask 内去除后再反演。
2. **SEPIA 官方教程 Exercise 3**：背景场去除后，
   *"globus pallidus, red nuclei and substantia nigra are visible"*
   —— 反证：不做 BFR 这些深部核团就看不清（正是本问题现象）。
3. **WH-QSM 配套**：QSM Challenge phantom 的 BFR 对比研究指出
   *"When paired with the Weak Harmonic QSM algorithm, LBV showed the best overall
   performance"* —— WH-QSM 的弱谐波项只处理**残余**背景场，**不能替代**完整 BFR。

## 修复（不触碰反演）

新增独立阶段 `modules/mod_field_preprocess.m`，在 WH-QSM 反演**之前**：
1. **空间相位解缠**（Laplacian，经 SEPIA/MEDI；补足 loader 仅做的回波维解缠）；
2. **背景场去除**：**LBV 优先**（文献：配 WH-QSM 最佳）→ 回退 V-SHARP → PDF；
   全部复用库内成熟封装（`bg_removal_lbv_medi` / `bg_removal_vsharp` /
   `bg_removal_pdf_medi`，底层是 MEDI/SEPIA/STI Suite）；
3. **QC 出图**：总场 vs 局部场三平面对比（局部场应露出深部核团偶极结构 = BFR 生效）。

WH-QSM 反演代码**一字未改**。

## 开关（关键）

`cfg.whqsm.do_bfr`：
- **真实 DICOM 数据** → `true`（已在 `run_whqsm_comparison.m` 自动开启）；
- **Challenge 数据**（输入已是 `phs_tissue` 局部场）→ `false`（默认），**不要**重复 BFR。

相关参数（`run_whqsm_comparison.m` 内）：
```matlab
cfg.whqsm.do_bfr = true;
cfg.whqsm.bfr_method = 'LBV';      % 'LBV'|'VSHARP'|'PDF'|'auto'
cfg.whqsm.bfr_tol = 0.005;
cfg.whqsm.bfr_peel = 2;
cfg.whqsm.do_spatial_unwrap = 'auto';
```

## 验证

重跑后用 `RUN_QSM_ROI_CHECK` 看苍白球数值是否回到文献范围（~0.10-0.18 ppm），
并查看 `field_preprocess_qc.png`：局部场应清晰露出苍白球/红核/黑质的偶极结构。

## 依赖

需要 MEDI toolbox（LBV/PDF）和/或 SEPIA（LBV wrapper、Laplacian 解缠）在 path。
已由 `whqsm_local_paths.m` 的 `P.mediRoot` / `P.sepiaRoot` 配置。
