# Changelog - dicom_adapter

## v3 (2026-06-13) — 当前版本

### 🔴 关键修正
- **`phs_tissue` 单位修正**：从 rad 改为 ppm
  - 新增 rad → ppm 转换：`ppm = rad × 1e6 / (2πγB₀TE)`
  - 解决了原 mod_dipole_inversion 无法直接处理 rad 数据的致命 bug
- **B0 自动检测**：从 DICOM FieldStrength 字段智能识别场强
- **容错处理**：每个被试独立 try/catch，单个失败不影响整体

### 🆕 新增功能
- **`run_whqsm_comparison.m` v3**：
  - 预检查（路径/SEPIA/Toolbox/数据完整性）
  - skip_sepia_check 选项（离线调试）
  - 详细的错误诊断信息
- **`setup.m`**：一键路径配置（自动检测原库、SEPIA、MEDI）
- **`test_pipeline.m`**：离线测试脚本（不需要 SEPIA）
- **PixelRepresentation 处理**：自动处理有符号/无符号相位数据

### 📝 文档
- **README.md v3**：完整的单位约定、预检查流程、故障排查
- **CHANGELOG.md**：版本历史

---

## v2 (2026-06-13)

### 重大修正
- **删除 `wh_qsm_standalone.m`**（不自己手写 WH-QSM）
- **改用原库 `mod_dipole_inversion.m → inversion_whqsm_stable`**（调用 SEPIA/FANSI）
- 完全复用原库 + SEPIA 工具箱

---

## v1 (2026-06-13)

### 初版
- DICOM 扫描器（多次迭代修复）
- 被试发现
- DICOM 加载器（初版）
- 自定义 WH-QSM 实现（**已删除**）
- 对比可视化
