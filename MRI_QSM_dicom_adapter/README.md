# MRI_QSM dicom_adapter 扩展模块  (v3)

> 用于处理 Siemens SWI DICOM 数据并执行 WH-QSM 重建 + 正常 vs 老年人 对比分析
>
> **核心原则（按用户要求）**：最大限度复用原库方法，调用 SEPIA/FANSI 工具箱，不自己手写 WH-QSM

## 一键运行

```matlab
% 默认路径
run_whqsm_comparison()

% 自定义数据路径
run_whqsm_comparison('D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course')

% 同时指定原库路径
run_whqsm_comparison('D:\path\data_course', 'D:\path\MRI_QSM')

% 跳过 SEPIA 检查（仅用于离线调试）
run_whqsm_comparison(..., 'skip_sepia_check', true)
```

## 模块清单

| 文件 | 版本 | 职责 |
|---|---|---|
| `run_whqsm_comparison.m` | v3 | 🚀 主入口（预检查 + SEPIA 复用 + 容错） |
| `discover_subjects.m` | v1 | 🔍 自动发现 NORMAL/ELDERLY 被试 |
| `dicom_loader_subject.m` | v2 | 🏥 DICOM → QSM2016 .mat（含 rad→ppm 转换） |
| `compare_subjects.m` | v1 | 🆚 正常 vs 老年人 对比 + 可视化 |
| `SWI202606_dicom_scanner.m` | v4 | 🔬 DICOM 元信息扫描（诊断工具） |
| `setup.m` | 🆕 | ⚙️ 一键配置路径 |
| `test_pipeline.m` | 🆕 | 🧪 快速测试（无需 SEPIA） |

## 架构（v3 增强）

```
run_whqsm_comparison()
        │
        ▼
[Init] 预检查
        ├── check_original_library (mod_dipole_inversion, mod_background_removal)
        ├── check_sepia_toolbox (QSMMacroIOWrapper)
        ├── check_image_processing_toolbox (niftiwrite)
        └── 路径自动配置
        │
        ▼
[1] discover_subjects → 找到 NORMAL + ELDERLY
        │
        ▼
[2] 对每个被试 (try/catch 容错):
    ┌─ dicom_loader_subject (v2)
    │     ├── 读 DICOM (Phase Ser#16, Mag Ser#14, T1 Ser#8)
    │     ├── 应用 Siemens 缩放: phase_rad = pixel × slope + intercept
    │     ├── 🔴 rad → ppm 转换 (关键!)
    │     ├── 多回波平均 magnitude
    │     └── 输出 11 个 .mat + data_full.mat
    │
    ├─ mod_background_removal  ◀══ 【复用原库】
    │
    ├─ mod_dipole_inversion    ◀══ 【复用原库】
    │     └── 调用 inversion_whqsm_stable (内嵌 subfunction)
    │           └── SEPIA QSMMacroIOWrapper + FANSI(isWeakHarmonic=true)
    │
    └─ 提取 qsm_results(:,:,:,idx_whqsm)
        │
        ▼
[3] compare_subjects → 三平面 + ROI + 直方图 + 差值图
```

## 🔴 关键单位约定（v2 修正）

| 字段 | 单位 | 来源 | 用途 |
|---|---|---|---|
| `phs_tissue` | **ppm** | rad × (1e6 / 2πγB₀TE) | mod_dipole_inversion 最终反演 |
| `phs_unwrap` | rad | DICOM (Siemens 缩放) | PDF/LBV 背景去除 |
| `phs_wrap` | rad | mod(unwrap + π, 2π) - π | 解缠测试 |
| `magn` | a.u. | DICOM (raw 平均) | mask 提取 + MEDI 先验 |

**转换公式**（在 `dicom_loader_subject.m` 中实现）：
```matlab
ppm = rad × 1e6 / (2π × γ × B0 × TE_sec)
    = rad × 1e6 / (2π × 42.577 × 3 × 0.00973)  % 你的数据
    = rad × 128.0
```

## 预检查流程

v3 新增预检查，**避免运行到一半才发现问题**：

| 检查项 | 失败后果 |
|---|---|
| 原库关键模块存在 | 立即报错 |
| SEPIA 工具箱完整 | 警告（WH-QSM 不可用） |
| Image Processing Toolbox | 警告（niftiwrite 缺失） |
| 数据根目录存在 | 立即报错 |
| NORMAL+ELDERLY 都识别 | 立即报错 |

## 复用原库的方法

| ✅ 复用 | ❌ 不自己写 |
|---|---|
| `mod_dipole_inversion.m` (TKD/CFL2/iLSQR/MEDI/WH-QSM) | ❌ 不自己实现 dipole inversion |
| `mod_background_removal.m` (VSHARP/PDF/LBV/WHQSM flag) | ❌ 不自己实现 SMV |
| `mod_load_data.m` (.mat 兼容) | ❌ 不自己实现数据加载 |
| `Utils_self/create_dipole_kernel.m` | ❌ 不自己构造偶极子核 |
| `Utils_self/qsm_diverging_cmap.m` | ❌ 不自己写色图 |
| `config/pipeline_config.m` (字段约定) | ❌ 不重新定义 cfg 结构 |
| SEPIA `QSMMacroIOWrapper` + FANSI | ❌ 不自己实现 k-space 反演 |

## SEPIA FANSI 参数（由原库 `inversion_whqsm_stable` 自动设置）

```matlab
algorParam.qsm.method = 'FANSI';
algorParam.qsm.isWeakHarmonic = true;   % ← WH-QSM 标志
algorParam.qsm.constraint = 'TV';
algorParam.qsm.lambda = 5e-4;
algorParam.qsm.tol = 1e-4;
algorParam.qsm.maxiter = 100;
algorParam.qsm.beta = 150;
```

修改：在 `mod_dipole_inversion.m → inversion_whqsm_stable` 中调整。

## 自动识别的被试

`discover_subjects.m` 按以下规则分组：

| 文件夹名含 | 分组 |
|---|---|
| `normal` / `control` / `young` / `adult` / `hehongjian` | NORMAL |
| `elderly` / `aged` / `old` / `senior` | ELDERLY |
| 都不匹配 → 按 SWI 编号排序 | 小号=NORMAL，大号=ELDERLY |

**手动覆盖**：编辑 `discover_subjects.m`，找到对应被试，强制设置：
```matlab
subjects(1).group = 'NORMAL';
subjects(2).group = 'ELDERLY';
```

## 输出结构

```
data_course/
├── SWI202606/                           ← NORMAL（何洪建）
├── SWIxxx/                              ← ELDERLY（如果存在）
└── _qsm_comparison_results/             ← 自动生成
    ├── normal_SWI202606/
    │   ├── qsm2016_format/              ← 11 个 .mat (与原库 mod_load_data 兼容)
    │   │   ├── phs_tissue.mat           ← ppm（关键！）
    │   │   ├── phs_unwrap.mat           ← rad
    │   │   ├── msk.mat
    │   │   ├── magn.mat
    │   │   ├── mp_rage.mat
    │   │   ├── spatial_res.mat
    │   │   └── ...
    │   ├── results/
    │   │   ├── background_removal_results.mat
    │   │   ├── dipole_inversion_results.mat  ← 含 WH-QSM
    │   │   └── pipeline_complete_results.mat
    │   ├── figures/
    │   ├── chi_normal.mat
    │   └── all_qsm_normal.mat           ← 所有 5 种方法的结果
    ├── elderly_SWIXXX/
    │   └── (同上)
    └── comparison/
        ├── compare_3view.png            ← 三平面对比
        ├── compare_roi_basal_ganglia.png
        ├── compare_histogram.png
        ├── compare_diff_map.png
        ├── roi_comparison.csv
        └── all_results.mat
```

## 故障排查

### 问题 1: SEPIA 找不到
```
❌ SEPIA 工具箱未找到
```
**解决**：确认 SEPIA 安装路径 `D:\MRI_PRO\MRILAB_X\sepia`，或修改 `run_whqsm_comparison.m` 中 `check_sepia_toolbox` 的 candidates。

### 问题 2: ppm 值域异常
```
⚠️ ppm 值域异常 [-4096, +4094]
```
**原因**：Siemens 缩放公式特殊
**解决**：
1. 跑 `SWI202606_dicom_scanner()` 检查 Phase 字段
2. 在 `dicom_loader_subject.m` 的 `load_phase_volume_rad` 调试
3. 可能需要 `phase = (pixel - 2048) × π/2048` 转换

### 问题 3: WH-QSM 输出异常
```
WH-QSM: 已读取输出文件，但结果数值异常
```
**解决**：
- 加入真正的背景去除：`cfg.bgRemoval.methods = {'VSHARP', 'WHQSM'}`
- 检查 phase 单位（应该在 ±π rad / ±0.5 ppm 范围）

### 问题 4: 只有一个被试被处理
**原因**：未同时识别 NORMAL + ELDERLY
**解决**：手动编辑 `discover_subjects.m` 强制分组

## 与原库的关系

### 完全复用
- ✅ `mod_dipole_inversion.m` — **核心反演**
- ✅ `mod_background_removal.m` — 背景去除
- ✅ `mod_load_data.m` — .mat 兼容
- ✅ `Utils_self/create_dipole_kernel.m` — 偶极子核
- ✅ `Utils_self/qsm_diverging_cmap.m` — 色图

### 仅新增
- 🆕 `discover_subjects.m` — 被试发现
- 🆕 `dicom_loader_subject.m` — DICOM 适配
- 🆕 `compare_subjects.m` — 双被试对比
- 🆕 `setup.m` — 路径配置
- 🆕 `test_pipeline.m` — 离线测试

## 调优建议

### WH-QSM 调优
修改 `mod_dipole_inversion.m → inversion_whqsm_stable` 的 `algorParam`：

| 参数 | 当前 | 调小（更锐利） | 调大（更平滑） |
|---|---|---|---|
| `lambda` | 5e-4 | 1e-4 | 1e-3 |
| `maxiter` | 100 | 50 | 200 |
| `beta` | 150 | 50 | 300 |
| `tol` | 1e-4 | 1e-3 | 1e-5 |

### 背景场去除
在 `pipeline_config.m`（或 `build_pipeline_cfg`）中：
```matlab
cfg.bgRemoval.methods = {'VSHARP', 'WHQSM'};  % 同时跑 V-SHARP
```

## 下一步

1. ✅ 确认 SEPIA 已安装且路径正确
2. ✅ 运行 `setup.m`（首次使用）
3. ✅ 运行 `test_pipeline.m`（快速验证）
4. 🚀 运行 `run_whqsm_comparison()`
5. 📊 查看 `_qsm_comparison_results/comparison/` 下的对比图
