# χ-separation via ONNX Runtime（绕过 MATLAB onnxmex）

## 解决了什么问题

在部分 MATLAB / Windows 环境，运行 SNU-LIST χ-separation 工具箱会报：

```
错误使用 nnet.internal.cnn.onnx.onnxmex
MEX 文件 onnxmex.mexw64 无效: 找不到指定的程序。
```

根因：MATLAB 的 `importNetworkFromONNX` / `importONNXNetwork` 依赖编译好的
MEX（`onnxmex.mexw64`），它在某些机器上因 VC++ 运行库 / 支持包损坏而无法加载。
这是**环境问题，不是模型或代码问题**（SNU-LIST 官方 issue #5 是同一错误）。

本桥接**完全不调用 MATLAB 的 ONNX 导入**：MATLAB 把输入写成 `.mat`，调用一个
独立 Python 脚本，用 `onnxruntime` 直接加载**同一批 `.onnx` 模型**做推理，再把结果
读回 MATLAB。推理逻辑严格复刻官方 `chi_sepnet`（test.py / custom_dataset.py），
结果与官方一致。

## 新增/改动文件

| 文件 | 作用 |
|---|---|
| `modules/DL/python/infer_chisep_from_mat.py` | **新增**：onnxruntime 推理脚本（QSMnet→R2'→χ-sepnet） |
| `modules/snu_chisep_onnxruntime_adapter.m` | **新增**：MATLAB 适配器，桥接到上面的 Python（与 v121 adapter 同签名，可热切换） |
| `MRI_QSM_dicom_adapter/whqsm_local_paths.m` | **改动**：新增 `useOnnxRuntimeChiSep` 开关与 ONNX 桥接配置 |
| `MRI_QSM_dicom_adapter/run_whqsm_comparison.m` | **改动**：透传 ONNX 桥接参数 |
| `MRI_QSM_dicom_adapter/RUN_WHQSM_ONECLICK.m` | **改动**：把 ONNX 参数从 `whqsm_local_paths` 传下去 |
| `modules/run_chisep_only_impl.m` | **改动**：χ-sep-only 入口也注入 ONNX 配置 |
| `MRI_QSM_dicom_adapter/DIAGNOSE_CHISEP_ONNXRUNTIME.m` | **新增**：诊断 Python/包/模型/onnx 可打开性 |
| `run_chisep_onnx_smoketest.m` | **新增**：合成数据端到端冒烟测试 |

## 一次性环境准备

1. 安装一个带 numpy/scipy/onnxruntime 的 Python（你已有 Anaconda）：
   ```bat
   "D:\Anaconda3\python.exe" -m pip install numpy scipy onnxruntime
   ```
   有 NVIDIA GPU 想加速：把 `onnxruntime` 换成 `onnxruntime-gpu`。

2. 打开 `MRI_QSM_dicom_adapter/whqsm_local_paths.m`，确认：
   ```matlab
   P.useOnnxRuntimeChiSep = true;                  % 开启绕过方案
   P.onnxPythonExecutable = 'D:\Anaconda3\python.exe';
   P.onnxNormFactor = '...\Chisep_Toolbox_v1.2.1\models\norm_factor.mat';
   P.onnxQsmModel   = '...\Chisep_Toolbox_v1.2.1\models\240904_QSMnet.onnx';
   % chi-sepnet / R2PRIMEnet 留空会自动在 models\ 目录按常见名查找；
   % 若你的文件名特殊，请显式指定：
   % P.onnxXsepModel    = '...\models\chi_sepnet.onnx';
   % P.onnxR2primeModel = '...\models\R2PRIMEnet.onnx';   % 仅 r2* 流程需要
   ```

## 模型文件名说明（重要）

桥接需要 3 个网络的 `.onnx`：
- **QSMnet**：`240904_QSMnet.onnx`（你已确认存在）
- **χ-sepnet**：把 local field / R2' / QSM → χpara, χdia（**3 入 2 出**）
- **R2PRIMEnet**：R2\* → R2'（**仅 r2\* 流程需要**）

请到 `Chisep_Toolbox_v1.2.1\models\` 看实际文件名，填到对应的 `P.onnxXsepModel` /
`P.onnxR2primeModel`。`norm_factor.mat` 必须含这些键：
`field_mean/std, r2prime_mean/std, r2star_mean/std, x_pos_mean/std, x_neg_mean/std, cosmos_sus_mean/std`。

## 使用步骤

```matlab
cd D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\MRI_QSM_dicom_adapter

% 1) 诊断环境（强烈建议先跑）
DIAGNOSE_CHISEP_ONNXRUNTIME

% 2) 合成数据冒烟测试（验证绕过链路）
cd ..
run_chisep_onnx_smoketest

% 3) 真实数据：已有 WH-QSM 输出时，只跑 χ-separation
RUN_CHISEP_ONLY            % 或 RUN_CHISEP_ONLY('normal') / ('elderly')

% 或者整条一键流程（DICOM→WH-QSM→χ-separation）
cd MRI_QSM_dicom_adapter
RUN_WHQSM_ONECLICK
```

## r2' vs r2\* 流程

- **有 GRE+SE（有 R2 测量）→ r2' 流程（推荐，质量更高）**：
  把 R2 map 提供给桥接即可触发。两种方式：
  - 在 `data` 里提供 `data.R2_Hz`；或
  - 设置 `P.onnxR2Map = '...\R2.mat'`（或直接传变量）。
  桥接内部 `R2' = R2* - R2`（负值截零），**不需要 R2PRIMEnet**。
- **只有 GRE → r2\* 流程**：不提供 R2，桥接自动用 R2PRIMEnet 从 R2\* 估 R2'。
  此时必须配置 `P.onnxR2primeModel`。

`P.onnxPipeline='auto'` 会自动按是否有 R2 选择；也可强制 `'r2p'` / `'r2s'`。

## 统一 QSM 来源（χ-sepnet 的 QSM 输入用谁）

χ-sepnet 需要一张 QSM 作为输入。默认它用工具箱自带的 **QSMnet** 重建；但你的主
流程用的是 **WH-QSM**。为避免两条线 QSM 不一致，新增可配置开关：

```matlab
P.onnxQsmSource = 'qsmnet';    % 官方默认：用 QSMnet onnx 重建 QSM
P.onnxQsmSource = 'external';  % 用你主流程的 WH-QSM(chi_total) 作为 QSM 输入
```

- 选 `external` 时，**无需 QSMnet onnx**。桥接会把传入 adapter 的 `chi_total_ppm`
  （= WH-QSM 结果，ppm）当作 QSM，并用 `cosmos_sus_mean/std` 归一化到 χ-sepnet
  的输入空间（与官方 resolution-generalization 分支喂入预计算 QSM 的方式一致）。
- WH-QSM 数据本来就在库里：`whqsm_*_complete.mat` 的 `chi`，经 `mod_susceptibility_
  separation` 作为 `chi_total_ppm` 传给 adapter，**无需你额外准备数据**。
- 建议 `external` 时保持 `P.snuLocalFieldMode='forward_from_whqsm'`，让 local field
  与该 QSM 自洽。

## 正则化 / "脑内偏淡" 调参

WH-QSM 偏淡多为 **lambda 过大（过平滑）** 或 **显示窗位过宽**。本版已：
- 默认 `P.whqsmLambda` 5e-4 → **3e-4**（更锐、对比更强）；
- 显示窗位收紧到 `±0.10 ppm`（`P.qsmDisplayClim` / `cfg.vis.clim_qsm`）。

客观定位用扫描脚本：
```matlab
RUN_WHQSM_LAMBDA_SWEEP            % 对已有被试扫 [5e-4 3e-4 2e-4 1e-4]
RUN_WHQSM_LAMBDA_SWEEP('normal', [5e-4 3e-4 2e-4 1e-4])
```
输出每个 lambda 的 QSM 对比图 + 全脑 std/p1/p99（直方图宽度）+（若有 ROI 标签）
深部核团均值 ppm 表。lambda 越大 std/p99 越低（越平滑越淡），据此选定后写回
`P.whqsmLambda`。

## 总入口（一步完成）与各阶段单独运行

新增总入口 `RUN_ALL_ONECLICK.m`，一步串起三阶段：
```
阶段1 WH-QSM  →  阶段2 磁化率分离(DL vs 优化)  →  阶段3 两被试配准比较
```
```matlab
RUN_ALL_ONECLICK                          % 全流程 1→3
RUN_ALL_ONECLICK('from','chisep')         % 已有 WH-QSM 结果，从分离开始
RUN_ALL_ONECLICK('from','chisep','to','chisep')  % 只做分离对比
RUN_ALL_ONECLICK('from','compare')        % 只做配准比较
RUN_ALL_ONECLICK('subject','normal')      % 限定被试(阶段1/2)
```
各阶段任一失败会告警并继续后续阶段（不中断整条链）。

**单独只跑磁化率分离（你有预处理 whqsm_*_complete.mat 时）**：
```matlab
RUN_CHISEP_ONLY            % 基于已有 WH-QSM 结果，不重跑 DICOM/WH-QSM
```
其行为由 `P.chisepRunMethodCompare` 控制：true=方法对比，false=单一方法。

## 方法对比：深度学习 vs 传统优化

新增传统凸优化 χ-separation（`snu_chisep_optimization_adapter.m`，**不依赖 ONNX**，
原生 MATLAB 可跑），可与深度学习 χ-sepnet 做算法对比。

模型（Shin 2021，与 MEDI+0/iLSQR 同对比度）：令 x_pos=χ_para, x_neg=|χ_dia|，
- 场模型：`d*(x_pos - x_neg) = local_field(ppm)`
- 弛豫模型：`Dr*(x_pos + x_neg) = R2'(ppm)`
用 s=x_pos−x_neg、t=x_pos+x_neg 解耦：s 走偶极反演(Tikhonov/L1-TV)，t 由 R2' 闭式解。

```matlab
P.chisepCompareMethods = {'onnx','opt'};   % 两种都跑
P.optMethod = 'iLSQR';   % 'iLSQR'(L2,快) | 'MEDI'(L1-TV)
P.optLambda = 1e-2;

RUN_CHISEP_COMPARE            % 对已有被试同时跑两法 + 对比图/统计/相关
RUN_CHISEP_COMPARE('normal')
```
输出 `chisep_method_comparison/`：每法的 χ_para/χ_dia NIfTI、并排对比图、
全局统计 CSV、两法体素相关系数（para/dia）、可选 ROI 均值表。

> 沙箱已用合成体模验证优化求解器：与真值相关 0.99/1.00、峰值还原准确，
> 顺磁/抗磁符号正确。

## 两被试(59 vs 72)配准后比较

原 `compare_subjects.m` 刻意不做相减（无配准的相减无意义）。新增
`mod_two_subject_registered_compare.m`，用 Image Processing Toolbox
（`imregtform/imwarp`，项目已依赖）做配准后再比较：

1. 以 magnitude/T1 做强度配准（rigid，可选 affine）；
2. 同一变换应用到该被试所有图（χ_total / χ_para / χ_dia）；
3. 取两被试脑 mask 交集，计算**有效的**体素差异 + 统计 + 直方图；
4. 可选 ROI 标签(固定空间)给深部核团均值对比。

```matlab
P.twoSubjFixed     = 'normal';   % 参考空间: normal | elderly
P.twoSubjTransform = 'rigid';    % rigid | affine | rigid+affine
P.roiLabelFile     = '';         % 可选: 固定空间的整型标签体 .mat

RUN_TWO_SUBJECT_COMPARE
```
输出 `_two_subject_registered_compare/`：配准后 χ 并排图 + Δχ 差异图、
统计 CSV（含配对 t 统计）、可选 ROI CSV。

> 配准比较前需先有两个被试的 WH-QSM 结果（normal_* 与 elderly_*）。
> 若想比较 χ_para/χ_dia，先跑 `RUN_CHISEP_ONLY` 或 `RUN_CHISEP_COMPARE` 生成分离图。

## 回退

如果以后修好了 MATLAB 的 ONNX，把 `P.useOnnxRuntimeChiSep = false;` 即可切回
原 `snu_chisep_v121_adapter`（调用 SNU p-code）。两条路输出结构完全一致，
下游 `mod_susceptibility_separation` 的保存/出图逻辑不变。

## 网络固定输入尺寸的处理（重要）

SNU 的 onnx 模型是以**固定空间尺寸导出**的（如 `240904_xsepnet.onnx` = 192×192×128），
**不是动态轴**。因此桥接会自动：
1. 读取模型要求的固定尺寸；
2. 把任意尺寸输入（如 336×384×96）**居中裁剪/补零**到该尺寸；
3. 推理后把输出**还原回原始尺寸**。

若某模型是全动态轴，则回退到官方的"裁到 16 倍数"。这一切自动完成，无需配置。

> 已用 336×384×96 输入 + 192×192×128 固定模型端到端验证（r2p/r2s × qsmnet/external 全通过）。

## r2' vs r2* 的自动选择（你的数据当前会走 r2*）

桥接按是否提供 R2 图自动选流程：
- 有 R2（来自 SE 序列，`data.R2_Hz` 或 `P.onnxR2Map`）→ **r2' 流程**（更优）。
- 无 R2 → **r2\* 流程**，用 R2PRIMEnet 从 R2\* 估 R2'。

你当前 `whqsm_*_complete.mat` 只含 R2\*（DICOM loader 只拟合了 R2\*），所以日志显示
`pipeline: r2s` 是**正确行为**，不是错误。若你有 SE 的 R2 测量，把它放到
`data.R2_Hz` 或设 `P.onnxR2Map` 指向 R2 的 .mat，即可自动切到更优的 r2' 流程。

## 算法一致性要点（已逐行核对官方源码）

1. 单位：field Hz→ppm = `field/CF*1e6`；R2\*、R2' 除以 `Dr(=114)` 转 ppm。
2. 维度：三个空间维必须是 16 的倍数 → 居中裁剪（`crop_img_16x`），输出再补回原尺寸。
3. 归一化：`(x-mean)/std` 后乘脑 mask。
4. 通道拼接顺序固定 **[QSM, local_field, R2']**。
5. χ-sepnet 输出 2 通道 = **[χpara, χdia]**；反归一化 `pred*std+mean`，再**负值截零**。
6. 约定：`χpara≥0`，存储 `χdia≤0`，`x_tot = χpara - |χdia|`。
