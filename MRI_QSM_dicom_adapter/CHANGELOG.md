# Changelog - MRI_QSM dicom_adapter

## v16 (2026-06-16) — 真实数据反演委托给 Challenge 已验证的 inversion_whqsm_stable

### 系统性诊断结论(纠正多次错误假设)
- DIAGNOSE_PHASE_UNWRAP 证明: 脑内空间 wrap 仅 0.1%, 空间解缠后深部场无增强(比值0.98)
  -> 缺空间解缠【不是】根因(此前假设错误, 已排除)。
- 关键发现: 真实数据 pipeline(mod_whqsm_reconstruction) 与 Challenge 验证过的
  inversion_whqsm_stable(mod_dipole_inversion) 是【两套不同反演代码】, 差异包括:
    * 真实路径无 remove_mask_mean(脑内去均值), Challenge 有;
    * 我此前误把 mu1 由 5e-5 改成 100*lambda("文献修复"), 但 Challenge 用 5e-5
      跑得好, 证明 5e-5 才是对的 —— 已撤销该错误改动。
- 用 Challenge(有金标准、已验证)作对照, 锁定问题在"两套反演不一致", 而非数据/场/参数。

### 修复
- 提取 inversion_whqsm_stable 为独立文件 modules/inversion_whqsm_stable.m
  (原为 mod_dipole_inversion 的子函数, 跨文件 exist 不可见, 无法复用)。
- mod_whqsm_reconstruction 现优先【委托】该已验证反演(ppm 输入, 内部 remove_mask_mean
  + Challenge 参数), 失败才回退内置实现。real-data 由此走与 Challenge 完全相同的反演。
- 撤销 run_whqsm_comparison 中错误的 mu1=100*lambda, 恢复 Challenge 验证值 mu1=5e-5。

## v14 (2026-06-16) — 修复 FANSI 反演 mu1 配置(噪声纹理根因)

### 诊断(R2* 交叉验证 + 文献标定, 决定性)
- DIAGNOSE_FIELD_CHAIN 证据: R2* 深部高值=38/s(正常,数据好); 总场/局部场量级正常
  (苍白球 ROI 局部场 std=0.025ppm 与理论吻合)。=> 输入端没问题。
- 但 WH-QSM 输出 χ 是满屏高频红蓝噪声纹理、核团被淹没 => 问题在【反演】。
- 根因: FANSI ADMM 的梯度一致性 mu1=5e-5 < lambda=5e-4(小 10 倍)。文献铁律
  mu1 必须 ≈ 100*lambda(SEPIA 默认/Milovic 2019/7T iron 论文一致)。mu1 过小 ->
  ADMM 梯度约束失效 -> 反演退化成近乎无正则病态解 -> 噪声纹理。

### 修复(文献标定, 不瞎调)
- mu1 = 100*lambda, alpha1 = lambda(随最终 lambda 自动保持比例, 防外部覆盖失配)。
- lambda=4e-4, maxiter=150, beta=150, muh=10(WH-QSM 论文值; muh 之前误设 5)。
- whqsm_local_paths.m 注明 mu1 自动=100*lambda, 勿手改成 < lambda。

## v13.1 (2026-06-16) — 修复 BFR 调用逻辑(子函数不可见) + 移除破坏性解缠

### 真实根因(v13 调用失败)
- BFR 三函数(bg_removal_vsharp/lbv_medi/pdf_medi)原是 mod_background_removal.m 的
  【局部子函数】, 从 mod_field_preprocess.m 调用时 exist(...,'file')==2 恒为 false,
  导致三方法全被静默跳过 -> "全失败"。修复: 提取为 modules/ 下独立文件。
- 另一 bug: 自制 Laplacian "一致化" fallback 会抹平拟合频率场(std 2.37->0.28)。
  已移除; 多回波拟合频率场默认【不做】空间解缠(do_spatial_unwrap=false)。
- BFR 失败时现在打印每个方法的【真实错误原因】(不再静默吞 warning)。
- 已用合成数据验证 V-SHARP(自包含,无需 MEDI/SEPIA)能正确去背景、产生有效局部场。

## v13 (2026-06-16) — 补背景场去除(BFR) + 空间解缠: 根治苍白球偏低

### 诊断(代码对比 + 文献双证)
- 真实 DICOM pipeline 缺 QSM 共识规定的"背景场去除"必需步骤: loader 的 fieldmap_Hz
  是【总场】(且仅回波维解缠), 之前直接当局部场反演 -> 深部核团场被压平, 苍白球~0.0x。
- Challenge 数据正常是因其 phs_tissue 已是做完空间解缠+LBV 的局部场(输入口径不同)。
- 文献: ISMRM QSM 共识 2024; SEPIA Exercise3(BFR 后 GP/RN/SN 才可见);
  WH-QSM 配 LBV 最佳, 弱谐波项只处理残余背景场不可替代 BFR。

### 修复(不触碰反演)
- 新增 modules/mod_field_preprocess.m: 反演前独立阶段, 空间 Laplacian 解缠 +
  背景场去除(LBV 优先->V-SHARP->PDF, 复用库内 MEDI/SEPIA 封装) + 前后场图 QC。
- mod_whqsm_reconstruction.m 增加 cfg.whqsm.do_bfr 开关(默认 false);
  run_whqsm_comparison.m 对真实数据自动开启 do_bfr=true, bfr_method='LBV'。
- Challenge 路径(phs_tissue 局部场)默认不做 BFR, 不受影响。
- 文档: docs/FIELD_PREPROCESS_NOTES.md。

## v12 (2026-06-16) — 显示窗位收窄 + 深部核团 χ 数值核查工具

### 修正之前"全脑 p99 正常"的误判
- 全脑 p99(0.15) 被血管/边缘伪影撑高，不代表苍白球。用户观察"苍白球偏低"是对的。
- 收窄写死的显示窗位: compare_subjects/磁化率分离/方法对比图 ±0.15→±0.10、0-0.20→0-0.15
  (compare_subjects 改为读 P.qsmDisplayClim)。
### 新增 RUN_QSM_ROI_CHECK
- 客观核查深部核团 χ: 有 atlas(P.roiLabelFile) 则精确测 GP/Put 均值;
  无 atlas 则导出 chi.nii 供手动画 ROI + 全脑直方图(不做不可靠的自动圈核)。
- 诚实说明: 苍白球与血管易混，无 atlas 时自动分离不可靠，给出可靠测量路径。
- lambda 仍保持 3e-4(数据证明空间模式正确,仅幅值需按 ROI 数值决定是否再降)。

## v11 (2026-06-15) — 修正 opt vs onnx 不公平对比(R2'口径) + 打印 spatial_res

### 审慎复核结论修正(此前"纯方法差异"说法不准确)
- 真正主因: onnx(r2s) 用 R2PRIMEnet 把 R2* 学习成"真 R2'"(去掉 R2 基线)，
  而 opt 此前直接拿 raw R2* 当 R2' → 被 R2 基线整体抬高 ~2.9x。这是我引入的
  口径不一致，不是单纯"方法本质差异"。
- 修复: 对比模块先跑 onnx、捕获其 R2PRIMEnet 估计的 R2'(已改为输出 ppm)，
  再喂给 opt(cfg.sep.opt_r2prime_ppm)，两法用同一 R2'，公平对比。
  opt 量级预期下降 ~2.9x，与 onnx 接近;剩余差异才是真正的方法差异。
- bridge: r2prime_map 现以 ppm(物理量)输出(原为归一化值)，可被 opt/QC 直接使用。
- adapter: 打印 data.spatial_res 与 matrix size;opt 打印 R2' 来源与 R2'(ppm) 统计。

## v10 (2026-06-15) — 修复 onnx "中间一块/周围全黑"(分辨率泛化)

### 现象
- onnx 输出仅覆盖脑中央方块，四周全黑；opt 覆盖全脑。
### 根因
- 网络固定 192×192×128，数据 336×384×96 被直接中心裁剪 → 两侧大块脑组织被裁掉。
### 修复(严格复刻官方 resolution-generalization, Ji 2023)
- 桥接新增: ①脑 bounding-box 裁背景 → ②k-space 重采样到 1mm 等效矩阵(+Tukey 窗)
  使其装入网络尺寸 → ③补/裁到网络固定尺寸 → 推理 → 逐步逆变换还原。
- 自动判定: 体素≠1mm 或 bbox 后仍超网络尺寸 → 启用 k-space 缩放。
- 新增 `--voxel_size`(从 data.spatial_res 传入) 与 `--resgen auto|on|off`、
  对应 `P.onnxResgen`。
- 合成验证: 1mm 大 FOV 与 0.5mm 高分辨率两种情况脑覆盖率均 99.7%(原为中央方块)。
- opt 与 onnx R2' 来源本就一致(均 max(R2*-R2,0)/Dr，无 R2 时同用 R2*/Dr)；
  两者量级差异属方法本质差异(opt 线性放大 R2*，深度网络更稀疏)，非 bug。

## v9 (2026-06-15) — 修复 opt 尺度bug + 根治 .mat 损坏(原子保存) + 尺寸处理依据

### 修复传统优化(opt)磁化率被严重低估(图全黑/偏淡)
- 根因: Python 桥接已把 R2' 转 ppm(=x_pos+x_neg)，opt 求解器却又按
  `Dr*(x_pos+x_neg)=R2'` 多除一次 Dr，导致 t 被压低 ~Dr(114)倍。
- 修复: 数据项改为 `||(x_pos+x_neg) - r2prime_ppm||^2`(去掉多余 Dr)。
  合成体模验证: xpos p99 从 0.06 → 0.14(真值 0.15)，尺度恢复正确。
- 注意: 无 SE 的 R2 时用 R2* 作 pseudo-R2'，opt 仍会偏大(数据限制，已标注)。

### 根治 .mat 损坏(不是简单跳过 — 修保存逻辑)
- 你指出"损坏文件是本库 QSM Pipeline 输出的"，确为保存逻辑隐患:
  大 v7.3 写入被中断/磁盘瞬时满会留下半截 HDF5 → "文件可能已损坏"。
- 新增 `Utils_self/save_mat_atomic.m`: 原子+校验保存(写临时→whos校验→原子改名)，
  保证 complete.mat 永不半截。`run_whqsm_comparison.m` 两处大保存已改用它。
- `run_chisep_only_impl.m`: complete.mat 损坏时自动从 chi_*.mat + data_full.mat
  恢复(recover_subject_from_parts)，否则给出明确重跑指引(不再裸崩)。
- 新增 `DIAGNOSE_MAT_INTEGRITY.m`: 体检所有 .mat，报告损坏/可恢复性。

### 尺寸处理有据可循(回应"别手搓")
- onnx 固定 192×192×128 的居中裁/补，正是官方 χ-sepnet 论文做法
  (k-space 裁/补处理分辨率；同分辨率仅矩阵尺寸差异时图像域居中裁/补等价)。
  已在代码注释引用 Kim et al., HBM 2025。

## v8 (2026-06-15) — 修复固定尺寸模型 + 损坏被试跳过

### 关键修复：onnx 固定输入尺寸
- 现象: `Got invalid dimensions for input index 2 Got 336 Expected 192`。
- 根因: SNU 的 onnx 是固定尺寸(192×192×128)导出，非动态轴；原来的"裁到16倍数"
  不适用。
- 修复: 桥接读取模型固定尺寸 → 输入居中裁剪/补零到该尺寸 → 推理 → 还原回原尺寸。
  动态模型仍回退裁16倍数。已用 336×384×96 实测全通过。

### 健壮性
- `run_chisep_only_impl.m`: 单个被试加载/处理失败(如 complete.mat 损坏)时
  告警并跳过，继续处理其余被试，不再整体崩溃。
- 文档澄清: 仅有 R2* 时走 r2* 流程是正确行为；有 SE 的 R2 可切 r2' 流程。

## v7 (2026-06-15) — 总入口 RUN_ALL_ONECLICK + 保留单独分离能力

- 新增根入口 `RUN_ALL_ONECLICK.m`：一步完成 WH-QSM → 磁化率分离对比 → 两被试
  配准比较。支持 `'from'/'to'` 分段运行（已有预处理数据时可从 'chisep' 开始），
  支持 `'subject'` 过滤；任一阶段失败告警续跑。
- `run_chisep_only_impl.m` 增加 `P.chisepRunMethodCompare` 开关：RUN_CHISEP_ONLY
  既可"只跑单一方法分离"，也可"跑方法对比"。保留了对已有 whqsm_*_complete.mat
  的单独分离能力（无需重跑 DICOM/WH-QSM）。

## v6 (2026-06-15) — 方法对比 + 两被试配准比较 + 统一 QSM 来源 + 正则化调参

### 统一 QSM 来源（χ-sepnet 的 QSM 输入可配置）
- `P.onnxQsmSource = 'qsmnet'|'external'`。选 'external' 时直接用主流程 WH-QSM
  结果作为 χ-sepnet 的 QSM 输入（用 cosmos_sus_mean/std 归一化），数据复用库内
  `whqsm_*_complete.mat` 的 chi，无需额外准备。已端到端验证。

### 正则化调参（治"脑内偏淡"）
- 默认 `P.whqsmLambda` 5e-4 → 3e-4；显示窗位 ±0.12 → ±0.10（`P.qsmDisplayClim`）。
- 新增 `RUN_WHQSM_LAMBDA_SWEEP.m`：lambda 扫描 + ROI/直方图统计，客观选参。

### 方法对比：深度学习 vs 传统优化
- 新增 `snu_chisep_optimization_adapter.m`：传统凸优化 χ-separation（iLSQR/MEDI），
  不依赖 ONNX，原生 MATLAB。合成体模验证：与真值相关 0.99/1.00，符号/尺度正确。
- 新增 `mod_chisep_method_comparison.m` + 根 `RUN_CHISEP_COMPARE.m`：同被试同时跑
  深度学习与优化两法，输出并排图、统计、体素相关。

### 两被试(59 vs 72)配准后比较
- 新增 `mod_two_subject_registered_compare.m` + 根 `RUN_TWO_SUBJECT_COMPARE.m`：
  用 Image Processing Toolbox（imregtform/imwarp，已依赖）配准后做有效体素差异 +
  统计 + 可选 ROI。补足原 compare_subjects.m 刻意回避的"配准后相减"。

## v5 (2026-06-15) — chi-separation via ONNX Runtime（绕过 MATLAB onnxmex）

### 背景问题
- 部分 MATLAB/Windows 环境运行 SNU χ-separation 报：
  `onnxmex.mexw64 无效: 找不到指定的程序`。
- 根因是 MATLAB `importNetworkFromONNX/importONNXNetwork` 依赖的 MEX/VC++ 损坏，
  与模型无关（SNU-LIST issue #5 同款）。多次修 DLL 未果。

### 解决：纯 onnxruntime 桥接，完全绕过 MATLAB ONNX 导入
- 新增 `modules/DL/python/infer_chisep_from_mat.py`：用 onnxruntime 直接加载
  同一批 `.onnx`（QSMnet / chi-sepnet / R2PRIMEnet），严格复刻官方 chi_sepnet
  推理（单位换算、crop_img_16x、归一化、通道顺序 [QSM,field,R2']、反归一化、零截断）。
- 新增 `modules/snu_chisep_onnxruntime_adapter.m`：与 `snu_chisep_v121_adapter`
  同签名的 MATLAB 适配器，MATLAB 写 .mat → 调 Python → 读回结果，支持 r2'/r2* 双流程。
- `whqsm_local_paths.m` 新增开关 `P.useOnnxRuntimeChiSep`（默认 true）及 ONNX 桥接配置。
- `run_whqsm_comparison.m` / `RUN_WHQSM_ONECLICK.m` / `run_chisep_only_impl.m`
  透传 ONNX 桥接参数。
- 新增 `DIAGNOSE_CHISEP_ONNXRUNTIME.m`（环境/模型诊断）、
  根目录 `run_chisep_onnx_smoketest.m`（合成数据端到端冒烟测试）、
  根目录 `RUN_CHISEP_ONNXRUNTIME.m`（一键：诊断+冒烟+真实数据）。
- 文档：`docs/CHISEP_ONNXRUNTIME_BRIDGE.md`。
- 回退：`P.useOnnxRuntimeChiSep=false` 即切回原 p-code adapter，下游不变。

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

## v22 (2026-06-16) — χ-sep QC 选层自动定位到基底节层
- 根因：`mod_susceptibility_separation.m` 的 QC 图用脑mask几何质心 `sz=round(mean(z))` 选轴位层，
  偏腹侧/颅底，切不到苍白球/壳核 → χ-sep 最该展示的深部顺磁对比缺席，整图"看着淡/可疑"。
  （算法、数值、分离一致性 χ_para+χ_dia≈χ_total 经核对全部正常，非算法bug。）
- 修复：新增 `select_basal_ganglia_slice(chi_para, mask, sz_fallback)`：
  1) 腐蚀脑mask剔除皮层/边缘/静脉壁强信号(`erode_mask_3d`, 不依赖image toolbox)；
  2) 在解剖先验范围 z∈[0.30,0.75]*Nz 内，按深部mask内 χ_para 高分位(p85)以上信号积分打分；
  3) 取得分最大层为基底节层；空/异常回退几何质心。
- 已用 Python 合成体模验证：几何质心 z=20，新选层正确跳到深部顺磁源附近 z=22(真值24±3blob内)，
  且成功忽略皮层rim。日志会打印 `QC 选层(基底节自动定位): z=.. (脑质心 z=..)`。
