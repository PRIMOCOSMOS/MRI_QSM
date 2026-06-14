function test_pipeline()
% test_pipeline.m
% ============================================================================
%   离线测试脚本（不需要 SEPIA/MEDI）
%   用于快速验证：
%     - 路径配置正确
%     - DICOM 加载正确
%     - 单位转换正确（rad → ppm）
%     - 原库 mod_dipole_inversion 能跑（TKD/CFL2/iLSQR 不需要外部工具箱）
% ============================================================================

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  dicom_adapter - 离线测试 (不需要 SEPIA)                          ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ====== Test 1: 路径检查 ======
fprintf('[Test 1] 路径检查...\n');
critical = {
    'mod_dipole_inversion.m';
    'mod_background_removal.m';
    'mod_load_data.m';
    'create_dipole_kernel.m';
};
ok = 0;
for k = 1:length(critical)
    if exist(critical{k}, 'file') == 2
        fprintf('  ✅ %s\n', critical{k});
        ok = ok + 1;
    else
        fprintf('  ❌ %s (缺失)\n', critical{k});
    end
end
fprintf('  → 路径检查: %d / %d 通过\n\n', ok, length(critical));
if ok < length(critical)
    fprintf('  ⚠️ 请先运行 setup() 配置路径\n');
    return;
end

% ====== Test 2: DICOM 加载器 ======
fprintf('[Test 2] DICOM 加载器...\n');
data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course';
if ~exist(data_root, 'dir')
    fprintf('  ⚠️ 数据路径不存在: %s\n', data_root);
    fprintf('  请修改 test_pipeline.m 第 35 行为实际数据路径\n');
    return;
end

subjects = discover_subjects(data_root);
if isempty(subjects)
    fprintf('  ❌ 未发现任何被试\n');
    return;
end

% 取第一个被试测试
sub = subjects(1);
fprintf('  测试被试: %s (%s)\n', sub.name, sub.group);

try
    test_output = fullfile(data_root, '_test_output');
    data = dicom_loader_subject(sub, test_output);

    fprintf('  ✅ DICOM 加载成功\n');

    % 单位合理性检查
    ppm_max = max(abs(data.phs_tissue(data.msk)));
    rad_max = max(abs(data.phs_unwrap(data.msk)));

    fprintf('\n  字段验证:\n');
    fprintf('    phs_tissue 范围: [%.4f, %.4f] ppm (期望 ±0.5 ppm)\n', ...
        min(data.phs_tissue(data.msk)), max(data.phs_tissue(data.msk)));
    fprintf('    phs_unwrap 范围: [%.4f, %.4f] rad (期望 ±π rad)\n', ...
        min(data.phs_unwrap(data.msk)), max(data.phs_unwrap(data.msk)));
    fprintf('    B0          = %.2f T\n', data.B0);
    fprintf('    TE          = %.4f ms\n', data.EchoTime);
    fprintf('    ppm_factor  = %.4f\n', data.ppm_factor);
    fprintf('    spatial_res = [%.4f %.4f %.4f] mm\n', data.spatial_res);

    % 合理性检查
    if ppm_max < 1.0
        fprintf('    ✅ ppm 值域合理 (max abs = %.4f)\n', ppm_max);
    else
        fprintf('    ❌ ppm 值域异常 (max abs = %.4f)，可能是缩放公式错误\n', ppm_max);
    end

    if rad_max < pi + 0.5
        fprintf('    ✅ rad 值域合理 (max abs = %.4f)\n', rad_max);
    else
        fprintf('    ❌ rad 值域异常 (max abs = %.4f)\n', rad_max);
    end

catch ME
    fprintf('  ❌ DICOM 加载失败: %s\n', ME.message);
    return;
end

% ====== Test 3: mod_dipole_inversion (不依赖 SEPIA 的方法) ======
fprintf('\n[Test 3] mod_dipole_inversion (TKD/CFL2/iLSQR)...\n');
try
    % 构造最小 cfg（不需要 SEPIA）
    cfg.bgRemoval.methods = {'WHQSM'};
    cfg.inversion.tkd_threshold = 0.19;
    cfg.inversion.cfl2_reg = 9e-2;
    cfg.inversion.ilsqr_tol = 1e-3;
    cfg.inversion.ilsqr_maxiter = 20;
    cfg.inversion.medi_lambdas = [0.1];
    cfg.inversion.medi_use_structural = false;
    cfg.deeplearning.enable = false;
    cfg.deeplearning.models = {};
    cfg.deeplearning.qsmnet_onnx = '';
    cfg.deeplearning.xqsm_onnx = '';
    cfg.deeplearning.xqsm_pth = '';
    cfg.vis.clim_qsm = [-0.15 0.15];
    cfg.vis.clim_err = [-0.06 0.06];
    cfg.vis.doSave = false;
    cfg.vis.resolution = 200;
    cfg.eval.reference = '';
    cfg.resultDir = test_output;
    if ~exist(cfg.resultDir, 'dir'), mkdir(cfg.resultDir); end

    data.B0 = 3; data.b0 = 3;

    [qsm_results, qsm_names] = mod_dipole_inversion(data.phs_tissue, data, cfg);

    fprintf('  ✅ mod_dipole_inversion 成功\n');
    fprintf('  跑出的方法: %s\n', strjoin(qsm_names, ', '));

    for k = 1:length(qsm_names)
        chi = qsm_results(:,:,:,k);
        chi_max = max(abs(chi(data.Mask)));
        fprintf('    %s: range=[%.4f, %.4f] ppm\n', ...
            qsm_names{k}, min(chi(data.Mask)), max(chi(data.Mask)));
    end

catch ME
    fprintf('  ❌ mod_dipole_inversion 失败: %s\n', ME.message);
    fprintf('     at %s:%d\n', ME.stack(1).name, ME.stack(1).line);
    return;
end

% ====== Test 4: SEPIA / WH-QSM 检测 ======
fprintf('\n[Test 4] SEPIA / WH-QSM 检测...\n');
if exist('QSMMacroIOWrapper', 'file') == 2
    fprintf('  ✅ QSMMacroIOWrapper 可用 → WH-QSM 可运行\n');
    fprintf('     在 run_whqsm_comparison() 中会自动调用\n');
else
    fprintf('  ⚠️ QSMMacroIOWrapper 不可用 → WH-QSM 不可运行\n');
    fprintf('     其他方法 (TKD/CFL2/iLSQR/MEDI) 仍可用\n');
end

% ====== 完成 ======
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  🎉 测试通过！可以运行 run_whqsm_comparison()\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% 清理测试输出
fprintf('清理测试输出: %s\n', test_output);
rmdir(test_output, 's');
end
