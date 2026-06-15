function check_and_force_update()
% check_and_force_update.m (v4)
% ============================================================================
% Verify that MATLAB path resolves to the v4 WH-QSM-only DICOM loader and
% reconstruction module. Useful after pulling/updating files.
% ============================================================================

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Verify WH-QSM-only v4 files on MATLAB path                 ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

rehash;
clear functions;

check_one('run_whqsm_comparison', 'WH-QSM-only real subject pipeline');
check_one('dicom_loader_subject', 'v4 - WH-QSM real-data loader');
check_one('discover_subjects', 'v5 - WH-QSM real-data robust discovery');
check_one('compare_subjects', 'WH-QSM QC summary without registration');
check_one('mod_whqsm_reconstruction', 'WH-QSM-only reconstruction module');

fprintf('\nIf any item is missing or does not contain the expected marker, run setup.m from MRI_QSM_dicom_adapter.\n\n');
end

function check_one(funcName, marker)
p = which(funcName);
if isempty(p)
    fprintf('❌ %-28s not found\n', funcName);
    return;
end
fprintf('✅ %-28s %s\n', funcName, p);
try
    txt = fileread(p);
    if contains(txt, marker)
        fprintf('   marker OK: %s\n', marker);
    else
        fprintf('   ⚠️ marker not found: %s\n', marker);
    end
catch ME
    fprintf('   ⚠️ could not read file: %s\n', ME.message);
end
end
