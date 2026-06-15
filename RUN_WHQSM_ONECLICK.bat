@echo off
REM RUN_WHQSM_ONECLICK.bat
REM Double-click entry for the fixed-root real-data WH-QSM pipeline.
REM Requires MATLAB command "matlab" to be available on Windows PATH.

set ROOT=D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge

echo ============================================================
echo One-click real-data WH-QSM pipeline
echo ROOT = %ROOT%
echo ============================================================

if not exist "%ROOT%" (
    echo ERROR: ROOT does not exist: %ROOT%
    pause
    exit /b 1
)

cd /d "%ROOT%"
matlab -nosplash -r "try, RUN_REALDATA_WHQSM_ONECLICK; catch ME, disp(getReport(ME,'extended')); end"

pause
