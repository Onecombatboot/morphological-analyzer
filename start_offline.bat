@echo off
REM ---------------------------------------------------------------------------
REM  Starts the application for an OFFLINE / JAVA-ONLY machine.
REM
REM  Use this instead of start.bat where Python is not available.
REM
REM  Nothing here needs Python, pip, the internet, or a GPU. Java 21 and the files
REM  in this folder are the whole requirement.
REM
REM  Two models run, both in process, both on the CPU:
REM
REM    llama.cpp + Qwen2.5-7B GGUF   drives the interface's full analysis. It
REM                                  works from value NAMES alone, which is what
REM                                  the interface collects.
REM
REM    ONNX + the fine-tuned 3B      the model this project produced, reached
REM                                  through /api/matrix/cca-score and
REM                                  /api/matrix/cca-model. It needs each value's
REM                                  REQUIRES and PROVIDES, so it is driven by API
REM                                  rather than from the interface.
REM
REM  Differences from start.bat, and why each one matters:
REM
REM    1. It does not start the Python scoring service, and does not wait three
REM       minutes for a service that cannot come up on this machine.
REM
REM    2. It passes --gma.llm.enabled=true. start.bat passes FALSE, which turns
REM       off the only engine available here. That single flag is the difference
REM       between a working assessment and an empty one.
REM
REM    3. It passes --gma.llm.suppress-thinking=true. The model deliberates by
REM       default, which on CPU costs hundreds of extra tokens for a sixteen-cell
REM       answer and is the single largest cost in a run.
REM
REM    4. It pins the thread count. llama.cpp otherwise picks for itself, and a
REM       fixed count is also what makes verdicts byte-reproducible between runs.
REM ---------------------------------------------------------------------------
setlocal

set ROOT=%~dp0
set APP_PORT=8080
if not "%~1"=="" set APP_PORT=%~1

REM Physical cores. Override by passing a second argument: start_offline.bat 8080 4
set THREADS=8
if not "%~2"=="" set THREADS=%~2

echo.
echo  ============================================================
echo   Morphological Analysis Studio  --  offline / Java-only
echo  ============================================================
echo.
echo   port     : %APP_PORT%
echo   threads  : %THREADS%
echo   engines  : llama.cpp + fine-tuned ONNX (both in-process, CPU)
echo.

java -version >nul 2>&1
if errorlevel 1 (
    echo  [!] Java was not found on PATH.
    echo      This application needs a Java 21 runtime and nothing else.
    echo      See SETUP.md, "Running with Java only".
    echo.
    pause
    exit /b 1
)

if not exist "%ROOT%morphological-analyzer\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" (
    echo  [!] The model file is missing:
    echo      morphological-analyzer\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf
    echo.
    echo      Run  java Restore.java  again -- this file is split across the last
    echo      two archive volumes and both are needed to rebuild it.
    echo.
    pause
    exit /b 1
)

if not exist "%ROOT%morphological-analyzer\models\cca-int4\model.onnx" (
    echo  [!] The fine-tuned model is missing:
    echo      morphological-analyzer\models\cca-int4\model.onnx
    echo.
    echo      The application will still start and the interface will still work,
    echo      but /api/matrix/cca-score and /api/matrix/cca-model will report 503.
    echo      Run  java Restore.java  again to rebuild it.
    echo.
)

echo  [1/2] Starting the application ...
echo        The model is 4.4 GB and is read from disk on startup; the first
echo        run takes about a minute longer than later ones.
echo.

start "GMA application" /MIN /D "%ROOT%morphological-analyzer" cmd /c "java -Xmx8g -jar target\morphological-analyzer-0.0.1-SNAPSHOT.jar --server.port=%APP_PORT% --gma.llm.enabled=true --gma.llm.suppress-thinking=true --gma.llm.threads=%THREADS% --gma.onnxcca.threads=%THREADS%"

echo        Waiting for it to come up ...
set /a TRIES=0
:APPLOOP
timeout /t 5 /nobreak >nul
curl -s -m 3 http://127.0.0.1:%APP_PORT%/api/status >nul 2>&1
if %ERRORLEVEL%==0 goto APPUP
set /a TRIES+=1
if %TRIES% GEQ 36 goto APPTIMEOUT
echo        still starting ...
goto APPLOOP

:APPTIMEOUT
echo.
echo  [!] The application did not answer within three minutes.
echo      If port %APP_PORT% is in use, start again on another one:
echo          start_offline.bat 8090
echo.
pause
exit /b 1

:APPUP
echo        Application ready.
echo.
echo  [2/2] Engine in use:
curl -s http://127.0.0.1:%APP_PORT%/api/status
echo.
echo.
echo   Confirm BOTH of these appear above:
echo     "llm":{"enabled":true,"loaded":true        (interface engine)
echo     "fineTunedCca":{"loaded":true              (the fine-tuned model)
echo   If either says loaded:false, that model file did not load -- see SETUP.md.
echo.
echo  ============================================================
echo   Open:  http://localhost:%APP_PORT%
echo  ============================================================
echo.
echo   A full four-parameter box takes roughly two minutes on eight cores.
echo   Leave this window open; closing it stops the application.
echo.
pause
