@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom start_windows.bat, aby utworzyc srodowisko aplikacji.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
set "MPLBACKEND=Agg"
if exist "WYNIKI_BADAWCZE.zip" move /Y "WYNIKI_BADAWCZE.zip" "WYNIKI_POPRZEDNIE.zip" >nul
echo Instalowanie bibliotek badawczych...
python -m pip install -r requirements-research.txt
if not "%ERRORLEVEL%"=="0" goto error
echo Pobieranie i przygotowanie pilotazowej probki PWLDS...
python scripts\prepare_pwlds.py --sample-size 100000
if not "%ERRORLEVEL%"=="0" goto error
echo Trenowanie i porownanie modeli...
python scripts\train_models.py
if not "%ERRORLEVEL%"=="0" goto error
python scripts\package_research_results.py
if not "%ERRORLEVEL%"=="0" goto error
echo.
echo GOTOWE. Model badawczy zostal zapisany w aplikacji.
pause
exit /b 0
:error
echo.
echo Wystapil blad.
pause
exit /b 1
