@echo off
cls
if exist build rmdir /s /q build
mkdir build
cd src
javac -d ../build com/krzem/language_prediction_more/Main.java&&jar cvmf ../manifest.mf ../build/language_prediction_more.jar -C ../build *&&goto run
cd ..
goto end
:run
cd ..
pushd "build"
for /D %%D in ("*") do (
	rd /S /Q "%%~D"
)
for %%F in ("*") do (
	if /I not "%%~nxF"=="language_prediction_more.jar" del "%%~F"
)
popd
cls
java -jar build/language_prediction_more.jar
:end
