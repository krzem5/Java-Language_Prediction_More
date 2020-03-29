echo off
echo NUL>_.class&&del /s /f /q *.class
cls
javac com/krzem/language_prediction_more/Main.java&&java com/krzem/language_prediction_more/Main
start /min cmd /c "echo NUL>_.class&&del /s /f /q *.class"