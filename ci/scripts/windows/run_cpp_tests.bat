@echo off
for %%X in (.\csp\tests\bin\*.exe) do (
    echo Executing: %%X
    call %%X
    if errorlevel 1 (
        echo Failed to execute: %%X
	exit /b 1
    )
)