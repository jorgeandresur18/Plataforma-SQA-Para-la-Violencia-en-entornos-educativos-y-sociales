@echo off
setlocal

echo =======================================================
echo Iniciando Plataforma SQA - Analisis Causal Neutrosofico
echo =======================================================
echo.

:: Verificar que Python este instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado o no se encuentra en el PATH.
    echo Por favor, descarga e instala Python desde https://www.python.org/downloads/
    echo Asegurate de marcar la casilla "Add Python to PATH" durante la instalacion.
    pause
    exit /b
)

:: Revisar si existe el entorno virtual (la carpeta venv)
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] No se encontro el entorno virtual.
    echo Creando el entorno virtual por primera vez (esto puede tardar un momento)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Hubo un problema al crear el entorno virtual.
        pause
        exit /b
    )
    
    echo Activando el entorno virtual e instalando dependencias...
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r app\requirements.txt
) else (
    echo Activando el entorno virtual...
    call venv\Scripts\activate.bat
)

echo.
echo Navegando a la carpeta de la aplicacion...
cd app

echo Iniciando Streamlit...
streamlit run app.py

pause
