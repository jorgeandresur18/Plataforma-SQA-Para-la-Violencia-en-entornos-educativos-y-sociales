@echo off
echo =======================================================
echo Iniciando Plataforma SQA - Analisis Causal Neutrosofico
echo =======================================================
echo.

echo Activando el entorno virtual...
call venv\Scripts\activate.bat

echo Navegando a la carpeta de la aplicacion...
cd app

echo Iniciando Streamlit...
streamlit run app.py

pause
