# Arquitectura de la Plataforma SQA

## Visión General
La "Plataforma Digital de Análisis Causal Neutrosófico (SQA)" es una aplicación web monolítica construida en Python, diseñada para procesar datos tabulares y aplicar lógica neutrosófica para descubrir patrones causales en fenómenos sociales.

## Stack Tecnológico

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| **Lenguaje** | Python 3.9+ | Núcleo lógico y ejecución |
| **Frontend** | Streamlit | Interfaz de usuario interactiva |
| **Procesamiento** | Pandas & NumPy | Manipulación de datos |
| **Visualización** | Plotly Express | Gráficos interactivos |
| **Lectura Archivos** | openpyxl, odfpy, pyreadstat | Soporte multi-formato |
| **Generación PDF** | fpdf2, kaleido | Exportación de reportes |

## Estructura del Proyecto

```
/app
├── app.py              # Capa de Presentación (Frontend)
│                       # - Configuración de página
│                       # - Carga de archivos (CSV, Excel, ODS, SPSS, Stata, JSON)
│                       # - Carga de diccionario de datos
│                       # - Generación de reportes PDF
│                       # - Orquestación del motor
│                       # - Renderizado de gráficos y tablas
│
├── sqa_engine.py       # Capa de Negocio (Backend Logic)
│                       # - Algoritmo 1: Conversión Neutrosófica
│                       # - Algoritmo 2: Resumen Comparativo
│                       # - Algoritmo 3: Clasificación de Relevancia
│                       # - Algoritmo 4: Detección de Patrones
│
├── requirements.txt    # Gestión de Dependencias
│
├── csv/                # Datasets de ejemplo (CSV)
├── xlsx/               # Datasets de ejemplo (Excel)
│
└── venv/               # Entorno Virtual (Local)
```

## Flujo de Datos

1. **Entrada (Input)**:
   - El usuario carga un archivo (CSV, Excel, ODS, SPSS, Stata, JSON)
   - Opcionalmente carga un diccionario de datos para nombres descriptivos
   - Los datos se almacenan en `st.session_state`

2. **Procesamiento (Engine)**:
   - `app.py` invoca funciones de `sqa_engine.py`
   - **Transformación**: DataFrame binario → DataFrame Neutrosófico (T, I, F)
   - **Cálculo**: Agregaciones y operaciones vectorizadas para ΔT
   - **Diccionario**: Los códigos se traducen a nombres descriptivos

3. **Salida (Output)**:
   - **Interfaz**: Tablas con nombres descriptivos, gráficos Plotly
   - **Exportación**: Reporte PDF con tabla, gráfico y patrones

## Componentes Principales

### Cargador Universal de Archivos
Función `load_file_to_dataframe()` que soporta:
- CSV/TSV con auto-detección de delimitadores y codificación
- Excel (.xlsx, .xls) via openpyxl
- ODS via odfpy
- SPSS (.sav) y Stata (.dta) via pyreadstat
- JSON

### Diccionario de Datos
- Carga opcional de archivo con mapeo código → descripción
- Función `get_column_label()` para traducir códigos
- Aplica a selectores, tablas y gráficos

### Generador de PDF
- Función `generate_pdf_report()` usando fpdf2
- Incluye: metadatos, tabla de análisis, gráfico, patrones, interpretación
- Gráficos exportados como imagen con kaleido
