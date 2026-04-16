# DISEÑO DE UNA PLATAFORMA DIGITAL CONTRA LA VIOLENCIA BASADA EN EL ANÁLISIS CUALITATIVO COMPARATIVO NEUTROSÓFICO (SQA) PARA IDENTIFICAR FACTORES CAUSALES EN CONTEXTOS SOCIALES Y EDUCATIVOS

> **Proyecto de Tesis (2026)**  
> **Autores:** Jorge Urgilés y Leidy Bagua (Universidad de Guayaquil)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-Academic-green)

## 📋 Descripción

Esta plataforma implementa algoritmos de **Lógica Neutrosófica** para analizar relaciones causales en datos sociales (violencia escolar, estrés familiar, bullying, etc.). Transforma datos binarios o categóricos en triplets neutrosóficos (T, I, F) - Verdad, Indeterminación, Falsedad - permitiendo un análisis más matizado que la lógica binaria tradicional.

## ✨ Características

| Funcionalidad | Estado |
|---------------|--------|
| **Formatos de Archivo** | |
| CSV (coma, punto y coma, tabulador) | ✅ |
| Excel (.xlsx, .xls) | ✅ |
| ODS (LibreOffice/OpenOffice) | ✅ |
| SPSS (.sav) | ✅ |
| Stata (.dta) | ✅ |
| JSON | ✅ |
| **Procesamiento de Datos** | |
| Auto-detección de delimitadores | ✅ |
| Auto-detección de codificación (UTF-8, Latin-1) | ✅ |
| Conversión texto → binario (Yes/No, Si/No) | ✅ |
| **Diccionario de Datos** | |
| Carga de diccionario de variables | ✅ |
| Nombres descriptivos en selectores | ✅ |
| Nombres descriptivos en tablas y gráficos | ✅ |
| **Algoritmos Neutrosóficos** | |
| Algoritmo 1: Conversión Neutrosófica | ✅ |
| Algoritmo 2: Resumen Comparativo | ✅ |
| Algoritmo 3: Clasificación de Relevancia | ✅ |
| Algoritmo 4: Detección de Patrones | ✅ |
| **Visualización y Exportación** | |
| Gráficos interactivos con Plotly | ✅ |
| Exportación a PDF | ✅ |

## 🚀 Instalación

```bash
# 1. Clonar o navegar al directorio
cd app

# 2. Crear entorno virtual
python3 -m venv venv

# 3. Activar entorno
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate  # Windows

# 4. Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

```bash
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`.

### Flujo de Trabajo

1. **Subir Datos**: CSV, Excel, ODS, SPSS, Stata o JSON
2. **Subir Diccionario** (opcional): Para ver nombres descriptivos en lugar de códigos
3. **Seleccionar Objetivo**: Variable dependiente (ej. `Violencia_Escolar`)
4. **Seleccionar Factores**: Variables independientes a analizar
5. **Explorar Pestañas**:
   - **Datos**: Vista previa del dataset con nombres descriptivos
   - **Neutrosofía**: Transformación T, I, F
   - **Análisis Causal**: Tabla de relevancia + gráfico + **botón de descarga PDF**
   - **Patrones**: Combinaciones de factores frecuentes

## 📊 Algoritmos Implementados

### 1. Conversión Neutrosófica
Transforma valores en triplets (T, I, F):

| Valor | T | I | F |
|-------|---|---|---|
| 1, Yes, Always | 0.9 | 0.05 | 0.05 |
| 0, No, Never | 0.05 | 0.05 | 0.9 |
| Sometimes, Rarely | 0.45 | 0.6 | 0.45 |
| Desconocido | 0.3 | 0.4 | 0.3 |

### 2. Resumen Comparativo
Calcula Mean_T para casos positivos vs negativos de la variable objetivo.

### 3. Clasificación de Relevancia
- **Alta**: ΔT ≥ 0.15
- **Media**: 0.05 ≤ ΔT < 0.15
- **Baja**: ΔT < 0.05

### 4. Detección de Patrones
Identifica combinaciones de factores (T > 0.7) frecuentes en casos positivos.

## 📁 Estructura del Proyecto

```
app/
├── app.py              # Frontend Streamlit + Generación PDF
├── sqa_engine.py       # Algoritmos neutrosóficos
├── requirements.txt    # Dependencias
├── csv/                # Datasets de ejemplo
├── xlsx/               # Datasets Excel
├── utils/
│   └── data_converter.py
├── ARCHITECTURE.md     # Documentación técnica
├── FUNCTIONALITY.md    # Manual de algoritmos
├── DATA_GUIDE.md       # Guía de datos
└── INSTRUCTIONS.md     # Manual de usuario
```

## 🔧 Dependencias

```
streamlit
pandas
numpy
matplotlib
plotly
openpyxl
odfpy
pyreadstat
fpdf2
kaleido
```

## 📈 Progreso del Desarrollo

### Fase 1: Core ✅
- [x] Estructura del proyecto
- [x] Implementación de 4 algoritmos
- [x] Interfaz Streamlit básica
- [x] Visualizaciones con Plotly

### Fase 2: Robustez ✅
- [x] Manejo de datos no binarios
- [x] Conversión automática de texto
- [x] Auto-detección de delimitadores CSV
- [x] Selector de factores causales

### Fase 3: Compatibilidad ✅
- [x] Soporte multi-formato (CSV, Excel, ODS, SPSS, Stata, JSON)
- [x] Auto-detección de codificación
- [x] Limpieza automática de columnas

### Fase 4: Usabilidad ✅
- [x] Diccionario de datos (nombres descriptivos)
- [x] Exportación a PDF
- [x] Interfaz usuario-amigable

### Fase 5: Próximos Pasos
- [ ] Guardar configuraciones de análisis
- [ ] Modo comparativo entre datasets
- [ ] Integración con bases de datos

## 📚 Referencias

- Tesis: "DISEÑO DE UNA PLATAFORMA DIGITAL CONTRA LA VIOLENCIA BASADA EN EL ANÁLISIS CUALITATIVO COMPARATIVO NEUTROSÓFICO (SQA) PARA IDENTIFICAR FACTORES CAUSALES EN CONTEXTOS SOCIALES Y EDUCATIVOS" - Jorge Urgilés & Leidy Bagua, Universidad de Guayaquil (2026)
- Smarandache, F. (1999). A Unifying Field in Logics: Neutrosophic Logic

## 📝 Licencia

Proyecto académico - Universidad.

---

**Última actualización**: Enero 2026
