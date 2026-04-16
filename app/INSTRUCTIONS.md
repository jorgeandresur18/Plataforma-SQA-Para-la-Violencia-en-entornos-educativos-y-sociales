# Instrucciones de Instalación y Uso

## Requisitos Previos
- Python 3.9 o superior
- Terminal / Línea de comandos

## 1. Configuración del Entorno

**Paso 1: Abra su terminal en la carpeta del proyecto**
```bash
cd /ruta/hacia/app
```

**Paso 2: Crear el entorno virtual**
```bash
python3 -m venv venv
```

**Paso 3: Activar el entorno virtual**
- En **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```
- En **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```

**Paso 4: Instalar dependencias**
```bash
pip install -r requirements.txt
```

---

## 2. Ejecutar la Aplicación

Con el entorno virtual activado:
```bash
streamlit run app.py
```

Esto abrirá automáticamente `http://localhost:8501`.

---

## 3. Guía de Uso Rápido

### Cargar Datos
1. Use el panel lateral izquierdo
2. Suba su archivo de datos (formatos soportados):
   - CSV, TSV
   - Excel (.xlsx, .xls)
   - ODS (LibreOffice)
   - SPSS (.sav)
   - Stata (.dta)
   - JSON

### Cargar Diccionario (Opcional)
1. En la sección "Diccionario de Datos"
2. Suba archivo con mapeo código → descripción
3. La plataforma mostrará nombres descriptivos automáticamente

### Configurar Análisis
1. **Variable Objetivo**: Seleccione la columna efecto
2. **Factores Causales**: Seleccione columnas a analizar
   - ⚠️ Excluya columnas de ID, fechas o texto

### Navegar por Pestañas
| Pestaña | Contenido |
|---------|-----------|
| **Datos** | Vista previa con nombres descriptivos |
| **Neutrosofía** | Transformación a valores (T, I, F) |
| **Análisis Causal** | Tabla de relevancia + gráfico + **Descargar PDF** |
| **Patrones** | Top 5 configuraciones frecuentes |

### Exportar a PDF
1. Vaya a la pestaña **Análisis Causal**
2. Haga clic en **📄 Descargar Reporte PDF**
3. El archivo se descarga automáticamente

---

## 4. Solución de Problemas

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError` | Active el entorno virtual antes de ejecutar |
| Navegador no abre | Copie `http://localhost:8501` manualmente |
| Error de PDF | Verifique que kaleido esté instalado |
| Caracteres extraños | El archivo puede tener codificación diferente (se detecta automáticamente) |
| Columnas sin nombre | Se renombran automáticamente como `_Columna_N` |
