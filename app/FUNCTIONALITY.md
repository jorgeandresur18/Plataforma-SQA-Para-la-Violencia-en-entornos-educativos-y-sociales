# Manual de Funcionamiento: Algoritmos y Procesos

Este documento detalla la lógica matemática y funcional implementada en el motor `sqa_engine.py`.

## 1. Algoritmo de Conversión Neutrosófica

**Objetivo**: Transformar datos binarios o de texto en lógica neutrosófica de tres valores.

**Mapeo Lógico**:

| Valor Original | T | I | F | Interpretación |
|----------------|-----|-----|-----|----------------|
| `1`, "Yes", "Always", "True" | 0.9 | 0.05 | 0.05 | Alta Verdad |
| `0`, "No", "Never", "False" | 0.05 | 0.05 | 0.9 | Alta Falsedad |
| "Sometimes", "Rarely" | 0.45 | 0.6 | 0.45 | Indeterminación |
| Otros/Desconocidos | 0.3 | 0.4 | 0.3 | Neutro |

> [!NOTE]
> El algoritmo detecta valores en inglés y español (ej. "Si", "Sí", "Nunca").

---

## 2. Análisis Comparativo (Componente T)

**Objetivo**: Evaluar influencia de cada factor comparando grupos.

**Proceso**:
1. Dividir dataset según Variable Objetivo ($Y$):
   - **Grupo Positivo**: Casos donde $Y=1$
   - **Grupo Negativo**: Casos donde $Y=0$
2. Calcular **Promedio de Verdad** ($Mean\_T$) por factor en cada grupo.

---

## 3. Clasificación de Relevancia

**Cálculo**:
$$\Delta T = Mean\_T(Positivo) - Mean\_T(Negativo)$$

**Reglas**:
| Condición | Clasificación |
|-----------|---------------|
| $\Delta T \ge 0.15$ | **Alta Relevancia** ✅ |
| $0.05 \le \Delta T < 0.15$ | **Media Relevancia** ⚠️ |
| $\Delta T < 0.05$ | **Baja Relevancia** ❌ |

---

## 4. Detección de Patrones

**Objetivo**: Encontrar combinaciones de factores frecuentes en casos positivos.

**Lógica**:
1. Filtrar filas donde $Y=1$
2. Identificar factores con $T > 0.7$
3. Crear "Configuración Causal" (ej. "Alcoholismo & Desempleo")
4. Reportar **Top 5** por frecuencia

---

## 5. Exportación a PDF

**Contenido del Reporte**:
1. **Metadatos**: Fecha, archivo, variable objetivo, factores
2. **Tabla de Análisis**: Factor, T(+), T(-), Delta, Relevancia
3. **Gráfico Comparativo**: Barras agrupadas con Plotly
4. **Patrones Detectados**: Top configuraciones frecuentes
5. **Interpretación**: Resumen automático de hallazgos

**Características**:
- Orientación horizontal (landscape) para tablas anchas
- Nombres descriptivos del diccionario de datos
- Gráfico embebido como imagen PNG
