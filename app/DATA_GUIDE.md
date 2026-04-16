# Guía de Datos para Plataforma SQA

## 1. Formatos de Archivo Soportados

La plataforma acepta múltiples formatos de datos:

| Formato | Extensiones | Notas |
|---------|-------------|-------|
| CSV | `.csv` | Delimitadores `,` `;` auto-detectados |
| TSV | `.tsv` | Tabulador como delimitador |
| Excel | `.xlsx`, `.xls` | Requiere openpyxl |
| ODS | `.ods` | LibreOffice/OpenOffice |
| SPSS | `.sav` | Con metadatos preservados |
| Stata | `.dta` | Con etiquetas de variables |
| JSON | `.json` | Formato tabular |

## 2. Tipos de Valores Aceptados

| Tipo | Ejemplos | Conversión Neutrosófica |
|------|----------|-------------------------|
| **Binario Numérico** | `1`, `0` | `1` → T=0.9; `0` → F=0.9 |
| **Texto Afirmativo** | "Yes", "Si", "True", "Always" | → T=0.9 |
| **Texto Negativo** | "No", "Never", "False" | → F=0.9 |
| **Texto Parcial** | "Sometimes", "Rarely" | → T=0.45, I=0.6 |
| **Otros** | Texto no reconocido | → T=0.3, I=0.4, F=0.3 |

> [!TIP]
> La plataforma convierte automáticamente valores de texto. No necesitas preprocesar tus datos.

---

## 3. Diccionario de Datos (Opcional)

Para mejorar la legibilidad, puedes cargar un **diccionario de datos** que traduce códigos de columnas a nombres descriptivos.

### Formato del Diccionario

El diccionario debe tener al menos 2 columnas:
- **Columna 1**: Código original de la variable
- **Columna 2**: Descripción o nombre legible

**Ejemplo:**
| Variable | Descripcion |
|----------|-------------|
| f4_s6b_8 | ¿Cuándo acudió a denunciar? |
| p03 | Relación de parentesco |
| area | Área geográfica (Urbano/Rural) |

### Beneficios
- Los selectores muestran nombres descriptivos
- Las tablas de resultados usan etiquetas legibles
- Los gráficos tienen leyendas claras
- El reporte PDF incluye nombres descriptivos

---

## 4. Variable Objetivo (Target)

La variable objetivo debería ser **binaria (0/1)**:
- ✅ `0` = Ausencia del fenómeno
- ✅ `1` = Presencia del fenómeno

> [!WARNING]
> Si tu variable tiene muchos valores únicos (ej. escalas 1-10), considera binarizarla. Ejemplo: valores ≥7 → `1`, valores <7 → `0`.

---

## 5. Selector de Factores Causales

En la barra lateral, usa el **multiselect** para elegir qué columnas analizar:
- ⚠️ Excluye columnas de ID (ej. `User_id`)
- ⚠️ Excluye columnas de fecha
- ⚠️ Excluye columnas de texto descriptivo

---

## 6. Fuentes de Datos Recomendadas

### Repositorios Públicos
1. **[Kaggle](https://www.kaggle.com/)**: Busca "School Violence", "Bullying"
2. **[UCI ML Repository](https://archive.ics.uci.edu/)**: Filtra por "Social Sciences"
3. **[Datos Abiertos Ecuador](https://www.datosabiertos.gob.ec/)**: INEC, ENEMDU

### Conversión de Escalas Likert (1-5)
- **Estricta**: 4-5 → `1`; 1-3 → `0`
- **Laxa**: 3-5 → `1`; 1-2 → `0`
