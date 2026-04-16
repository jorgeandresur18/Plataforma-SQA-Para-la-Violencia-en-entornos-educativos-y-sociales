import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqa_engine
import ml_engine
import map_engine
import io
import tempfile
import os
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="Plataforma SQA - Análisis Neutrosófico",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for better chart visibility ---
st.markdown("""
<style>
    .stPlotlyChart {
        min-height: 500px !important;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }
    /* Metrics styling for both light and dark mode */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"] label {
        color: white !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    /* Custom info boxes */
    .split-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .split-box h3 {
        margin: 0;
        font-size: 16px;
        opacity: 0.9;
    }
    .split-box h1 {
        margin: 10px 0 0 0;
        font-size: 32px;
    }
    .train-box {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
    }
    .test-box {
        background: linear-gradient(135deg, #2196F3 0%, #03A9F4 100%);
    }
    .total-box {
        background: linear-gradient(135deg, #9C27B0 0%, #E91E63 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- PDF Report Generation ---
def generate_pdf_report(filename, target_var, target_label, selected_factors, 
                        df_analysis, neutro_df, summary_df, final_summary, patterns,
                        fig, dict_lookup, has_dict, ml_results=None):
    """
    Generates a PDF report with all analysis results.
    Returns bytes of the PDF file.
    """
    from fpdf import FPDF
    
    # Simple PDF without custom header/footer to avoid issues
    pdf = FPDF(orientation='L')  # Landscape for more width
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    def clean_text(text, max_len=100):
        """Clean and truncate text for PDF compatibility"""
        text = str(text)
        # Replace problematic characters
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N', 'ü': 'u', 'Ü': 'U',
            '¿': '?', '¡': '!', '"': '"', '"': '"', ''': "'", ''': "'"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace('\n', ' ').replace('\r', ' ')
        if len(text) > max_len:
            return text[:max_len-3] + '...'
        return text
    
    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 12, 'Plataforma SQA - Reporte de Analisis Neutrosofico', align='C')
    pdf.ln(15)
    
    # Metadata
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    pdf.ln()
    pdf.cell(0, 7, f'Archivo: {clean_text(filename, 80)}')
    pdf.ln()
    pdf.cell(0, 7, f'Variable objetivo: {clean_text(target_label, 80)}')
    pdf.ln()
    pdf.cell(0, 7, f'Factores analizados: {len(selected_factors)} | Registros: {len(df_analysis)}')
    pdf.ln(12)
    
    # Section 1: Analysis Summary Table
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '1. Resumen del Analisis Comparativo')
    pdf.ln(12)
    
    # Simple table approach - use wider columns in landscape
    pdf.set_font('Helvetica', 'B', 9)
    # Landscape A4 effective width ~277mm, use 260mm for table
    col_w = [130, 30, 30, 25, 45]  # Total = 260
    
    pdf.cell(col_w[0], 8, 'Factor', border=1, align='C')
    pdf.cell(col_w[1], 8, 'T(+)', border=1, align='C')
    pdf.cell(col_w[2], 8, 'T(-)', border=1, align='C')
    pdf.cell(col_w[3], 8, 'Delta', border=1, align='C')
    pdf.cell(col_w[4], 8, 'Relevancia', border=1, align='C')
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 8)
    for _, row in final_summary.iterrows():
        factor = str(row['Factor'])
        if has_dict:
            factor = str(dict_lookup.get(factor, factor))
        factor = clean_text(factor, 70)
        
        pdf.cell(col_w[0], 7, factor, border=1, align='L')
        pdf.cell(col_w[1], 7, f"{row['Mean_T_Pos']:.3f}", border=1, align='C')
        pdf.cell(col_w[2], 7, f"{row['Mean_T_Neg']:.3f}", border=1, align='C')
        pdf.cell(col_w[3], 7, f"{row['Delta_T']:.3f}", border=1, align='C')
        rel = str(row['Relevancia']).replace(' Relevancia', '')
        pdf.cell(col_w[4], 7, rel, border=1, align='C')
        pdf.ln()
    
    pdf.ln(10)
    
    # Section 2: Chart
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '2. Grafico Comparativo')
    pdf.ln(12)
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.write_image(tmp.name, width=1000, height=500, scale=2)
            pdf.image(tmp.name, x=20, w=250)
            os.unlink(tmp.name)
    except Exception:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 10, '(Grafico no disponible - instale kaleido)')
    
    # New page for patterns
    pdf.add_page()
    
    # Section 3: Patterns
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '3. Patrones Frecuentes Detectados')
    pdf.ln(12)
    
    if patterns is not None and not patterns.empty:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(200, 8, 'Configuracion', border=1, align='C')
        pdf.cell(40, 8, 'Frecuencia', border=1, align='C')
        pdf.ln()
        
        pdf.set_font('Helvetica', '', 8)
        for _, row in patterns.iterrows():
            config = str(row['Configuracion'])
            if has_dict and config != "Ninguno":
                parts = config.split(" & ")
                parts = [clean_text(dict_lookup.get(p.strip(), p.strip()), 30) for p in parts]
                config = " & ".join(parts)
            config = clean_text(config, 110)
            
            pdf.cell(200, 7, config, border=1, align='L')
            pdf.cell(40, 7, str(row['Frecuencia']), border=1, align='C')
            pdf.ln()
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 10, 'No se detectaron patrones frecuentes.')
    
    pdf.ln(15)
    
    # Section 4: Machine Learning Results (if available)
    if ml_results and ml_results.get('success'):
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, '4. Resultados de Machine Learning')
        pdf.ln(12)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 7, f'Tipo de tarea: {ml_results.get("task_type", "N/A").title()}')
        pdf.ln()
        split_info = ml_results.get('split_info', {})
        pdf.cell(0, 7, f'Division: {split_info.get("train_percentage", 80)}% Entrenamiento / {split_info.get("test_percentage", 20)}% Prueba')
        pdf.ln()
        pdf.cell(0, 7, f'Mejor modelo: {ml_results.get("best_model_name", "N/A")}')
        pdf.ln(10)
        
        # Model results table
        model_results = ml_results.get('model_results', [])
        if model_results:
            pdf.set_font('Helvetica', 'B', 9)
            if ml_results.get('task_type') == 'classification':
                cols = ['Modelo', 'Accuracy', 'Sensibilidad', 'Especificidad', 'F1', 'CV Mean']
                widths = [50, 35, 40, 40, 30, 35]
            else:
                cols = ['Modelo', 'MSE', 'RMSE', 'R2', 'CV Mean']
                widths = [50, 40, 40, 40, 40]
            
            for i, col in enumerate(cols):
                pdf.cell(widths[i], 8, col, border=1, align='C')
            pdf.ln()
            
            pdf.set_font('Helvetica', '', 8)
            for result in model_results:
                if ml_results.get('task_type') == 'classification':
                    vals = [result.get('Modelo', ''), 
                           f"{result.get('Accuracy', 0):.4f}",
                           f"{result.get('Sensibilidad', 0):.4f}",
                           f"{result.get('Especificidad', 0):.4f}",
                           f"{result.get('F1-Score', 0):.4f}",
                           f"{result.get('CV Mean', 0):.4f}"]
                else:
                    vals = [result.get('Modelo', ''),
                           f"{result.get('MSE', 0):.4f}",
                           f"{result.get('RMSE', 0):.4f}",
                           f"{result.get('R²', 0):.4f}",
                           f"{result.get('CV Mean', 0):.4f}"]
                
                for i, val in enumerate(vals):
                    pdf.cell(widths[i], 7, str(val), border=1, align='C')
                pdf.ln()
    
    # Section 5: Interpretation
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '5. Interpretacion')
    pdf.ln(12)
    
    alta = len(final_summary[final_summary['Relevancia'] == 'Alta Relevancia'])
    media = len(final_summary[final_summary['Relevancia'] == 'Media Relevancia'])
    baja = len(final_summary[final_summary['Relevancia'] == 'Baja Relevancia'])
    
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 7, f'Se analizaron {len(selected_factors)} factores causales.')
    pdf.ln()
    pdf.cell(0, 7, f'Alta Relevancia: {alta} | Media: {media} | Baja: {baja}')
    pdf.ln()
    pdf.cell(0, 7, 'Metodologia: Logica Neutrosofica (Verdad, Indeterminacion, Falsedad)')
    pdf.ln()
    
    return bytes(pdf.output())


# --- Helper Function: Get Column Labels from Dictionary ---
def get_column_label(col_name, dict_lookup, max_len=50):
    """
    Returns the descriptive label for a column from the dictionary.
    Falls back to original column name if not found.
    Truncates if too long.
    """
    label = dict_lookup.get(col_name, dict_lookup.get(str(col_name).strip(), col_name))
    if len(str(label)) > max_len:
        return str(label)[:max_len] + "..."
    return str(label)

def build_dict_lookup():
    """
    Builds a lookup dictionary from the loaded data dictionary.
    Returns empty dict if no dictionary is loaded.
    """
    if 'data_dict' not in st.session_state:
        return {}
    
    dict_df = st.session_state['data_dict']
    if dict_df.shape[1] < 2:
        return {}
    
    # First column = variable name, second column = description
    dict_col_name = dict_df.columns[0]
    dict_col_desc = dict_df.columns[1]
    
    lookup = dict(zip(
        dict_df[dict_col_name].astype(str).str.strip(),
        dict_df[dict_col_desc].astype(str)
    ))
    return lookup

# --- Sidebar ---
st.sidebar.title("Plataforma SQA")

# Supported file formats for statistics/data
SUPPORTED_FORMATS = ["csv", "xlsx", "xls", "ods", "tsv", "sav", "dta", "json"]

uploaded_file = st.sidebar.file_uploader(
    "📁 Subir Dataset", 
    type=SUPPORTED_FORMATS,
    help="Formatos soportados: CSV, Excel, ODS, TSV, SPSS (.sav), Stata (.dta), JSON"
)

# --- Data Dictionary Uploader ---
st.sidebar.markdown("---")
st.sidebar.markdown("**📖 Diccionario de Datos (Opcional)**")
dict_file = st.sidebar.file_uploader(
    "Subir diccionario de datos", 
    type=SUPPORTED_FORMATS, 
    key="dict_uploader",
    help="Formatos: CSV, Excel, ODS, TSV, SPSS, Stata, JSON"
)

def load_file_to_dataframe(file_obj, show_messages=True):
    """
    Universal file loader supporting multiple formats.
    Returns (DataFrame, format_info_string) or raises Exception.
    """
    file_ext = file_obj.name.split('.')[-1].lower()
    file_obj.seek(0)
    
    # CSV / TSV
    if file_ext in ['csv', 'tsv']:
        sep = '\t' if file_ext == 'tsv' else ','
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings:
            try:
                file_obj.seek(0)
                # Use low_memory=False and on_bad_lines='warn' to handle large/messy files
                df = pd.read_csv(file_obj, sep=sep, encoding=enc, low_memory=False, on_bad_lines='warn')
                # Check if delimiter is wrong (single column)
                if df.shape[1] < 2 and file_ext == 'csv':
                    file_obj.seek(0)
                    df_semi = pd.read_csv(file_obj, sep=';', encoding=enc, low_memory=False, on_bad_lines='warn')
                    if df_semi.shape[1] > 1:
                        return df_semi, f"CSV (sep=;, enc={enc})"
                return df, f"{file_ext.upper()} (enc={enc})"
            except UnicodeDecodeError:
                continue
        raise ValueError("No se pudo decodificar el archivo CSV/TSV")
    
    # Excel (xlsx, xls)
    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(file_obj, engine='openpyxl')
        return df, f"Excel ({file_ext})"
    
    # ODS (OpenDocument Spreadsheet - LibreOffice/OpenOffice)
    elif file_ext == 'ods':
        df = pd.read_excel(file_obj, engine='odf')
        return df, "ODS (OpenDocument)"
    
    # SPSS (.sav)
    elif file_ext == 'sav':
        import pyreadstat
        import tempfile
        import os
        # pyreadstat needs a file path, so save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        try:
            df, meta = pyreadstat.read_sav(tmp_path)
            return df, "SPSS (.sav)"
        finally:
            os.unlink(tmp_path)
    
    # Stata (.dta)
    elif file_ext == 'dta':
        import pyreadstat
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dta') as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        try:
            df, meta = pyreadstat.read_dta(tmp_path)
            return df, "Stata (.dta)"
        finally:
            os.unlink(tmp_path)
    
    # JSON
    elif file_ext == 'json':
        df = pd.read_json(file_obj)
        return df, "JSON"
    
    else:
        raise ValueError(f"Formato no soportado: {file_ext}")

if dict_file:
    try:
        dict_df, fmt = load_file_to_dataframe(dict_file)
        st.session_state['data_dict'] = dict_df
        st.session_state['dict_filename'] = dict_file.name
        st.sidebar.success(f"✅ Diccionario: {fmt}")
    except Exception as e:
        st.sidebar.error(f"Error al cargar diccionario: {e}")

if st.sidebar.button("Cargar Datos de Ejemplo"):
    st.session_state['data'] = sqa_engine.generate_dummy_data()
    st.session_state['filename'] = "Dummy Data"
    st.success("Datos de ejemplo cargados!")

if uploaded_file:
    try:
        df, fmt = load_file_to_dataframe(uploaded_file)
        st.sidebar.success(f"✅ Dataset: {fmt}")
        
        # --- CLEAN COLUMN NAMES ---
        # Convert all column names to strings, replacing None/NaN with placeholder
        new_cols = []
        for i, col in enumerate(df.columns):
            if col is None or (isinstance(col, float) and pd.isna(col)) or str(col).lower() == 'nan':
                new_cols.append(f"_Columna_{i}")
            elif str(col).startswith('Unnamed:'):
                new_cols.append(f"_Columna_{i}")
            else:
                new_cols.append(str(col))
        df.columns = new_cols
        # --- END CLEAN ---
            
        st.session_state['data'] = df
        st.session_state['filename'] = uploaded_file.name
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")

# --- Main App Logic ---
if 'data' in st.session_state:
    df = st.session_state['data']
    
    # --- Build Dictionary Lookup EARLY for selectors ---
    dict_lookup = build_dict_lookup()
    has_dict = len(dict_lookup) > 0
    
    # Target Variable Selector
    # Prepare columns as simple list of strings, filtering out invalid names
    all_columns_raw = list(df.columns.astype(str))
    # Filter out NaN, Unnamed, and empty column names
    all_columns_list = [c for c in all_columns_raw if c and c.lower() != 'nan' and not c.startswith('Unnamed:')]
    
    # Determined default index safely
    default_ix = 0
    if 'Violencia_Escolar' in all_columns_list:
        default_ix = all_columns_list.index('Violencia_Escolar')

    # Function to display labels in selectbox/multiselect
    def get_display_label(code):
        """Returns descriptive label for selector display"""
        if has_dict:
            return get_column_label(code, dict_lookup, max_len=60)
        return code

    target_var = st.sidebar.selectbox(
        "Seleccione Variable Objetivo",
        options=all_columns_list,
        index=default_ix,
        format_func=get_display_label
    )
    
    # [NEW] Factor Selection to avoid using IDs/Dates as causal factors
    factor_cols = [c for c in all_columns_list if c != target_var]
        
    selected_factors = st.sidebar.multiselect(
        "Seleccione Factores Causales (Variables)",
        options=factor_cols,
        default=factor_cols[:5] if len(factor_cols) >= 5 else factor_cols, # Default to first 5 to avoid clutter
        format_func=get_display_label
    )
    
    # Filter DF to only selected columns + target
    if selected_factors:
        df_analysis = df[selected_factors + [target_var]].copy()
    else:
        df_analysis = df[[target_var]].copy() # Fallback to just target
    
    st.title("Plataforma Digital de Análisis Causal Neutrosófico")
    st.markdown(f"**Archivo analizado:** `{st.session_state.get('filename', 'Unknown')}`")
    
    # Validation Warning
    unique_targets = df[target_var].unique()
    if len(unique_targets) > 20:
        st.warning(f"⚠️ La variable objetivo '{target_var}' tiene {len(unique_targets)} valores únicos. Verifique que no sea un ID o Fecha.")
    
    # Intelligent Text Warning
    st.info("ℹ️ **Mejora Inteligente:** El algoritmo convertirá respuestas de texto (Yes, Always, True) a Verdad (1) y (No, Never, False) a Falsedad (0).")

    # Tabs - UPDATED with new tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Datos", 
        "🔬 Neutrosofía", 
        "📈 Análisis Causal", 
        "🔍 Patrones",
        "🗺️ Mapa",
        "🤖 Machine Learning"
    ])
    
    # 1. Tab Datos
    with tab1:
        st.subheader("Dataset (Columnas Seleccionadas)")
        
        # Display with descriptive column names if dictionary is available
        if has_dict:
            display_df = df_analysis.copy()
            # Create unique column names by appending original code if there are duplicates
            new_col_names = {}
            seen_labels = {}
            for col in display_df.columns:
                label = get_column_label(col, dict_lookup)
                if label in seen_labels:
                    seen_labels[label] += 1
                    new_col_names[col] = f"{label} ({col})"
                else:
                    seen_labels[label] = 1
                    new_col_names[col] = label
            display_df = display_df.rename(columns=new_col_names)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(df_analysis, use_container_width=True)
            st.info("💡 **Tip:** Puedes subir un diccionario de datos en la barra lateral para ver nombres descriptivos en lugar de códigos.")
        
        st.caption(f"Dimensiones: {df_analysis.shape}")

    # 2. Tab Neutrosofía
    with tab2:
        st.subheader("Transformación Neutrosófica (Algoritmo 1)")
        
        # Run Algo 1
        neutro_df = sqa_engine.convert_to_neutrosophic(df_analysis, target_col=target_var)
        
        # Display with descriptive names
        if has_dict:
            neutro_display = neutro_df.copy()
            # Rename columns like Factor_T, Factor_I, Factor_F to DescriptiveName_T, etc.
            new_neutro_cols = {}
            seen_labels = {}
            for col in neutro_display.columns:
                for suffix in ['_T', '_I', '_F']:
                    if col.endswith(suffix):
                        base_name = col[:-2]  # Remove suffix
                        desc_name = get_column_label(base_name, dict_lookup, max_len=40)
                        new_label = f"{desc_name}{suffix}"
                        if new_label in seen_labels:
                            new_label = f"{desc_name} ({base_name}){suffix}"
                        seen_labels[new_label] = 1
                        new_neutro_cols[col] = new_label
                        break
            neutro_display = neutro_display.rename(columns=new_neutro_cols)
            st.dataframe(neutro_display, use_container_width=True)
        else:
            st.dataframe(neutro_df, use_container_width=True)
        
        st.markdown("**Leyenda:** T = Verdad, I = Indeterminación, F = Falsedad")
        
    # 3. Tab Análisis Causal
    with tab3:
        st.subheader("Análisis Comparativo y Relevancia (Algoritmos 2 & 3)")
        
        if not selected_factors:
            st.error("Por favor seleccione al menos un factor en la barra lateral.")
        else:
            # Run Algo 2
            summary_df = sqa_engine.calculate_comparative_summary(neutro_df, df_analysis, target_var)
            
            # Run Algo 3
            final_summary = sqa_engine.classify_relevance(summary_df)
            
            # Get target variable label for legend
            target_label = get_column_label(target_var, dict_lookup) if has_dict else target_var
            
            # Create chart for display and PDF - FIXED SIZING
            chart_df = summary_df.copy()
            if has_dict:
                chart_df['Factor'] = chart_df['Factor'].apply(
                    lambda x: get_column_label(x, dict_lookup, max_len=35)
                )
            
            melted_df = chart_df.melt(id_vars='Factor', value_vars=['Mean_T_Pos', 'Mean_T_Neg'], 
                                       var_name='Grupo', value_name='Promedio Verdad (T)')
            
            melted_df['Grupo'] = melted_df['Grupo'].map({
                'Mean_T_Pos': f'Casos Positivos ({target_label}=1)', 
                'Mean_T_Neg': f'Casos Negativos ({target_label}=0)'
            })
            
            # IMPROVED CHART with better sizing
            fig = px.bar(
                melted_df, 
                x='Factor', 
                y='Promedio Verdad (T)', 
                color='Grupo', 
                barmode='group',
                title="Comparación de Niveles de Verdad por Factor",
                color_discrete_map={
                    f'Casos Positivos ({target_label}=1)': '#FF4B4B',
                    f'Casos Negativos ({target_label}=0)': '#4B4BFF'
                }
            )
            
            # Fix chart sizing and layout
            fig.update_layout(
                height=600,  # Fixed height
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=80, b=150),  # More bottom margin for labels
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12)
                ),
                xaxis=dict(
                    tickfont=dict(size=11),
                    title_font=dict(size=14)
                ),
                yaxis=dict(
                    tickfont=dict(size=12),
                    title_font=dict(size=14),
                    range=[0, 1]  # T values are between 0 and 1
                ),
                title=dict(
                    font=dict(size=18)
                )
            )
            
            # Generate PDF on-the-fly for download button
            patterns_for_pdf = sqa_engine.detect_patterns(neutro_df, df_analysis, target_var)
            
            # Check if ML results exist
            ml_results_for_pdf = st.session_state.get('ml_results', None)
            
            try:
                pdf_bytes = generate_pdf_report(
                    filename=st.session_state.get('filename', 'Datos'),
                    target_var=target_var,
                    target_label=target_label,
                    selected_factors=selected_factors,
                    df_analysis=df_analysis,
                    neutro_df=neutro_df,
                    summary_df=final_summary,
                    final_summary=final_summary,
                    patterns=patterns_for_pdf,
                    fig=fig,
                    dict_lookup=dict_lookup,
                    has_dict=has_dict,
                    ml_results=ml_results_for_pdf
                )
                
                # Download button at the top
                st.download_button(
                    label="📄 Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name=f"Reporte_SQA_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            except Exception as e:
                st.warning(f"PDF no disponible: {e}")
            
            st.markdown("---")
            
            # Apply dictionary labels to Factor column for display
            if has_dict:
                final_summary_display = final_summary.copy()
                final_summary_display['Factor'] = final_summary_display['Factor'].apply(
                    lambda x: get_column_label(x, dict_lookup, max_len=60)
                )
            else:
                final_summary_display = final_summary
            
            # Display Table
            st.dataframe(final_summary_display.style.format({
                'Mean_T_Pos': '{:.3f}',
                'Mean_T_Neg': '{:.3f}',
                'Delta_T': '{:.3f}'
            }).map(
                lambda v: 'color: green; font-weight: bold' if v == 'Alta Relevancia' else 
                          ('color: orange' if v == 'Media Relevancia' else 'color: gray'),
                subset=['Relevancia']
            ), use_container_width=True)
            
            # Visualization - ALWAYS show with proper sizing
            st.subheader("Gráfico Comparativo de Verdad (T)")
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
            
            # Store for reference
            st.session_state['last_fig'] = fig
            st.session_state['last_summary'] = final_summary
            st.session_state['last_target_label'] = target_label

    # 4. Tab Patrones
    with tab4:
        st.subheader("Detección de Patrones Frecuentes (Algoritmo 4)")
        if not selected_factors:
             st.error("Seleccione factores.")
        else:
            patterns = sqa_engine.detect_patterns(neutro_df, df_analysis, target_var)
            
            if not patterns.empty:
                # Apply dictionary labels to pattern configurations
                if has_dict:
                    patterns_display = patterns.copy()
                    
                    def translate_pattern(pattern_str):
                        """Translate factor codes to descriptive names in pattern string"""
                        if pattern_str == "Ninguno":
                            return pattern_str
                        factors = pattern_str.split(" & ")
                        translated = [get_column_label(f.strip(), dict_lookup, max_len=40) for f in factors]
                        return " & ".join(translated)
                    
                    patterns_display['Configuracion'] = patterns_display['Configuracion'].apply(translate_pattern)
                    st.table(patterns_display)
                else:
                    st.table(patterns)
            else:
                st.info("No se detectaron patrones con el umbral especificado.")
            
            # Store patterns for PDF
            st.session_state['last_patterns'] = patterns

    # 5. Tab Mapa (NEW)
    with tab5:
        st.subheader("🗺️ Mapa de Índices por Ubicación Geográfica")
        
        # Detect geographic columns
        geo_columns = map_engine.detect_geographic_columns(df)
        
        if geo_columns:
            st.success(f"✅ Se detectaron {len(geo_columns)} posibles columnas geográficas")
            
            # Let user select geographic column
            geo_options = [col[0] for col in geo_columns]
            selected_geo_col = st.selectbox(
                "Seleccione columna geográfica:",
                options=geo_options,
                format_func=get_display_label
            )
            
            # Aggregation method
            agg_method = st.selectbox(
                "Método de agregación:",
                options=['proportion', 'mean', 'count', 'sum'],
                format_func=lambda x: {
                    'proportion': 'Proporción de casos positivos',
                    'mean': 'Promedio del valor',
                    'count': 'Conteo de registros',
                    'sum': 'Suma de valores'
                }.get(x, x)
            )
            
            # Generate aggregated data
            target_label_map = get_column_label(target_var, dict_lookup) if has_dict else target_var
            agg_df = map_engine.aggregate_by_location(df, selected_geo_col, target_var, agg_func=agg_method)
            
            if not agg_df.empty:
                # Show summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ubicaciones", len(agg_df))
                with col2:
                    st.metric("Valor Máximo", f"{agg_df['Value'].max():.3f}")
                with col3:
                    st.metric("Valor Mínimo", f"{agg_df['Value'].min():.3f}")
                with col4:
                    st.metric("Promedio", f"{agg_df['Value'].mean():.3f}")
                
                # Generate map visualization
                st.markdown("---")
                
                # Try to show Folium map first
                try:
                    import folium
                    from streamlit_folium import st_folium
                    
                    map_html = map_engine.generate_map_html(agg_df, target_label=target_label_map)
                    if map_html:
                        st.subheader("Mapa Interactivo de Ecuador")
                        st.components.v1.html(map_html, height=600, scrolling=True)
                except ImportError:
                    pass  # Folium not available, will use plotly
                
                # Always show Plotly bar chart as alternative
                st.subheader("Gráfico de Barras por Ubicación")
                fig_map = map_engine.generate_plotly_map(agg_df, target_label=target_label_map)
                st.plotly_chart(fig_map, use_container_width=True, key="map_chart")
                
                # Show data table
                with st.expander("📋 Ver datos agregados"):
                    st.dataframe(agg_df, use_container_width=True)
            else:
                st.warning("No se pudieron agregar datos por ubicación. Verifique la columna seleccionada.")
        else:
            st.info("""
            ℹ️ **No se detectaron columnas geográficas automáticamente.**
            
            Para usar esta funcionalidad, su dataset debe contener una columna con:
            - Nombres de provincias de Ecuador (Guayas, Pichincha, Azuay, etc.)
            - Nombres de cantones (Guayaquil, Quito, Cuenca, etc.)
            - O una columna con nombre como: 'provincia', 'canton', 'ciudad', 'zona', etc.
            """)

    # 6. Tab Machine Learning (NEW)
    with tab6:
        st.subheader("🤖 Flujo de Machine Learning")
        
        # Display the ML workflow image reference
        st.markdown("""
        Este módulo implementa el flujo completo de Machine Learning:
        - **Preprocesamiento**: Limpieza, codificación, escalado
        - **División de Datos**: 80% Entrenamiento / 20% Prueba
        - **Múltiples Algoritmos**: SVM, Random Forest, Decision Tree, KNN, Gradient Boosting
        - **Validación Cruzada**: K-Fold Cross Validation
        - **Métricas**: Accuracy, Sensibilidad, Especificidad, F1, MCC (Clasificación) | MSE, RMSE, R² (Regresión)
        """)
        
        st.markdown("---")
        
        # ML Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
        with col2:
            cv_folds = st.slider("Folds para Cross-Validation", 3, 10, 5)
        with col3:
            use_pca = st.checkbox("Usar PCA", value=False)
        
        # Run ML Pipeline
        if st.button("🚀 Ejecutar Pipeline de ML", type="primary"):
            with st.spinner("Ejecutando pipeline de Machine Learning..."):
                ml_results = ml_engine.run_full_ml_pipeline(
                    df=df_analysis,
                    target_col=target_var,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    use_pca=use_pca
                )
                
                st.session_state['ml_results'] = ml_results
        
        # Display results if available
        if 'ml_results' in st.session_state:
            ml_results = st.session_state['ml_results']
            
            if ml_results.get('success'):
                st.success("✅ Pipeline ejecutado exitosamente")
                
                # Task type
                task_type = ml_results.get('task_type', 'classification')
                st.info(f"📋 **Tipo de tarea detectado:** {task_type.title()}")
                
                # Split info
                st.subheader("📊 División de Datos")
                split_info = ml_results.get('split_info', {})
                
                # Custom HTML boxes for better visibility in dark mode
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="split-box total-box">
                        <h3>📊 TOTAL DE MUESTRAS</h3>
                        <h1>{split_info.get('total_samples', 0)}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="split-box train-box">
                        <h3>🎯 ENTRENAMIENTO</h3>
                        <h1>{split_info.get('train_samples', 0)} ({split_info.get('train_percentage', 80)}%)</h1>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="split-box test-box">
                        <h3>🧪 PRUEBA</h3>
                        <h1>{split_info.get('test_samples', 0)} ({split_info.get('test_percentage', 20)}%)</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Pie chart for split visualization
                fig_split = px.pie(
                    values=[split_info.get('train_samples', 80), split_info.get('test_samples', 20)],
                    names=['Entrenamiento (80%)', 'Prueba (20%)'],
                    title='Distribución del Dataset',
                    color_discrete_sequence=['#4CAF50', '#2196F3'],
                    hole=0.4  # Donut chart for better visual
                )
                fig_split.update_layout(
                    height=350,
                    font=dict(size=14),
                    title=dict(font=dict(size=18))
                )
                fig_split.update_traces(
                    textposition='outside',
                    textinfo='label+percent',
                    textfont_size=14
                )
                st.plotly_chart(fig_split, use_container_width=True, key="split_chart")
                
                # PCA Info if used
                if ml_results.get('pca_info'):
                    st.subheader("🔄 Análisis de Componentes Principales (PCA)")
                    pca_info = ml_results['pca_info']
                    st.write(f"Componentes seleccionados: {pca_info['n_components']} de {pca_info['original_features']}")
                    st.write(f"Varianza total explicada: {pca_info['total_variance_explained']*100:.2f}%")
                    
                    # Variance explained chart
                    fig_pca = px.bar(
                        x=[f"PC{i+1}" for i in range(len(pca_info['explained_variance_ratio']))],
                        y=pca_info['explained_variance_ratio'],
                        title='Varianza Explicada por Componente',
                        labels={'x': 'Componente', 'y': 'Varianza Explicada'}
                    )
                    fig_pca.update_layout(height=300)
                    st.plotly_chart(fig_pca, use_container_width=True, key="pca_chart")
                
                # Model Results
                st.subheader("🏆 Resultados de Modelos")
                model_results = ml_results.get('model_results', [])
                
                if model_results:
                    results_df = pd.DataFrame(model_results)
                    
                    # Highlight best model
                    best_model = ml_results.get('best_model_name', '')
                    
                    def highlight_best(row):
                        if row['Modelo'] == best_model:
                            return ['background-color: #90EE90'] * len(row)
                        return [''] * len(row)
                    
                    styled_results = results_df.style.apply(highlight_best, axis=1)
                    
                    if task_type == 'classification':
                        styled_results = styled_results.format({
                            'Accuracy': '{:.4f}',
                            'Sensibilidad': '{:.4f}',
                            'Especificidad': '{:.4f}',
                            'Precision': '{:.4f}',
                            'F1-Score': '{:.4f}',
                            'MCC': '{:.4f}',
                            'CV Mean': '{:.4f}',
                            'CV Std': '{:.4f}'
                        })
                    else:
                        styled_results = styled_results.format({
                            'MSE': '{:.4f}',
                            'RMSE': '{:.4f}',
                            'MAE': '{:.4f}',
                            'R²': '{:.4f}',
                            'CV Mean': '{:.4f}',
                            'CV Std': '{:.4f}'
                        })
                    
                    st.dataframe(styled_results, use_container_width=True)
                    st.caption(f"🥇 Mejor modelo: **{best_model}**")
                    
                    # Model comparison chart
                    if task_type == 'classification':
                        metric_col = 'Accuracy'
                    else:
                        metric_col = 'R²'
                    
                    fig_models = px.bar(
                        results_df.sort_values(metric_col, ascending=True),
                        y='Modelo',
                        x=metric_col,
                        orientation='h',
                        title=f'Comparación de Modelos por {metric_col}',
                        color=metric_col,
                        color_continuous_scale='RdYlGn'
                    )
                    fig_models.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_models, use_container_width=True, key="models_chart")
                
                # Confusion Matrix (for classification)
                if task_type == 'classification' and 'confusion_matrix' in ml_results:
                    try:
                        st.subheader("📊 Matriz de Confusión")
                        cm_data = ml_results['confusion_matrix']
                        cm_matrix = np.array(cm_data['matrix'])
                        cm_labels = cm_data['labels']
                        
                        fig_cm = px.imshow(
                            cm_matrix,
                            labels=dict(x="Predicho", y="Real", color="Conteo"),
                            x=cm_labels,
                            y=cm_labels,
                            title=f'Matriz de Confusión - {best_model}',
                            color_continuous_scale='Blues',
                            text_auto=True
                        )
                        fig_cm.update_traces(hovertemplate="Valor Real: %{y}<br>Valor Predicho: %{x}<br>Conteo: %{z}<extra></extra>")
                        fig_cm.update_layout(
                            height=400,
                            xaxis=dict(type='category', title='Valores de Predicción'),
                            yaxis=dict(type='category', title='Valores Reales', autorange='reversed')
                        )
                        st.plotly_chart(fig_cm, use_container_width=True, key="cm_chart")
                    except Exception as e:
                        st.error(f"Error generando matriz de confusión: {e}")
                
                # Cross-validation visualization
                st.subheader("🔄 Validación Cruzada")
                st.markdown(f"""
                Se utilizó **{cv_folds}-fold cross validation** para evaluar la estabilidad de los modelos.
                Los valores de CV Mean y CV Std muestran el rendimiento promedio y la desviación estándar
                a través de los {cv_folds} folds.
                """)
                
                # PDF Export for ML Results
                st.subheader("📥 Exportar Resultados")
                
                # Recreate results_df from ml_results for PDF export
                model_results_pdf = ml_results.get('model_results', [])
                if model_results_pdf:
                    results_df = pd.DataFrame(model_results_pdf)
                else:
                    results_df = pd.DataFrame()
                
                # Generate ML-specific PDF
                from fpdf import FPDF
                
                pdf_ml = FPDF(orientation='L')
                pdf_ml.add_page()
                pdf_ml.set_auto_page_break(auto=True, margin=15)
                
                # Title
                pdf_ml.set_font('Arial', 'B', 18)
                pdf_ml.cell(0, 15, 'Reporte de Machine Learning - Plataforma SQA', ln=True, align='C')
                pdf_ml.ln(10)
                
                # Dataset info
                pdf_ml.set_font('Arial', 'B', 14)
                pdf_ml.cell(0, 10, 'Informacion del Dataset', ln=True)
                pdf_ml.set_font('Arial', '', 11)
                pdf_ml.cell(0, 7, f"Variable Objetivo: {target_var}", ln=True)
                pdf_ml.cell(0, 7, f"Tipo de Tarea: {task_type.title()}", ln=True)
                pdf_ml.cell(0, 7, f"Total de Muestras: {split_info.get('total_samples', 0)}", ln=True)
                pdf_ml.cell(0, 7, f"Entrenamiento: {split_info.get('train_samples', 0)} ({split_info.get('train_percentage', 80)}%)", ln=True)
                pdf_ml.cell(0, 7, f"Prueba: {split_info.get('test_samples', 0)} ({split_info.get('test_percentage', 20)}%)", ln=True)
                pdf_ml.cell(0, 7, f"Validacion Cruzada: {cv_folds}-fold", ln=True)
                pdf_ml.ln(10)
                
                # Model results
                pdf_ml.set_font('Arial', 'B', 14)
                pdf_ml.cell(0, 10, 'Resultados de los Modelos', ln=True)
                pdf_ml.set_font('Arial', '', 10)
                
                # Table header
                if task_type == 'classification':
                    headers = ['Modelo', 'Accuracy', 'Sensibilidad', 'Especificidad', 'F1-Score', 'MCC', 'CV Mean']
                    col_widths = [50, 30, 35, 35, 30, 25, 30]
                else:
                    headers = ['Modelo', 'MSE', 'RMSE', 'MAE', 'R2', 'CV Mean']
                    col_widths = [50, 35, 35, 35, 30, 35]
                
                pdf_ml.set_font('Arial', 'B', 9)
                for i, header in enumerate(headers):
                    pdf_ml.cell(col_widths[i], 8, header, border=1, align='C')
                pdf_ml.ln()
                
                pdf_ml.set_font('Arial', '', 9)
                for _, row in results_df.iterrows():
                    if task_type == 'classification':
                        values = [
                            row['Modelo'][:15],
                            f"{row['Accuracy']:.4f}",
                            f"{row['Sensibilidad']:.4f}",
                            f"{row['Especificidad']:.4f}",
                            f"{row['F1-Score']:.4f}",
                            f"{row['MCC']:.4f}",
                            f"{row['CV Mean']:.4f}"
                        ]
                    else:
                        values = [
                            row['Modelo'][:15],
                            f"{row['MSE']:.4f}",
                            f"{row['RMSE']:.4f}",
                            f"{row['MAE']:.4f}",
                            f"{row['R²']:.4f}",
                            f"{row['CV Mean']:.4f}"
                        ]
                    for i, val in enumerate(values):
                        pdf_ml.cell(col_widths[i], 7, str(val), border=1, align='C')
                    pdf_ml.ln()
                
                pdf_ml.ln(10)
                
                # Best model highlight
                pdf_ml.set_font('Arial', 'B', 12)
                best_model_name_export = ml_results.get('best_model_name', 'N/A')
                pdf_ml.cell(0, 10, f'Mejor Modelo: {best_model_name_export}', ln=True)
                pdf_ml.ln(5)
                
                # PCA info if used
                if ml_results.get('pca_info'):
                    pca_info = ml_results['pca_info']
                    pdf_ml.set_font('Arial', 'B', 14)
                    pdf_ml.cell(0, 10, 'Analisis de Componentes Principales (PCA)', ln=True)
                    pdf_ml.set_font('Arial', '', 11)
                    pdf_ml.cell(0, 7, f"Componentes: {pca_info['n_components']} de {pca_info['original_features']}", ln=True)
                    pdf_ml.cell(0, 7, f"Varianza Explicada: {pca_info['total_variance_explained']*100:.2f}%", ln=True)
                
                # Methodology section
                pdf_ml.add_page()
                pdf_ml.set_font('Arial', 'B', 14)
                pdf_ml.cell(0, 10, 'Metodologia Aplicada', ln=True)
                pdf_ml.set_font('Arial', '', 11)
                pdf_ml.multi_cell(0, 7, """
El flujo de Machine Learning implementado sigue las mejores practicas:

1. Preprocesamiento: Limpieza de datos, codificacion de variables categoricas, escalado de features.
2. Division de Datos: Separacion en conjuntos de entrenamiento y prueba.
3. Entrenamiento: Aplicacion de 5 algoritmos (SVM, Random Forest, Decision Tree, KNN, Gradient Boosting).
4. Validacion Cruzada: K-Fold Cross Validation para evaluar estabilidad.
5. Evaluacion: Calculo de metricas segun el tipo de tarea.

Metricas de Clasificacion: Accuracy, Sensibilidad, Especificidad, F1-Score, MCC
Metricas de Regresion: MSE, RMSE, MAE, R2
                """)
                
                # Footer
                pdf_ml.ln(20)
                pdf_ml.set_font('Arial', 'I', 10)
                import datetime
                pdf_ml.cell(0, 10, f'Generado: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C')
                
                # Get PDF bytes
                pdf_ml_bytes = bytes(pdf_ml.output())
                
                st.download_button(
                    label="📥 Descargar Reporte ML (PDF)",
                    data=pdf_ml_bytes,
                    file_name="reporte_machine_learning.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                
            else:
                st.error(f"❌ Error en el pipeline: {ml_results.get('error', 'Error desconocido')}")
                if ml_results.get('traceback'):
                    with st.expander("Ver detalles del error"):
                        st.code(ml_results['traceback'])

else:
    st.info("👈 Por favor, sube un archivo CSV o carga los datos de ejemplo desde la barra lateral.")
