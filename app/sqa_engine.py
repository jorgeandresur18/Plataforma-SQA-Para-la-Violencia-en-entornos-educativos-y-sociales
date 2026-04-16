import pandas as pd
import numpy as np

def generate_dummy_data(n_rows=500):
    """
    Genera datos dummy realistas para pruebas de la aplicación.
    Los datos tienen correlaciones reales entre las variables y la variable objetivo,
    para que los modelos de ML produzcan resultados coherentes con una diagonal
    fuerte en la matriz de confusión.
    """
    np.random.seed(42)
    
    # --- Variables independientes (factores causales) ---
    padres_divorciados = np.random.choice([0, 1], size=n_rows, p=[0.65, 0.35])
    acoso_redes = np.random.choice([0, 1], size=n_rows, p=[0.6, 0.4])
    pandillas_barrio = np.random.choice([0, 1], size=n_rows, p=[0.75, 0.25])
    bajo_rendimiento = np.random.choice([0, 1], size=n_rows, p=[0.55, 0.45])
    consumo_alcohol = np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])
    problemas_conducta = np.random.choice([0, 1], size=n_rows, p=[0.6, 0.4])
    falta_supervision = np.random.choice([0, 1], size=n_rows, p=[0.5, 0.5])
    baja_autoestima = np.random.choice([0, 1], size=n_rows, p=[0.55, 0.45])
    entorno_violento = np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])
    uso_drogas = np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2])
    ausentismo = np.random.choice([0, 1], size=n_rows, p=[0.6, 0.4])
    conflictos_familiares = np.random.choice([0, 1], size=n_rows, p=[0.55, 0.45])
    
    # --- Variable objetivo con correlaciones reales ---
    # Puntaje basado en factores con pesos realistas
    score = (
        padres_divorciados * 0.15 +
        acoso_redes * 0.20 +
        pandillas_barrio * 0.25 +
        bajo_rendimiento * 0.10 +
        consumo_alcohol * 0.18 +
        problemas_conducta * 0.22 +
        falta_supervision * 0.12 +
        baja_autoestima * 0.14 +
        entorno_violento * 0.28 +
        uso_drogas * 0.20 +
        ausentismo * 0.08 +
        conflictos_familiares * 0.16 +
        np.random.normal(0, 0.15, n_rows)  # Ruido aleatorio
    )
    
    # Clasificar en 3 niveles: 0=Sin violencia, 1=Violencia moderada, 2=Violencia alta
    violencia = np.zeros(n_rows, dtype=int)
    violencia[score > 0.45] = 1
    violencia[score > 0.85] = 2
    
    data = {
        'Padres_Divorciados': padres_divorciados,
        'Acoso_Redes': acoso_redes,
        'Pandillas_Barrio': pandillas_barrio,
        'Bajo_Rendimiento': bajo_rendimiento,
        'Consumo_Alcohol': consumo_alcohol,
        'Problemas_Conducta': problemas_conducta,
        'Falta_Supervision': falta_supervision,
        'Baja_Autoestima': baja_autoestima,
        'Entorno_Violento': entorno_violento,
        'Uso_Drogas': uso_drogas,
        'Ausentismo': ausentismo,
        'Conflictos_Familiares': conflictos_familiares,
        'Violencia_Escolar': violencia
    }
    return pd.DataFrame(data)


def convert_to_neutrosophic(df, target_col=None):
    """
    Algoritmo 1: Conversión a Lógica Neutrosófica
    Input: Dataframe con valores.
    
    Maneja:
    - Variables binarias (0/1, Sí/No)
    - Variables numéricas continuas (Edad, Ingresos, etc.) -> Normalización min-max
    - Variables categóricas (texto)
    
    Mapeo:
    Binario Positivo (1, Yes) -> (T=0.9, I=0.05, F=0.05)
    Binario Negativo (0, No) -> (T=0.05, I=0.05, F=0.9)
    Numérico Continuo -> Normalización: T = (val - min) / (max - min) * 0.85 + 0.05
    Indeterminado -> (T=0.3, I=0.4, F=0.3)
    """
    neutro_df = pd.DataFrame()
    
    # Identify columns to process
    feature_cols = [c for c in df.columns if c != target_col]
    
    for col in feature_cols:
        col_data = df[col]
        
        # Detectar el tipo de variable
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        unique_values = col_data.nunique()
        
        if is_numeric:
            # Verificar si es binaria (solo 0 y 1)
            unique_vals = set(col_data.dropna().unique())
            is_binary = unique_vals.issubset({0, 1, 0.0, 1.0})
            
            if is_binary:
                # Variable binaria numérica
                neutro_df[f'{col}_T'] = col_data.apply(lambda x: 0.9 if x == 1 else 0.05 if x == 0 else 0.3)
                neutro_df[f'{col}_I'] = 0.05
                neutro_df[f'{col}_F'] = col_data.apply(lambda x: 0.05 if x == 1 else 0.9 if x == 0 else 0.3)
            else:
                # Variable numérica continua -> NORMALIZAR
                min_val = col_data.min()
                max_val = col_data.max()
                
                if max_val > min_val:
                    # Normalizar a rango [0.05, 0.9]
                    normalized = (col_data - min_val) / (max_val - min_val)
                    neutro_df[f'{col}_T'] = normalized * 0.85 + 0.05  # Rango [0.05, 0.9]
                else:
                    # Todos los valores son iguales
                    neutro_df[f'{col}_T'] = 0.5
                
                # Indeterminación baja para datos numéricos (son precisos)
                neutro_df[f'{col}_I'] = 0.1
                # Falsedad es complementaria
                neutro_df[f'{col}_F'] = 1 - neutro_df[f'{col}_T'] - neutro_df[f'{col}_I']
                neutro_df[f'{col}_F'] = neutro_df[f'{col}_F'].clip(0.05, 0.9)
        else:
            # Variable categórica (texto)
            s_lower = col_data.astype(str).str.lower().str.strip()
            unique_categories = s_lower.unique()
            n_categories = len(unique_categories)
            
            # Check for known positive/negative words first
            positive_words = ['yes', 'si', 'sí', 'true', 'always', 'siempre', 
                            'most of the time', 'often', 'many times', '1', 'alto', 'high']
            negative_words = ['no', 'never', 'nunca', 'false', 'falso', 'none', '0', 'bajo', 'low']
            partial_words = ['sometimes', 'rarely', 'a veces', 'raramente', 'poco', 'medio', 'medium']
            
            # Check if any known words are present
            has_known_words = any(w in unique_categories for w in positive_words + negative_words + partial_words)
            
            if has_known_words or n_categories <= 2:
                # Binary-like categorical: Use known word mapping
                high_t_cond = s_lower.isin(positive_words)
                high_f_cond = s_lower.isin(negative_words)
                partial_cond = s_lower.isin(partial_words)
                
                conditions = [high_t_cond, high_f_cond, partial_cond]
                
                neutro_df[f'{col}_T'] = np.select(conditions, [0.9, 0.05, 0.45], default=0.3)
                neutro_df[f'{col}_I'] = np.select(conditions, [0.05, 0.05, 0.6], default=0.4)
                neutro_df[f'{col}_F'] = np.select(conditions, [0.05, 0.9, 0.45], default=0.3)
            else:
                # MULTIPLE CATEGORIES -> USE ONE-HOT ENCODING
                # Create a separate column for each category
                # This allows proper analysis of each category's relationship to the outcome
                
                for category in unique_categories:
                    # Skip empty/nan categories
                    if pd.isna(category) or category == '' or category == 'nan':
                        continue
                    
                    # Create readable column name
                    # Capitalize first letter and limit length
                    cat_name = str(category).strip()
                    if len(cat_name) > 20:
                        cat_name = cat_name[:17] + '...'
                    cat_name = cat_name.title().replace(' ', '_')
                    
                    # Column name: Original_Category
                    col_name = f'{col}={cat_name}'
                    
                    # Binary encoding: 1 if matches this category, 0 otherwise
                    is_this_category = (s_lower == category)
                    
                    # Convert to neutrosophic values
                    # If belongs to this category: T=0.9 (high truth)
                    # If not: T=0.05 (low truth)
                    neutro_df[f'{col_name}_T'] = is_this_category.apply(lambda x: 0.9 if x else 0.05)
                    neutro_df[f'{col_name}_I'] = 0.05  # Low indeterminacy for clear categories
                    neutro_df[f'{col_name}_F'] = is_this_category.apply(lambda x: 0.05 if x else 0.9)
    
    return neutro_df

def calculate_comparative_summary(neutro_df, original_df, target_col):
    """
    Algoritmo 2: Resumen Comparativo (Componente T)
    Input: Dataframe neutrosófico y una variable objetivo seleccionada.
    Lógica:
    Dividir el dataset en dos grupos: Casos Positivos y Casos Negativos.
    Detecta automáticamente qué valores representan positivo/negativo.
    Calcular el promedio de la columna T (Mean_T) para cada condición en ambos grupos.
    """
    if target_col not in original_df.columns:
        raise ValueError(f"Target column {target_col} not found in original dataframe")

    # Combine neutro T columns with target for grouping
    t_cols = [c for c in neutro_df.columns if c.endswith('_T')]
    
    if not t_cols:
        raise ValueError("No T columns found in neutrosophic dataframe")
    
    # Create a working df with T cols and target
    work_df = neutro_df[t_cols].copy()
    target_values = original_df[target_col].values
    work_df['__Target__'] = target_values
    
    # Get unique values of target
    unique_targets = pd.Series(target_values).dropna().unique()
    
    # Determine which value is "positive" and which is "negative"
    # Priority: 1/0, True/False, Yes/No, Si/No, or highest value = positive
    positive_indicators = [1, 1.0, True, 'yes', 'si', 'sí', '1', 'true']
    negative_indicators = [0, 0.0, False, 'no', '0', 'false', 'nunca', 'never']
    
    positive_val = None
    negative_val = None
    
    # Try to find known positive/negative values
    for val in unique_targets:
        val_lower = str(val).lower().strip() if not isinstance(val, (int, float, bool)) else val
        
        if val in positive_indicators or val_lower in [str(x).lower() for x in positive_indicators]:
            positive_val = val
        elif val in negative_indicators or val_lower in [str(x).lower() for x in negative_indicators]:
            negative_val = val
    
    # If not found, use the two most common values
    if positive_val is None or negative_val is None:
        value_counts = pd.Series(target_values).value_counts()
        if len(value_counts) >= 2:
            # Assume larger numeric value or last alphabetically = positive
            sorted_vals = sorted(unique_targets, key=lambda x: (isinstance(x, str), str(x)))
            negative_val = sorted_vals[0] if negative_val is None else negative_val
            positive_val = sorted_vals[-1] if positive_val is None else positive_val
        elif len(value_counts) == 1:
            # Only one value - can't compare
            positive_val = unique_targets[0]
            negative_val = None
    
    # Group by Target and calculate mean
    grouped = work_df.groupby('__Target__').mean()
    
    # Extract means for positive and negative cases
    try:
        if positive_val is not None and positive_val in grouped.index:
            mean_y1 = grouped.loc[positive_val][t_cols]
        else:
            mean_y1 = pd.Series([0.0] * len(t_cols), index=t_cols)
    except (KeyError, IndexError):
        mean_y1 = pd.Series([0.0] * len(t_cols), index=t_cols)
        
    try:
        if negative_val is not None and negative_val in grouped.index:
            mean_y0 = grouped.loc[negative_val][t_cols]
        else:
            mean_y0 = pd.Series([0.0] * len(t_cols), index=t_cols)
    except (KeyError, IndexError):
        mean_y0 = pd.Series([0.0] * len(t_cols), index=t_cols)
    
    summary_df = pd.DataFrame({
        'Factor': [col.replace('_T', '') for col in t_cols],
        'Mean_T_Pos': mean_y1.values,
        'Mean_T_Neg': mean_y0.values
    })
    
    # Store the detected values for display
    summary_df.attrs['positive_value'] = positive_val
    summary_df.attrs['negative_value'] = negative_val
    
    return summary_df

def classify_relevance(summary_df):
    """
    Algoritmo 3: Clasificación de Relevancia
    Lógica: Calcular la diferencia Delta T = Mean_T(Y=1) - Mean_T(Y=0).
    Clasificación:
    Si Delta T >= 0.15 -> "Alta Relevancia"
    Si 0.05 <= Delta T < 0.15 -> "Media Relevancia"
    Si Delta T < 0.05 -> "Baja Relevancia"
    """
    df = summary_df.copy()
    df['Delta_T'] = df['Mean_T_Pos'] - df['Mean_T_Neg']
    
    conditions = [
        df['Delta_T'] >= 0.15,
        (df['Delta_T'] >= 0.05) & (df['Delta_T'] < 0.15)
    ]
    choices = ['Alta Relevancia', 'Media Relevancia']
    
    df['Relevancia'] = np.select(conditions, choices, default='Baja Relevancia')
    
    return df

def detect_patterns(neutro_df, original_df, target_col, threshold=0.5):
    """
    Algoritmo 4: Detección de Patrones (Configuraciones)
    Input: Solo filas donde la variable objetivo es positiva.
    Lógica:
    Filtrar factores donde el valor T > umbral.
    Crear una cadena de texto combinando los nombres de estos factores.
    Contar la frecuencia de estas combinaciones.
    Retornar el Top 5 de configuraciones más frecuentes.
    """
    # Determine which values are "positive"
    target_values = original_df[target_col].values
    unique_targets = pd.Series(target_values).dropna().unique()
    
    positive_indicators = [1, 1.0, True, 'yes', 'si', 'sí', '1', 'true']
    positive_val = None
    
    # Try to find positive value
    for val in unique_targets:
        val_check = val
        if isinstance(val, str):
            val_check = val.lower().strip()
        
        if val in positive_indicators or val_check in [str(x).lower() for x in positive_indicators]:
            positive_val = val
            break
    
    # If not found, use the last value when sorted
    if positive_val is None and len(unique_targets) >= 2:
        sorted_vals = sorted(unique_targets, key=lambda x: (isinstance(x, str), str(x)))
        positive_val = sorted_vals[-1]
    elif positive_val is None and len(unique_targets) >= 1:
        positive_val = unique_targets[0]
    
    # Filter rows where Target is positive
    target_mask = original_df[target_col] == positive_val
    subset_neutro = neutro_df[target_mask].copy()
    
    if subset_neutro.empty:
        return pd.DataFrame(columns=['Configuracion', 'Frecuencia', 'Porcentaje'])
    
    # Get only T columns
    t_cols = [c for c in subset_neutro.columns if c.endswith('_T')]
    
    if not t_cols:
        return pd.DataFrame(columns=['Configuracion', 'Frecuencia', 'Porcentaje'])
    
    configurations = []
    
    for idx, row in subset_neutro.iterrows():
        # Find factors with T > threshold
        high_t_factors = []
        for col in t_cols:
            if row[col] > threshold:
                # Clean name: remove _T and format nicely
                factor_name = col.replace('_T', '')
                # Limit length for readability
                if len(factor_name) > 30:
                    factor_name = factor_name[:27] + '...'
                high_t_factors.append(factor_name)
        
        if high_t_factors:
            # Limit to top 5 factors per configuration to avoid huge strings
            if len(high_t_factors) > 5:
                high_t_factors = sorted(high_t_factors)[:5]
            conf_str = " & ".join(sorted(high_t_factors))
            configurations.append(conf_str)
        else:
            configurations.append("Sin factores destacados")
            
    # Count frequencies
    if not configurations:
        return pd.DataFrame(columns=['Configuracion', 'Frecuencia', 'Porcentaje'])
        
    config_series = pd.Series(configurations)
    counts = config_series.value_counts().reset_index()
    counts.columns = ['Configuracion', 'Frecuencia']
    
    # Add percentage
    total = len(configurations)
    counts['Porcentaje'] = (counts['Frecuencia'] / total * 100).round(1)
    
    return counts.head(10)
