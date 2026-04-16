"""
map_engine.py - Motor de Mapas para Plataforma SQA

Genera mapas interactivos de Ecuador mostrando índices de violencia por zona geográfica.
Funciona dinámicamente detectando columnas con nombres de provincias/cantones.
"""

import pandas as pd
import numpy as np
import json
import requests
from io import StringIO

# Mapeo de nombres de provincias de Ecuador (variaciones comunes)
ECUADOR_PROVINCES = {
    'azuay': 'Azuay',
    'bolivar': 'Bolívar',
    'cañar': 'Cañar',
    'carchi': 'Carchi',
    'chimborazo': 'Chimborazo',
    'cotopaxi': 'Cotopaxi',
    'el oro': 'El Oro',
    'esmeraldas': 'Esmeraldas',
    'galapagos': 'Galápagos',
    'guayas': 'Guayas',
    'imbabura': 'Imbabura',
    'loja': 'Loja',
    'los rios': 'Los Ríos',
    'manabi': 'Manabí',
    'morona santiago': 'Morona Santiago',
    'napo': 'Napo',
    'orellana': 'Orellana',
    'pastaza': 'Pastaza',
    'pichincha': 'Pichincha',
    'santa elena': 'Santa Elena',
    'santo domingo': 'Santo Domingo de los Tsáchilas',
    'santo domingo de los tsachilas': 'Santo Domingo de los Tsáchilas',
    'sucumbios': 'Sucumbíos',
    'tungurahua': 'Tungurahua',
    'zamora chinchipe': 'Zamora Chinchipe',
    # Abreviaturas comunes
    'gye': 'Guayas',
    'uio': 'Pichincha',
    'cuenca': 'Azuay',
    'guayaquil': 'Guayas',
    'quito': 'Pichincha',
    'duran': 'Guayas',
    'durán': 'Guayas',
    'manta': 'Manabí',
    'machala': 'El Oro',
    'ambato': 'Tungurahua',
    'riobamba': 'Chimborazo',
    'loja ciudad': 'Loja',
    'ibarra': 'Imbabura',
    'latacunga': 'Cotopaxi',
    'portoviejo': 'Manabí',
    'babahoyo': 'Los Ríos',
    'tulcan': 'Carchi',
    'tulcán': 'Carchi'
}

# Cantones principales por provincia
ECUADOR_CANTONES = {
    'guayaquil': ('Guayaquil', 'Guayas'),
    'duran': ('Durán', 'Guayas'),
    'durán': ('Durán', 'Guayas'),
    'samborondon': ('Samborondón', 'Guayas'),
    'daule': ('Daule', 'Guayas'),
    'milagro': ('Milagro', 'Guayas'),
    'quito': ('Quito', 'Pichincha'),
    'cayambe': ('Cayambe', 'Pichincha'),
    'rumiñahui': ('Rumiñahui', 'Pichincha'),
    'cuenca': ('Cuenca', 'Azuay'),
    'machala': ('Machala', 'El Oro'),
    'manta': ('Manta', 'Manabí'),
    'portoviejo': ('Portoviejo', 'Manabí'),
    'ambato': ('Ambato', 'Tungurahua'),
    'riobamba': ('Riobamba', 'Chimborazo'),
    'loja': ('Loja', 'Loja'),
    'ibarra': ('Ibarra', 'Imbabura'),
    'esmeraldas': ('Esmeraldas', 'Esmeraldas'),
    'santo domingo': ('Santo Domingo', 'Santo Domingo de los Tsáchilas'),
    'latacunga': ('Latacunga', 'Cotopaxi'),
    'babahoyo': ('Babahoyo', 'Los Ríos'),
    'quevedo': ('Quevedo', 'Los Ríos'),
    'tulcan': ('Tulcán', 'Carchi'),
    'nueva loja': ('Lago Agrio', 'Sucumbíos'),
    'lago agrio': ('Lago Agrio', 'Sucumbíos'),
    'puyo': ('Pastaza', 'Pastaza'),
    'tena': ('Tena', 'Napo'),
    'coca': ('Francisco de Orellana', 'Orellana'),
    'macas': ('Morona', 'Morona Santiago'),
    'zamora': ('Zamora', 'Zamora Chinchipe'),
    'azogues': ('Azogues', 'Cañar'),
    'guaranda': ('Guaranda', 'Bolívar'),
    'puerto ayora': ('Santa Cruz', 'Galápagos'),
    'salinas': ('Salinas', 'Santa Elena'),
    'la libertad': ('La Libertad', 'Santa Elena'),
}


def detect_geographic_columns(df):
    """
    Detecta automáticamente columnas que podrían contener información geográfica.
    
    Returns:
        List of tuples: [(column_name, geo_type, match_count), ...]
    """
    geo_columns = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Check column name for geographic keywords
        geo_keywords = ['provincia', 'canton', 'cantón', 'parroquia', 'ciudad', 'zona', 
                       'region', 'región', 'ubicacion', 'ubicación', 'sector', 'barrio',
                       'localidad', 'distrito', 'departamento', 'prov', 'cant', 'parr']
        
        is_geo_name = any(kw in col_lower for kw in geo_keywords)
        
        # Count how many values match known geographic names
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            values_lower = df[col].astype(str).str.lower().str.strip()
            
            # Count province matches
            province_matches = sum(1 for v in values_lower.unique() 
                                 if v in ECUADOR_PROVINCES or v in [p.lower() for p in ECUADOR_PROVINCES.values()])
            
            # Count canton matches
            canton_matches = sum(1 for v in values_lower.unique() if v in ECUADOR_CANTONES)
            
            total_matches = province_matches + canton_matches
            
            if total_matches > 0 or is_geo_name:
                geo_type = 'provincia' if province_matches > canton_matches else 'canton'
                geo_columns.append((col, geo_type, total_matches))
    
    # Sort by match count
    geo_columns.sort(key=lambda x: x[2], reverse=True)
    
    return geo_columns


def normalize_location(value, geo_type='provincia'):
    """
    Normaliza un nombre de ubicación al formato estándar.
    """
    if pd.isna(value):
        return None
    
    value_clean = str(value).lower().strip()
    value_clean = value_clean.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    
    if geo_type == 'provincia':
        # Try direct match
        if value_clean in ECUADOR_PROVINCES:
            return ECUADOR_PROVINCES[value_clean]
        
        # Try matching normalized province names
        for key, normalized in ECUADOR_PROVINCES.items():
            key_clean = key.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
            if key_clean == value_clean:
                return normalized
        
        # Try matching the value itself if it's already normalized
        for normalized in ECUADOR_PROVINCES.values():
            if normalized.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n') == value_clean:
                return normalized
    
    elif geo_type == 'canton':
        if value_clean in ECUADOR_CANTONES:
            return ECUADOR_CANTONES[value_clean]
    
    return str(value).strip()  # Return original if no match


def aggregate_by_location(df, geo_col, target_col, agg_func='mean'):
    """
    Agrega datos por ubicación geográfica.
    
    Returns:
        DataFrame con columnas: Location, Value, Count
    """
    df_work = df.copy()
    
    # Normalize location names
    df_work['_location_normalized'] = df_work[geo_col].apply(
        lambda x: normalize_location(x, 'provincia')
    )
    
    # Remove null locations
    df_work = df_work[df_work['_location_normalized'].notna()]
    
    # IMPORTANT: Convert target column to numeric
    # This fixes the "agg function failed [how->mean,dtype->object]" error
    if df_work[target_col].dtype == 'object':
        # Try to convert to numeric, coerce errors to NaN
        df_work[target_col] = pd.to_numeric(df_work[target_col], errors='coerce')
        # Fill NaN with 0 for aggregation
        df_work[target_col] = df_work[target_col].fillna(0)
    
    # Remove rows where target is NaN
    df_work = df_work[df_work[target_col].notna()]
    
    if df_work.empty:
        return pd.DataFrame(columns=['Location', 'Value', 'Count'])
    
    try:
        # Aggregate
        if agg_func == 'mean':
            agg_df = df_work.groupby('_location_normalized')[target_col].mean().reset_index()
            agg_df.columns = ['Location', 'Value']
        elif agg_func == 'sum':
            agg_df = df_work.groupby('_location_normalized')[target_col].sum().reset_index()
            agg_df.columns = ['Location', 'Value']
        elif agg_func == 'count':
            agg_df = df_work.groupby('_location_normalized').size().reset_index()
            agg_df.columns = ['Location', 'Value']
        else:  # proportion of positive cases
            def calc_proportion(x):
                try:
                    return (x == 1).sum() / len(x) if len(x) > 0 else 0
                except:
                    return 0
            agg_df = df_work.groupby('_location_normalized')[target_col].apply(calc_proportion).reset_index()
            agg_df.columns = ['Location', 'Value']
        
        # Add count
        count_df = df_work.groupby('_location_normalized').size().reset_index()
        count_df.columns = ['Location', 'Count']
        
        agg_df = agg_df.merge(count_df, on='Location', how='left')
        
        return agg_df
    
    except Exception as e:
        # Return empty DataFrame if aggregation fails
        print(f"Error in aggregation: {e}")
        return pd.DataFrame(columns=['Location', 'Value', 'Count'])


def classify_danger_level(value, thresholds=None):
    """
    Clasifica el nivel de peligrosidad basado en el valor.
    
    Returns:
        str: 'alto', 'medio', 'bajo'
    """
    if thresholds is None:
        # Use quartiles
        thresholds = {'alto': 0.66, 'medio': 0.33}
    
    if value >= thresholds['alto']:
        return 'alto'
    elif value >= thresholds['medio']:
        return 'medio'
    else:
        return 'bajo'


def get_ecuador_geojson():
    """
    Obtiene el GeoJSON de las provincias de Ecuador.
    Primero intenta cargar desde archivo local, si no existe lo descarga.
    
    Returns:
        dict: GeoJSON data o None si hay error
    """
    import os
    
    local_path = os.path.join(os.path.dirname(__file__), 'data', 'ecuador_provinces.geojson')
    
    # Try local file first
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    # Try to download from a public source
    geojson_urls = [
        'https://raw.githubusercontent.com/jpmarindiaz/geo-collection/master/ec/ec-all.geo.json',
        'https://raw.githubusercontent.com/johan/world.geo.json/master/countries/ECU.geo.json'
    ]
    
    for url in geojson_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                geojson = response.json()
                
                # Save locally for future use
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'w', encoding='utf-8') as f:
                    json.dump(geojson, f)
                
                return geojson
        except:
            continue
    
    return None


def create_choropleth_data(agg_df, target_label='Índice'):
    """
    Prepara datos para un mapa coroplético.
    
    Returns:
        Dict con datos para plotly
    """
    # Normalize values to 0-1 range
    if agg_df['Value'].max() > agg_df['Value'].min():
        agg_df['Value_Normalized'] = (agg_df['Value'] - agg_df['Value'].min()) / (agg_df['Value'].max() - agg_df['Value'].min())
    else:
        agg_df['Value_Normalized'] = 0.5
    
    # Classify danger levels
    q33 = agg_df['Value'].quantile(0.33)
    q66 = agg_df['Value'].quantile(0.66)
    
    agg_df['Nivel'] = agg_df['Value'].apply(
        lambda x: 'Alto' if x >= q66 else ('Medio' if x >= q33 else 'Bajo')
    )
    
    # Color mapping
    color_map = {
        'Alto': '#FF4444',    # Rojo
        'Medio': '#FFAA00',   # Amarillo/Naranja
        'Bajo': '#44AA44'     # Verde
    }
    
    agg_df['Color'] = agg_df['Nivel'].map(color_map)
    
    return agg_df.to_dict('records')


def generate_map_html(agg_df, target_label='Índice de Violencia', title='Mapa de Ecuador'):
    """
    Genera HTML para un mapa interactivo usando Folium.
    
    Returns:
        str: HTML del mapa o None si hay error
    """
    try:
        import folium
        from folium import plugins
    except ImportError:
        return None
    
    # Center of Ecuador
    center_lat = -1.8312
    center_lon = -78.1834
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='CartoDB positron'
    )
    
    # Color scale based on values
    if len(agg_df) == 0:
        return m._repr_html_()
    
    min_val = agg_df['Value'].min()
    max_val = agg_df['Value'].max()
    
    def get_color(value):
        if max_val == min_val:
            return '#FFAA00'
        
        normalized = (value - min_val) / (max_val - min_val)
        if normalized >= 0.66:
            return '#FF4444'  # Rojo - Alto
        elif normalized >= 0.33:
            return '#FFAA00'  # Amarillo - Medio
        else:
            return '#44AA44'  # Verde - Bajo
    
    # Province approximate coordinates (simplified)
    province_coords = {
        'Azuay': (-2.9, -79.0),
        'Bolívar': (-1.6, -79.0),
        'Cañar': (-2.5, -79.0),
        'Carchi': (0.8, -77.8),
        'Chimborazo': (-1.7, -78.7),
        'Cotopaxi': (-0.9, -78.6),
        'El Oro': (-3.3, -79.9),
        'Esmeraldas': (0.9, -79.6),
        'Galápagos': (-0.9, -89.6),
        'Guayas': (-2.2, -79.9),
        'Imbabura': (0.4, -78.1),
        'Loja': (-4.0, -79.2),
        'Los Ríos': (-1.5, -79.5),
        'Manabí': (-1.0, -80.4),
        'Morona Santiago': (-2.3, -78.1),
        'Napo': (-0.9, -77.8),
        'Orellana': (-0.9, -76.9),
        'Pastaza': (-1.5, -77.5),
        'Pichincha': (-0.2, -78.5),
        'Santa Elena': (-2.2, -80.8),
        'Santo Domingo de los Tsáchilas': (-0.25, -79.15),
        'Sucumbíos': (0.1, -77.0),
        'Tungurahua': (-1.3, -78.6),
        'Zamora Chinchipe': (-4.0, -78.9)
    }
    
    # Add markers or circles for each location
    for _, row in agg_df.iterrows():
        location = row['Location']
        value = row['Value']
        count = row.get('Count', 0)
        color = get_color(value)
        
        # Try to find coordinates
        coords = province_coords.get(location)
        
        if coords:
            # Circle marker
            folium.CircleMarker(
                location=coords,
                radius=max(10, min(50, count / 10)) if count > 0 else 20,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"<b>{location}</b><br>{target_label}: {value:.3f}<br>Registros: {count}",
                tooltip=f"{location}: {value:.3f}"
            ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 10px; border: 2px solid gray;
                border-radius: 5px; font-family: Arial;">
        <b>{target_label}</b><br>
        <span style="color: #FF4444;">●</span> Alto<br>
        <span style="color: #FFAA00;">●</span> Medio<br>
        <span style="color: #44AA44;">●</span> Bajo
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m._repr_html_()


def generate_plotly_map(agg_df, target_label='Índice'):
    """
    Genera un mapa de barras/scatter como alternativa si no hay GeoJSON.
    
    Returns:
        plotly Figure
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    if agg_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No hay datos geográficos disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calculate normalized values and levels
    if agg_df['Value'].max() > agg_df['Value'].min():
        norm = (agg_df['Value'] - agg_df['Value'].min()) / (agg_df['Value'].max() - agg_df['Value'].min())
    else:
        norm = pd.Series([0.5] * len(agg_df))
    
    agg_df = agg_df.copy()
    agg_df['Normalized'] = norm
    agg_df['Nivel'] = agg_df['Normalized'].apply(
        lambda x: 'Alto' if x >= 0.66 else ('Medio' if x >= 0.33 else 'Bajo')
    )
    
    # Sort by value descending
    agg_df = agg_df.sort_values('Value', ascending=True)
    
    # Create horizontal bar chart (easier to read with province names)
    fig = px.bar(
        agg_df, 
        y='Location', 
        x='Value',
        color='Nivel',
        color_discrete_map={'Alto': '#FF4444', 'Medio': '#FFAA00', 'Bajo': '#44AA44'},
        orientation='h',
        title=f'🗺️ {target_label} por Ubicación Geográfica',
        labels={'Value': target_label, 'Location': 'Ubicación'},
        hover_data=['Count']
    )
    
    fig.update_layout(
        height=max(400, len(agg_df) * 30),
        showlegend=True,
        legend_title_text='Nivel de Riesgo',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig
