"""
ml_engine.py - Motor de Machine Learning para Plataforma SQA

Implementa el flujo completo de Machine Learning según el diagrama del tutor:
1. Análisis Exploratorio de Datos (EDA) con PCA
2. Preprocesamiento y limpieza de datos
3. División de datos (80% entrenamiento / 20% prueba)
4. Múltiples algoritmos: SVM, Random Forest, Decision Tree, KNN, GBM
5. Validación cruzada (Cross-Validation)
6. Métricas de evaluación:
   - Clasificación: Accuracy, Sensibilidad, Especificidad, F1, MCC
   - Regresión: MSE, RMSE, R²
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


def detect_task_type(y):
    """
    Detecta automáticamente si es un problema de clasificación o regresión.
    
    Returns:
        str: 'classification' o 'regression'
    """
    unique_values = pd.Series(y).nunique()
    
    # Si hay menos de 10 valores únicos o son todos enteros pequeños, es clasificación
    if unique_values <= 10:
        return 'classification'
    
    # Si los valores son flotantes con decimales, es regresión
    if pd.Series(y).dtype in ['float64', 'float32']:
        # Check if values are actually integers stored as floats
        if all(pd.Series(y).dropna().apply(lambda x: x == int(x))):
            if unique_values <= 10:
                return 'classification'
        return 'regression'
    
    return 'classification'


def preprocess_data(df, target_col):
    """
    Preprocesa los datos para machine learning.
    - Limpia valores nulos
    - Codifica variables categóricas
    - Escala features numéricas
    
    Returns:
        X: Features preprocesadas
        y: Variable objetivo
        feature_names: Nombres de las features
        scaler: StandardScaler fitted
        label_encoders: Dict de LabelEncoders usados
    """
    df_clean = df.copy()
    
    # Separar features y target
    feature_cols = [c for c in df_clean.columns if c != target_col]
    
    # Manejar valores nulos
    for col in df_clean.columns:
        if df_clean[col].dtype in ['object', 'category']:
            df_clean[col] = df_clean[col].fillna('Unknown')
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Codificar variables categóricas
    label_encoders = {}
    for col in feature_cols:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    # Codificar target si es categórico
    if df_clean[target_col].dtype == 'object' or df_clean[target_col].dtype.name == 'category':
        le_target = LabelEncoder()
        df_clean[target_col] = le_target.fit_transform(df_clean[target_col].astype(str))
        label_encoders['__target__'] = le_target
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler, label_encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en entrenamiento (80%) y prueba (20%).
    Maneja automáticamente casos donde las clases tienen muy pocos ejemplos.
    
    Returns:
        X_train, X_test, y_train, y_test, split_info
    """
    task_type = detect_task_type(y)
    use_stratify = False
    
    if task_type == 'classification':
        # Check if we can use stratification
        # Need at least 2 samples per class to stratify
        value_counts = pd.Series(y).value_counts()
        min_samples_per_class = value_counts.min()
        
        # Also need at least 1 sample per class after split
        # test_size of 0.2 means we need at least ceil(1/0.2) = 5 samples minimum
        min_samples_needed = max(2, int(np.ceil(1 / test_size))) if test_size > 0 else 2
        
        if min_samples_per_class >= min_samples_needed:
            use_stratify = True
    
    try:
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y
            )
        else:
            # Fall back to non-stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    except ValueError as e:
        # If stratification still fails, use non-stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    split_info = {
        'total_samples': len(y),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'train_percentage': round((len(y_train) / len(y)) * 100, 1),
        'test_percentage': round((len(y_test) / len(y)) * 100, 1),
        'stratified': use_stratify
    }
    
    return X_train, X_test, y_train, y_test, split_info


def perform_pca(X, n_components=None, variance_threshold=0.95):
    """
    Realiza Análisis de Componentes Principales (PCA).
    
    Returns:
        X_pca: Datos transformados
        pca: Objeto PCA fitted
        pca_info: Información sobre varianza explicada
    """
    if n_components is None:
        # Determinar número de componentes para explicar 95% de varianza
        pca_full = PCA()
        pca_full.fit(X)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        n_components = min(n_components, X.shape[1], X.shape[0])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    pca_info = {
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_variance_explained': sum(pca.explained_variance_ratio_),
        'original_features': X.shape[1]
    }
    
    return X_pca, pca, pca_info


def feature_selection(X, y, feature_names, k='auto', task_type='classification'):
    """
    Selección de características más relevantes.
    
    Returns:
        X_selected: Features seleccionadas
        selected_features: Nombres de features seleccionadas
        feature_scores: Puntuaciones de importancia
    """
    if k == 'auto':
        k = min(10, X.shape[1])
    
    if task_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_classif, k=k)  # f_classif works for regression too
    
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices and names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Get all scores
    feature_scores = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_,
        'Selected': [i in selected_indices for i in range(len(feature_names))]
    }).sort_values('Score', ascending=False)
    
    return X_selected, selected_features, feature_scores


def get_models(task_type='classification'):
    """
    Retorna diccionario de modelos según el tipo de tarea.
    """
    if task_type == 'classification':
        return {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
    else:
        return {
            'SVM': SVR(kernel='rbf'),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }


def cross_validate_model(model, X, y, cv=5, task_type='classification'):
    """
    Realiza validación cruzada para un modelo.
    
    Returns:
        cv_results: Diccionario con resultados de CV
    """
    if task_type == 'classification':
        scoring = 'accuracy'
        try:
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)
        except:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    else:
        scoring = 'r2'
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'scores': scores.tolist(),
        'mean': round(scores.mean(), 4),
        'std': round(scores.std(), 4),
        'min': round(scores.min(), 4),
        'max': round(scores.max(), 4)
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test, task_type='classification', cv=5):
    """
    Entrena y evalúa múltiples modelos.
    
    Returns:
        results: DataFrame con resultados de todos los modelos
        best_model: Mejor modelo entrenado
        best_model_name: Nombre del mejor modelo
        predictions: Dict con predicciones de cada modelo
    """
    models = get_models(task_type)
    results = []
    predictions = {}
    
    for name, model in models.items():
        try:
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            # Cross-validation
            cv_results = cross_validate_model(model, X_train, y_train, cv=cv, task_type=task_type)
            
            if task_type == 'classification':
                # Métricas de clasificación
                accuracy = accuracy_score(y_test, y_pred)
                
                # Para multiclase, usar average='weighted'
                n_classes = len(np.unique(y_test))
                avg = 'binary' if n_classes == 2 else 'weighted'
                
                precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
                recall = recall_score(y_test, y_pred, average=avg, zero_division=0)  # Sensibilidad
                f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
                
                # MCC (Matthews Correlation Coefficient)
                try:
                    mcc = matthews_corrcoef(y_test, y_pred)
                except:
                    mcc = 0.0
                
                # Especificidad (para binario)
                if n_classes == 2:
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    specificity = 0  # No aplica directamente para multiclase
                
                results.append({
                    'Modelo': name,
                    'Accuracy': round(accuracy, 4),
                    'Sensibilidad': round(recall, 4),
                    'Especificidad': round(specificity, 4),
                    'Precision': round(precision, 4),
                    'F1-Score': round(f1, 4),
                    'MCC': round(mcc, 4),
                    'CV Mean': cv_results['mean'],
                    'CV Std': cv_results['std']
                })
            else:
                # Métricas de regresión
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Modelo': name,
                    'MSE': round(mse, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'R²': round(r2, 4),
                    'CV Mean': cv_results['mean'],
                    'CV Std': cv_results['std']
                })
                
        except Exception as e:
            print(f"Error entrenando {name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Encontrar mejor modelo
    if task_type == 'classification':
        best_idx = results_df['Accuracy'].idxmax() if not results_df.empty else 0
    else:
        best_idx = results_df['R²'].idxmax() if not results_df.empty else 0
    
    if not results_df.empty:
        best_model_name = results_df.loc[best_idx, 'Modelo']
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)  # Re-fit to ensure it's trained
    else:
        best_model_name = None
        best_model = None
    
    return results_df, best_model, best_model_name, predictions


def get_confusion_matrix_data(y_true, y_pred, labels=None):
    """
    Genera datos para la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    return {
        'matrix': cm.tolist(),
        'labels': [str(l) for l in labels]
    }


def run_full_ml_pipeline(df, target_col, test_size=0.2, cv_folds=5, use_pca=False, use_feature_selection=False, k_features='auto'):
    """
    Ejecuta el pipeline completo de Machine Learning.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        test_size: Proporción para test (default 0.2 = 20%)
        cv_folds: Número de folds para cross-validation
        use_pca: Si usar PCA para reducción de dimensionalidad
        use_feature_selection: Si usar selección de características
        k_features: Número de features a seleccionar (o 'auto')
    
    Returns:
        Dictionary con todos los resultados del pipeline
    """
    results = {
        'success': False,
        'error': None
    }
    
    try:
        # 1. Preprocesamiento
        X, y, feature_names, scaler, label_encoders = preprocess_data(df, target_col)
        results['preprocessing'] = {
            'n_samples': len(y),
            'n_features': len(feature_names),
            'feature_names': feature_names
        }
        
        # 2. Detectar tipo de tarea
        task_type = detect_task_type(y)
        results['task_type'] = task_type
        results['target_info'] = {
            'unique_values': int(pd.Series(y).nunique()),
            'value_counts': pd.Series(y).value_counts().to_dict()
        }
        
        # 3. División de datos
        X_train, X_test, y_train, y_test, split_info = split_data(X, y, test_size=test_size)
        results['split_info'] = split_info
        
        # 4. PCA (opcional)
        if use_pca and X_train.shape[1] > 2:
            X_train_pca, pca, pca_info = perform_pca(X_train)
            X_test_pca = pca.transform(X_test)
            results['pca_info'] = pca_info
            X_train_final = X_train_pca
            X_test_final = X_test_pca
        else:
            X_train_final = X_train
            X_test_final = X_test
            results['pca_info'] = None
        
        # 5. Selección de características (opcional)
        if use_feature_selection and not use_pca:
            X_train_sel, selected_features, feature_scores = feature_selection(
                X_train, y_train, feature_names, k=k_features, task_type=task_type
            )
            # Apply same selection to test
            selector_indices = feature_scores[feature_scores['Selected']].index.tolist()
            X_test_sel = X_test[:, selector_indices] if len(selector_indices) > 0 else X_test
            results['feature_selection'] = {
                'selected_features': selected_features,
                'feature_scores': feature_scores.to_dict('records')
            }
            X_train_final = X_train_sel
            X_test_final = X_test_sel
        else:
            results['feature_selection'] = None
        
        # 6. Entrenar y evaluar modelos
        model_results, best_model, best_model_name, predictions = train_and_evaluate_models(
            X_train_final, X_test_final, y_train, y_test, task_type=task_type, cv=cv_folds
        )
        results['model_results'] = model_results.to_dict('records')
        results['best_model_name'] = best_model_name
        
        # 7. Matriz de confusión (solo para clasificación)
        if task_type == 'classification' and best_model_name:
            y_pred_best = predictions[best_model_name]
            all_classes = sorted(np.unique(y))
            results['confusion_matrix'] = get_confusion_matrix_data(y_test, y_pred_best, labels=all_classes)
        
        # 8. Datos para visualización
        results['y_test'] = y_test.tolist()
        if best_model_name:
            results['y_pred'] = predictions[best_model_name].tolist()
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()
    
    return results
