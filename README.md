# Plataforma SQA - Análisis Causal Neutrosófico

Bienvenido al repositorio de la **Plataforma Digital de Análisis Causal Neutrosófico para Fenómenos Sociales**.

## 🚀 Inicio Rápido en Windows (Recomendado)

He creado una forma muy sencilla para que inicies tu programa con solo un par de clics, ideal si estás en el entorno de Windows.

1. **Descarga el código** o clona este repositorio en tu computadora.
2. Abre tu **Explorador de Archivos** y entra a la carpeta principal del proyecto.
3. Haz **Doble clic** sobre el archivo `iniciar_programa.bat`.

¡Y listo! Se abrirá una ventana de comandos que hará todo el trabajo por ti automáticamente:
- Verificará que tengas Python instalado.
- Si es la primera vez que lo abres, creará tu entorno virtual (`venv`) e instalará todas las dependencias necesarias.
- Iniciará de inmediato la aplicación web con Streamlit.

---

## 💻 Inicio Manual (A través de la consola / MAC / Linux)

Si utilizas otro sistema operativo o prefieres ejecutarlo paso a paso usando tu terminal de comandos (CMD, PowerShell, Bash), sigue estos pasos:

**Paso 1:** Abre tu terminal y navega hacia la carpeta principal del proyecto.
```bash
cd ruta/donde/descargaste/el/proyecto
```

**Paso 2:** Crea y activa el entorno virtual (solo la primera vez). Esto es esencial para que Python sepa qué librerías usar.
- **En Windows:**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **En Mac / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
*(Sabrás que funcionó porque aparecerá un `(venv)` al inicio de tu línea de comandos).*

**Paso 3:** Instala las dependencias del proyecto (solo la primera vez):
```bash
pip install -r app/requirements.txt
```

**Paso 4:** Ingresa a la subcarpeta de código fuente y ejecuta la aplicación:
```bash
cd app
streamlit run app.py
```

Tu navegador se abrirá automáticamente en `http://localhost:8501`.

---

## 📚 Más Documentación

Para ver la documentación detalla sobre las características de la plataforma, los algoritmos utilizados y los detalles técnicos, por favor revisa el archivo de documentación completa ubicado en [app/README.md](app/README.md) y los manuales en la carpeta `app/`.
