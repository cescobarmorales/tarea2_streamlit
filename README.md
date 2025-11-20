# Visualización Interactiva de Emisiones de CO₂

Aplicación web interactiva desarrollada con Streamlit para explorar las emisiones de CO₂ a nivel global utilizando datos de Our World In Data.

## Instalación y Ejecución

### Requisitos

- Python 3.8 o superior
- pip

### Pasos

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
   ```

3. Abrir en el navegador en `http://localhost:8501`

## Dependencias

### Versión mínima de Python
- Python 3.8

### Versiones mínimas de librerías

- `streamlit >= 1.28.0`
- `pandas >= 1.5.0`
- `geopandas >= 0.13.0`
- `plotly >= 5.15.0`
- `numpy >= 1.23.0`
- `shapely >= 2.0.0`
- `fiona >= 1.9.0`

Para ver todas las dependencias, consulta `requirements.txt`.

## Datos

Los datos de emisiones de CO₂ provienen de **Our World In Data** (OWID):
- Dataset: Annual CO₂ emissions per country
- Fuente: Global Carbon Budget (2025) – Global Carbon Project
- Rango temporal: 1750-2024
- Enlace: https://ourworldindata.org/co2-emissions

Los mapas utilizan shapefiles de **Natural Earth** (resolución 50m).
# tarea2_streamlit
