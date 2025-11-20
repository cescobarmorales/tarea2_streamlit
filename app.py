import os

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title='mapa de emisiones de co₂',
    layout='wide'
)

BASE_DIR = os.path.dirname(__file__)
SHP_PATH = os.path.join(BASE_DIR, 'datos_iniciales_tarea_2', '50m_cultural', 'ne_50m_admin_0_countries.shp')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'annual-co2-emissions-per-country.csv')


@st.cache_data
def load_world(shp_path: str):
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f'no se encontró el shapefile: {shp_path}')

    world = gpd.read_file(shp_path)

    world = world.rename(columns={'ISO_A3': 'code'})
    world['code'] = world['code'].str.upper()
    world_master = (
        world[['code', 'NAME', 'geometry']]
        .drop_duplicates(subset=['code'])
        .rename(columns={'NAME': 'country'})
        .set_index('code')
    )

    geojson_world = world_master['geometry'].__geo_interface__

    return world_master, geojson_world


@st.cache_data
def load_emissions(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'no se encontró el csv de emisiones: {csv_path}')

    df = pd.read_csv(csv_path)

    df = df.rename(columns={'Entity': 'country', 'Code': 'code', 'Year': 'year'})
    df['code'] = df['code'].str.upper()
    df = df[df['code'].str.len() == 3]
    value_col = [c for c in df.columns if c not in ['country', 'code', 'year']]
    if not value_col:
        raise ValueError('no se encontró ninguna columna de emisiones distinta de country/code/year')

    df = df.rename(columns={value_col[0]: 'co2'})

    return df[['country', 'code', 'year', 'co2']]


def make_co2_map(df_co2: pd.DataFrame,
                 world_master: gpd.GeoDataFrame,
                 geojson_world: dict,
                 year: int,
                 metric_type: str = 'Emisiones totales (toneladas)',
                 year_base: int = None):
    if 'Emisiones relativas' in metric_type:
        total_global = df_co2[df_co2['year'] == year]['co2'].sum()
        co2_year = (
            df_co2[df_co2['year'] == year][['code', 'co2']]
            .groupby('code', as_index=False)
            .agg({'co2': 'sum'})
        )
        if total_global > 0:
            co2_year['co2'] = (co2_year['co2'] / total_global) * 100
        co2_year = co2_year.set_index('code')
        colorbar_title = 'Emisiones (% del total global)'
        title_suffix = ' - Porcentaje del Total Global'
    elif 'Cambio respecto a año base' in metric_type and year_base is not None:
        df_base = (
            df_co2[df_co2['year'] == year_base][['code', 'co2']]
            .groupby('code', as_index=False)
            .agg({'co2': 'sum'})
            .rename(columns={'co2': 'co2_base'})
        )
        df_current = (
            df_co2[df_co2['year'] == year][['code', 'co2']]
            .groupby('code', as_index=False)
            .agg({'co2': 'sum'})
        )
        co2_year = df_current.merge(df_base, on='code', how='left')
        co2_year['co2_base'] = co2_year['co2_base'].fillna(0)
        co2_year['co2'] = co2_year.apply(
            lambda row: ((row['co2'] - row['co2_base']) / row['co2_base'] * 100) if row['co2_base'] > 0 else 0,
            axis=1
        )
        co2_year = co2_year[['code', 'co2']].set_index('code')
        colorbar_title = f'Cambio % respecto a {year_base}'
        title_suffix = f' - Cambio respecto a {year_base}'
    else:
        co2_year = (
            df_co2[df_co2['year'] == year][['code', 'co2']]
            .groupby('code', as_index=False)
            .agg({'co2': 'sum'})
            .set_index('code')
        )
        colorbar_title = 'Emisiones (toneladas CO₂)'
        title_suffix = ''

    world_y = world_master.join(co2_year, how='left')
    g_with = world_y[world_y['co2'].notna()].reset_index()
    g_no = world_y[world_y['co2'].isna()].reset_index()

    fig = px.choropleth(
        g_with,
        geojson=geojson_world,
        locations='code',
        color='co2',
        hover_name='country',
        projection='natural earth',
        color_continuous_scale='Reds'
    )

    if not g_no.empty:
        fig_grey = px.choropleth(
            g_no,
            geojson=geojson_world,
            locations='code',
            color_discrete_sequence=['#d0d0d0'],
            hover_name='country',
            projection='natural earth'
        )
        for trace in fig_grey.data:
            trace.showlegend = False
            fig.add_trace(trace)

    fig.update_geos(
        fitbounds='locations',
        visible=False,
        showcountries=True,
        showcoastlines=True,
        showland=True,
        showocean=True,
        landcolor='lightgray',
        oceancolor='lightblue'
    )
    
    fig.update_layout(
        title={
            'text': f'Emisiones de CO₂ por País - Año {year}{title_suffix}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        coloraxis_colorbar={
            'title': {
                'text': colorbar_title,
                'font': {'size': 12}
            }
        },
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    if not g_no.empty:
        fig.add_annotation(
            text=f"Países en gris: sin datos disponibles para {year}",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1
        )

    return fig


def make_time_series_plot(df_co2: pd.DataFrame, selected_countries: list, year: int, use_log_scale: bool = False):
    if not selected_countries:
        df_year = df_co2[df_co2['year'] == year].groupby('code')['co2'].sum().sort_values(ascending=False)
        selected_countries = df_year.head(5).index.tolist()
    
    df_filtered = df_co2[df_co2['code'].isin(selected_countries)].copy()
    
    if df_filtered.empty:
        fig = px.line()
        fig.update_layout(
            title='No hay datos disponibles para los países seleccionados',
            height=400
        )
        return fig
    
    df_series = (
        df_filtered.groupby(['code', 'country', 'year'])['co2']
        .sum()
        .reset_index()
    )
    
    fig = px.line(
        df_series,
        x='year',
        y='co2',
        color='country',
        markers=True,
        title='Evolución de Emisiones de CO₂ por País',
        labels={
            'year': 'Año',
            'co2': 'Emisiones (toneladas CO₂)',
            'country': 'País'
        }
    )
    
    fig.add_vline(
        x=year,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Año seleccionado: {year}",
        annotation_position="top"
    )
    
    if use_log_scale:
        fig.update_yaxes(type="log", title="Emisiones (toneladas CO₂) - Escala Logarítmica")
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def make_ranking_chart(df_co2: pd.DataFrame, year: int, top_n: int = 15):
    df_year = (
        df_co2[df_co2['year'] == year]
        .groupby(['code', 'country'])['co2']
        .sum()
        .reset_index()
        .sort_values('co2', ascending=False)
        .head(top_n)
    )
    
    if df_year.empty:
        fig = px.bar()
        fig.update_layout(
            title='No hay datos disponibles para este año',
            height=400
        )
        return fig
    
    fig = px.bar(
        df_year,
        x='co2',
        y='country',
        orientation='h',
        title=f'Principales {top_n} Países por Emisiones de CO₂ - Año {year}',
        labels={
            'co2': 'Emisiones (toneladas CO₂)',
            'country': 'País'
        },
        color='co2',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=max(400, top_n * 30),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig


def make_comparison_chart(df_co2: pd.DataFrame, selected_countries: list, year: int):
    if not selected_countries:
        return None
    
    df_comparison = (
        df_co2[(df_co2['code'].isin(selected_countries)) & (df_co2['year'] == year)]
        .groupby(['code', 'country'])['co2']
        .sum()
        .reset_index()
        .sort_values('co2', ascending=False)
    )
    
    if df_comparison.empty:
        fig = px.bar()
        fig.update_layout(
            title='No hay datos disponibles para los países seleccionados',
            height=400
        )
        return fig
    
    fig = px.bar(
        df_comparison,
        x='country',
        y='co2',
        title=f'Comparación de Emisiones de CO₂ - Año {year}',
        labels={
            'co2': 'Emisiones (toneladas CO₂)',
            'country': 'País'
        },
        color='co2',
        color_continuous_scale='Oranges'
    )
    
    fig.update_layout(
        height=500,
        xaxis={'categoryorder': 'total descending'},
        showlegend=False
    )
    
    fig.update_traces(
        texttemplate='%{y:,.0f}',
        textposition='outside'
    )
    
    return fig


def main():
    st.title('Visualización Interactiva de Emisiones de CO₂')
    st.markdown(
        """
        Esta aplicación muestra las emisiones anuales de CO₂ por país usando datos de **Our World In Data**.
        Explora distintos años y compara cómo cambia el mapa a lo largo del tiempo.
        """
    )

    world_master, geojson_world = load_world(SHP_PATH)
    df_co2 = load_emissions(CSV_PATH)

    st.sidebar.header('Controles')
    st.sidebar.markdown("---")

    min_year = int(df_co2['year'].min())
    max_year = int(df_co2['year'].max())

    años_destacados = [1751, 1851, 1951, 2024]
    años_destacados = [a for a in años_destacados if min_year <= a <= max_year]

    preset = st.sidebar.selectbox(
        'Años destacados',
        options=['ninguno'] + [str(a) for a in años_destacados],
        index=0
    )

    if preset != 'ninguno':
        year_default = int(preset)
    else:
        year_default = max_year

    year = st.sidebar.slider(
        'Año',
        min_value=min_year,
        max_value=max_year,
        value=year_default,
        step=1
    )

    st.sidebar.markdown("---")
    
    st.sidebar.subheader('Selección de Países')
    
    países_disponibles = sorted(df_co2['country'].unique())
    
    selected_countries_names = st.sidebar.multiselect(
        'Selecciona países para comparar',
        options=países_disponibles,
        default=[]
    )
    
    country_to_code = dict(zip(df_co2['country'], df_co2['code']))
    selected_countries = [country_to_code.get(name) for name in selected_countries_names if name in country_to_code]
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader('Opciones de Visualización')
    
    metric_type = st.sidebar.selectbox(
        'Tipo de métrica',
        options=['Emisiones totales (toneladas)', 'Emisiones relativas (% del total global)', 'Cambio respecto a año base'],
        index=0
    )
    
    year_base = None
    if 'Cambio respecto a año base' in metric_type:
        year_base = st.sidebar.number_input(
            'Año base para comparación',
            min_value=min_year,
            max_value=max_year,
            value=min_year,
            step=1
        )
    
    use_log_scale = st.sidebar.checkbox(
        'Usar escala logarítmica',
        value=False
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Nota:** 
        - Los países con datos se muestran en escala de rojos según sus emisiones
        - Los países sin datos para el año seleccionado se muestran en gris
        - Todos los países del mundo son visibles en el mapa
        - Selecciona países arriba para ver comparaciones y series temporales
        """
    )

    if year < min_year or year > max_year:
        st.warning(f'No hay datos para el año {year}. El rango válido es {min_year}–{max_year}.')
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Mapa Interactivo", 
        "Series Temporales",
        "Ranking de Países",
        "Comparación",
        "Tabla de Datos",
        "Acerca de"
    ])

    with tab1:
        st.subheader(f'Mapa de Emisiones de CO₂ - Año {year}')
        
        fig = make_co2_map(df_co2, world_master, geojson_world, year, metric_type, year_base)
        st.plotly_chart(fig, use_container_width=True)
        
        if 'Emisiones relativas' in metric_type:
            st.info("**Métrica:** Mostrando emisiones como porcentaje del total global. Útil para ver la contribución relativa de cada país.")
        elif 'Cambio respecto a año base' in metric_type and year_base:
            st.info(f"**Métrica:** Mostrando cambio porcentual respecto al año base {year_base}. Valores positivos indican aumento, negativos indican disminución.")
        else:
            st.info("**Métrica:** Mostrando emisiones totales en toneladas de CO₂.")
        df_year = df_co2[df_co2['year'] == year]
        países_con_datos = df_year['code'].nunique()
        total_países = len(world_master)
        países_sin_datos = total_países - países_con_datos
        
        st.info(
            f"""
            **Estadísticas del año {year}:**
            - Países con datos: {países_con_datos}
            - Países sin datos (mostrados en gris): {países_sin_datos}
            - Total de países en el mapa: {total_países}
            """
        )

    with tab2:
        st.subheader('Evolución Temporal de Emisiones')
        st.markdown(
            """
            Visualiza la evolución histórica de emisiones de CO₂ para países seleccionados.
            La línea vertical marca el año actual seleccionado.
            """
        )
        
        if selected_countries:
            fig_series = make_time_series_plot(df_co2, selected_countries, year, use_log_scale)
            st.plotly_chart(fig_series, use_container_width=True)
            
            st.info(
                f"""
                **Países mostrados:** {', '.join(selected_countries_names)}
                
                **Nota:** Si no seleccionas países, se mostrarán automáticamente los 5 países con mayores emisiones en el año {year}.
                """
            )
        else:
            fig_series = make_time_series_plot(df_co2, [], year, use_log_scale)
            st.plotly_chart(fig_series, use_container_width=True)
            st.info("**Nota:** Selecciona países en el sidebar para personalizar esta visualización.")
    
    with tab3:
        st.subheader(f'Ranking de Países - Año {year}')
        st.markdown(
            """
            Principales países emisores de CO₂ en el año seleccionado.
            """
        )
        
        top_n = st.slider(
            'Número de países a mostrar',
            min_value=5,
            max_value=30,
            value=15,
            step=5
        )
        
        fig_ranking = make_ranking_chart(df_co2, year, top_n)
        st.plotly_chart(fig_ranking, use_container_width=True)
        
        df_year_rank = (
            df_co2[df_co2['year'] == year]
            .groupby(['code', 'country'])['co2']
            .sum()
            .reset_index()
            .sort_values('co2', ascending=False)
            .head(top_n)
        )
        
        if not df_year_rank.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("País líder", df_year_rank.iloc[0]['country'])
                st.metric("Emisiones del líder", f"{df_year_rank.iloc[0]['co2']:,.0f} toneladas")
            with col2:
                total_top = df_year_rank['co2'].sum()
                total_global = df_co2[df_co2['year'] == year]['co2'].sum()
                porcentaje = (total_top / total_global * 100) if total_global > 0 else 0
                st.metric("Total principales países", f"{total_top:,.0f} toneladas")
                st.metric("% del total global", f"{porcentaje:.1f}%")
    
    with tab4:
        st.subheader(f'Comparación entre Países - Año {year}')
        st.markdown(
            """
            Compara las emisiones de CO₂ entre países seleccionados en el año actual.
            """
        )
        
        if selected_countries:
            fig_comparison = make_comparison_chart(df_co2, selected_countries, year)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                df_comp = (
                    df_co2[(df_co2['code'].isin(selected_countries)) & (df_co2['year'] == year)]
                    .groupby(['country'])['co2']
                    .sum()
                    .reset_index()
                    .sort_values('co2', ascending=False)
                )
                
                st.markdown("### Estadísticas de Comparación")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("País con más emisiones", df_comp.iloc[0]['country'])
                with col2:
                    ratio = df_comp.iloc[0]['co2'] / df_comp.iloc[-1]['co2'] if len(df_comp) > 1 and df_comp.iloc[-1]['co2'] > 0 else 0
                    st.metric("Ratio máximo/mínimo", f"{ratio:.1f}x")
                with col3:
                    st.metric("Total comparado", f"{df_comp['co2'].sum():,.0f} toneladas")
            else:
                st.warning("No hay datos disponibles para los países seleccionados en este año.")
        else:
            st.info("**Selecciona países en el sidebar** para ver la comparación.")
            st.markdown("---")
            st.markdown("### Países sugeridos para comparar:")
            
            df_suggest = (
                df_co2[df_co2['year'] == year]
                .groupby(['country', 'code'])['co2']
                .sum()
                .reset_index()
                .sort_values('co2', ascending=False)
                .head(10)
            )
            
            st.dataframe(
                df_suggest[['country', 'co2']],
                use_container_width=True,
                hide_index=True
            )
    
    with tab5:
        st.subheader(f'Tabla de Emisiones por País - Año {year}')
        
        df_year = (
            df_co2[df_co2['year'] == year][['country', 'code', 'co2']]
            .groupby(['country', 'code'], as_index=False)
            .agg({'co2': 'sum'})
            .sort_values('co2', ascending=False)
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de países con datos", len(df_year))
        with col2:
            st.metric("Emisiones totales (toneladas)", f"{df_year['co2'].sum():,.0f}")
        with col3:
            st.metric("Promedio por país", f"{df_year['co2'].mean():,.0f}")
        
        st.dataframe(
            df_year,
            use_container_width=True,
            hide_index=True
        )
        
        csv = df_year.to_csv(index=False)
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name=f'emisiones_co2_{year}.csv',
            mime='text/csv'
        )
    
    with tab6:
        st.subheader('Acerca de esta Aplicación')
        
        st.markdown("""
        Esta aplicación web interactiva permite explorar y analizar las emisiones de CO₂ a nivel global 
        utilizando datos de Our World In Data (OWID) y visualizaciones interactivas desarrolladas con 
        Plotly y Streamlit.
        """)
        
        st.markdown("---")
        st.markdown("### Datasets Utilizados")
        
        st.markdown("""
        #### 1. Annual CO₂ emissions per country
        - **Fuente:** Our World In Data (OWID)
        - **Datos originales:** Global Carbon Budget (2025) – Global Carbon Project
        - **Archivo:** `annual-co2-emissions-per-country.csv`
        - **Columnas principales:**
          - `Entity`: Nombre del país o región
          - `Code`: Código ISO alpha-3 del país
          - `Year`: Año de la observación
          - `Annual CO₂ emissions`: Emisiones anuales en toneladas de CO₂
        - **Cobertura:** Países y regiones agregadas (World, Asia, Europe, etc.) por año
        
        #### 2. Shapefiles Geográficos
        - **Fuente:** Natural Earth
        - **Archivo:** `ne_50m_admin_0_countries.shp`
        - **Resolución:** 50m
        - **Proyección:** WGS84
        - **Uso:** Geometrías de países para visualización en mapas coropléticos
        """)
        
        st.markdown("---")
        st.markdown("### Unidades y Períodos")
        
        st.markdown("""
        #### Unidades
        - **Emisiones de CO₂:** Toneladas de CO₂ (no millones de toneladas)
        - **Emisiones relativas:** Porcentaje del total global (%)
        - **Cambio porcentual:** Variación porcentual respecto a un año base (%)
        
        #### Período de Datos
        - **Rango temporal:** 1750 - 2024
        - **Años disponibles:** Varían por país según disponibilidad histórica
        - **Última actualización:** Noviembre 2025 (según fuente OWID)
        
        #### Notas sobre las Unidades
        - Las emisiones se basan en el **territorio** (producción dentro del país)
        - **No incluyen** emisiones de bienes importados (emisiones de consumo)
        - Las emisiones de **aviación y transporte marítimo internacional** no están incluidas 
          en los datos por país individual
        """)
        
        st.markdown("---")
        st.markdown("### Decisiones de Diseño")
        
        st.markdown("""
        #### 1. Paleta de Colores: Escala de Rojos (Reds)
        
        **Justificación:**
        - Los **rojos** son universalmente asociados con peligro, alerta y calor, lo que los hace 
          ideales para representar emisiones de CO₂ (un problema ambiental crítico)
        - La escala continua de rojos permite una lectura intuitiva: **tonos más oscuros = mayores emisiones**
        - Esta elección es consistente con visualizaciones estándar en ciencia climática y 
          referencias como Our World In Data
        - Los países **sin datos se muestran en gris (#d0d0d0)** para diferenciarlos claramente 
          de países con emisiones bajas (que aparecerían en rojo muy claro)
        
        **Alternativas consideradas:**
        - Escala de azules: Menos intuitiva para representar un problema ambiental
        - Escala divergente: No aplica ya que no hay valores "negativos" o "positivos" en emisiones
        """)
        
        st.markdown("""
        #### 2. Visibilidad de Todos los Países en el Mapa
        
        **Justificación:**
        - **Todos los países del mundo son visibles** en el mapa, incluso aquellos sin datos para 
          el año seleccionado
        - Los países sin datos se muestran en **gris** para mantener el contexto geográfico completo
        - Esta decisión permite:
          - **Comparación geográfica:** Ver qué regiones tienen datos y cuáles no
          - **Transparencia:** Evitar la impresión errónea de que ciertos países "no existen"
          - **Análisis de cobertura:** Identificar brechas en los datos disponibles
        
        **Implementación técnica:**
        - Se utiliza un **LEFT JOIN** entre el shapefile (maestro de países) y los datos de emisiones
        - Esto garantiza que ningún país se pierda, incluso si no tiene datos de emisiones
        - Se renderizan dos capas: países con datos (escala de rojos) y países sin datos (gris)
        
        **Alternativas consideradas:**
        - Ocultar países sin datos: Rechazada porque reduce el contexto geográfico
        - Mostrar solo países con datos: Rechazada porque puede confundir al usuario
        """)
        
        st.markdown("""
        #### 3. Escala Logarítmica Opcional
        
        **Justificación:**
        - Las emisiones de CO₂ varían enormemente entre países (desde menos de 1 tonelada hasta 
          millones de toneladas)
        - La escala logarítmica permite visualizar países con rangos muy diferentes en el mismo gráfico
        - Es especialmente útil en **series temporales** y **comparaciones** donde países pequeños 
          quedarían invisibles en escala lineal
        - Se ofrece como **opción** (checkbox) para que el usuario elija según su necesidad analítica
        
        **Cuándo usar:**
        - **Escala lineal:** Para comparar países con emisiones similares o analizar valores absolutos
        - **Escala logarítmica:** Para identificar patrones de crecimiento o comparar países con 
          órdenes de magnitud diferentes
        """)
        
        st.markdown("---")
        st.markdown("### Limitaciones y Consideraciones")
        
        st.markdown("""
        #### 1. Países Sin Datos
        
        **Situación:**
        - No todos los países tienen datos de emisiones para todos los años
        - Algunos países pueden tener datos solo para períodos recientes
        - Algunos países pequeños o territorios pueden no tener datos históricos
        
        **Manejo en la aplicación:**
        - Los países sin datos se muestran en **gris** en el mapa
        - Se incluyen estadísticas sobre cuántos países tienen datos vs. sin datos para cada año
        - Las visualizaciones de series temporales y comparación solo muestran países con datos disponibles
        
        **Impacto:**
        - El análisis global puede estar incompleto para ciertos períodos históricos
        - Las comparaciones entre países deben considerar la disponibilidad de datos
        """)
        
        st.markdown("""
        #### 2. Años Incompletos
        
        **Situación:**
        - Los datos más recientes (2024) pueden estar incompletos o ser estimaciones preliminares
        - Algunos países pueden tener lagunas en años específicos dentro del rango temporal
        
        **Manejo en la aplicación:**
        - El slider de años permite seleccionar cualquier año en el rango 1750-2024
        - Se muestra una advertencia si se selecciona un año fuera del rango válido
        - Los usuarios pueden verificar la cobertura de datos en la tabla de datos
        
        **Recomendación:**
        - Para análisis históricos completos, considerar usar años anteriores a 2024
        - Verificar la cobertura de datos antes de hacer comparaciones entre períodos
        """)
        
        st.markdown("""
        #### 3. Agregaciones y Normalizaciones
        
        **Limitaciones:**
        - Los datos se agregan por país y año sin considerar:
          - **Población:** No se incluyen emisiones per cápita (requeriría datos adicionales)
          - **Tamaño del país:** No se normaliza por área geográfica
          - **Sectores:** No se desglosan por sector económico (energía, transporte, industria, etc.)
        
        **Métricas disponibles:**
        - **Emisiones totales:** Valores absolutos en toneladas
        - **Emisiones relativas:** Porcentaje del total global
        - **Cambio porcentual:** Variación respecto a un año base
        
        **Consideraciones:**
        - Las emisiones totales favorecen países grandes o muy industrializados
        - Para análisis más equitativos, sería necesario incluir datos de población o PIB
        """)
        
        st.markdown("""
        #### 4. Cobertura Temporal y Geográfica
        
        **Limitaciones:**
        - Los datos históricos más antiguos (antes de 1900) tienen menor cobertura geográfica
        - Algunas regiones o territorios pueden estar agrupados bajo entidades políticas diferentes
        - Los cambios en fronteras políticas a lo largo del tiempo no se reflejan en los datos
        
        **Impacto:**
        - Las comparaciones históricas de largo plazo deben interpretarse con cautela
        - Algunos países pueden aparecer o desaparecer del dataset según cambios políticos
        """)
        
        st.markdown("""
        #### 5. Fuente y Metodología de los Datos
        
        **Limitaciones:**
        - Los datos provienen de estimaciones y modelos del Global Carbon Project
        - Pueden existir discrepancias con inventarios nacionales oficiales
        - Las emisiones de ciertos sectores (como deforestación) pueden tener mayor incertidumbre
        
        **Recomendación:**
        - Consultar las fuentes originales (OWID, Global Carbon Project) para detalles metodológicos
        - Los datos son adecuados para análisis comparativos y tendencias, pero pueden requerir 
          validación para usos específicos
        """)
        
        st.markdown("---")
        st.markdown("### Referencias y Fuentes")
        
        st.markdown("""
        - **Our World In Data - CO₂ Emissions:** https://ourworldindata.org/co2-emissions
        - **Global Carbon Project:** https://globalcarbonbudget.org/
        - **Natural Earth:** https://www.naturalearthdata.com/
        - **Streamlit:** https://streamlit.io/
        - **Plotly:** https://plotly.com/python/
        """)
        
        st.markdown("---")
        st.markdown("### Información Técnica")
        
        st.markdown("""
        - **Framework:** Streamlit
        - **Visualizaciones:** Plotly Express
        - **Datos geográficos:** GeoPandas
        - **Análisis de datos:** Pandas
        - **Proyección del mapa:** Natural Earth
        """)


if __name__ == '__main__':
    main()

