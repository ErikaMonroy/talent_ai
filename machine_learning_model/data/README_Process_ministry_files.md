# 🔍 Sistema de Filtros - process_ministry_files.py

## Descripción
El script `process_ministry_files.py` ahora incluye un sistema de filtros flexible que permite filtrar los programas educativos según diferentes criterios.


## origen de datos

https://www.datos.gov.co/Educaci-n/MEN_PROGRAMAS_DE_EDUCACI-N_SUPERIOR/upr9-nkiz/about_data
https://www.datos.gov.co/Educaci-n/MEN_PROGRAMAS-EDUCACI-N-PARA-EL-TRABAJO-Y-EL-DESAR/2v94-3ypi/about_data


## Filtro por Defecto
Por defecto, el script filtra **solo programas de Bogotá**:
```python
default_filters = {
    'ubicacion': 'Bogotá'  # Filtra programas que contengan "Bogotá" en la ubicación
}
```

## Tipos de Filtros Disponibles

### 1. Filtro por Texto (Contiene)
```python
filtros = {
    'ubicacion': 'Medellín',  # Programas que contengan "Medellín"
    'institucion': 'Universidad',  # Instituciones que contengan "Universidad"
    'areas_conocimiento': 'Salud'  # Áreas relacionadas con "Salud"
}
```

### 2. Filtro por Lista de Valores
```python
filtros = {
    'nivel_academico': ['Técnico', 'Universitario'],  # Solo estos niveles
    'formato': ['Presencial', 'Virtual'],  # Solo estos formatos
    'estado_registro': ['Activo']  # Solo programas activos
}
```

### 3. Filtro por Rango Numérico
```python
filtros = {
    'costo_estimado_cop': {'max': 5000000},  # Costo menor a 5M
    'duracion_semestres': {'min': 4, 'max': 8},  # Entre 4 y 8 semestres
    'duracion_semestres': {'min': 6}  # Mínimo 6 semestres
}
```

## Ejemplos de Uso

### Ejemplo 1: Solo programas de Medellín
```python
filtros = {
    'ubicacion': 'Medellín'
}
```

### Ejemplo 2: Programas universitarios presenciales en Bogotá
```python
filtros = {
    'ubicacion': 'Bogotá',
    'nivel_academico': ['Universitario'],
    'formato': ['Presencial']
}
```

### Ejemplo 3: Programas económicos de corta duración
```python
filtros = {
    'costo_estimado_cop': {'max': 3000000},
    'duracion_semestres': {'max': 4}
}
```

### Ejemplo 4: Programas de salud activos
```python
filtros = {
    'areas_conocimiento': 'Salud',
    'estado_registro': ['Activo']
}
```

## Cómo Modificar los Filtros

1. Abre el archivo `process_ministry_files.py`
2. Busca la función `main()`
3. Modifica el diccionario `default_filters`:

```python
# Cambiar de:
default_filters = {
    'ubicacion': 'Bogotá'
}

# A (ejemplo):
default_filters = {
    'ubicacion': 'Medellín',
    'nivel_academico': ['Universitario'],
    'costo_estimado_cop': {'max': 4000000}
}
```

4. Ejecuta el script: `python process_ministry_files.py`

## Campos Disponibles para Filtrar

- `id_programa`
- `nombre_programa`
- `trabajos_relacionados_1`
- `trabajos_relacionados_2`
- `trabajos_relacionados_3`
- `fuente_recomendacion`
- `nivel_academico`
- `duracion_semestres`
- `formato`
- `nivel_dificultad`
- `requisitos_ingreso`
- `estilo_aprendizaje`
- `costo_estimado_cop`
- `ubicacion`
- `institucion`
- `codigo_snies`
- `estado_registro`
- `metodologia`
- `titulo_otorgado`
- `perfil_ocupacional`
- `competencias_desarrollar`
- `areas_conocimiento`
- `fecha_actualizacion`

## Resultados

- **Sin filtros**: ~19,720 programas totales
- **Con filtro Bogotá**: ~2,440 programas
- El archivo de salida `programas.csv` contendrá solo los programas que cumplan los criterios
- Los logs mostrarán cuántos registros se filtraron

## Notas Importantes

- Los filtros de texto son **case-sensitive** (distinguen mayúsculas/minúsculas)
- Los filtros se aplican con **AND** (todos deben cumplirse)
- Los rangos numéricos usan `min` y/o `max`
- Las listas de valores usan **OR** (cualquier valor de la lista)
- Si no se especifican filtros, se procesan todos los programas