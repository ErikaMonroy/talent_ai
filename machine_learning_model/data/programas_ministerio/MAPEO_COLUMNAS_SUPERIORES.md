# Mapeo de Columnas - Educación Superior

## Archivo Fuente
`MEN_PROGRAMAS_DE_EDUCACI_N_SUPERIOR_20250811.csv`

## Columnas Disponibles en el Archivo Fuente

### Información de Institución
- `codigoinstitucion` - Código único de la institución
- `nombreinstitucion` - Nombre de la institución educativa
- `codigodepartinstitucion` - Código del departamento de la institución
- `nombredepartinstitucion` - Nombre del departamento de la institución
- `codigomunicipioinstitucion` - Código del municipio de la institución
- `nombremunicipioinstitucion` - Nombre del municipio de la institución
- `codigoorigeninstitucional` - Código del origen institucional
- `nombreorigeninstitucional` - Nombre del origen institucional (Oficial/Privado)
- `codigocaracteracademico` - Código del carácter académico
- `nombrecaracteracademico` - Nombre del carácter académico (Universidad, etc.)

### Información del Programa
- `codigoprograma` - Código único del programa
- `nombreprograma` - Nombre del programa académico
- `codigodepartprograma` - Código del departamento donde se ofrece el programa
- `nombredepartprograma` - Nombre del departamento donde se ofrece el programa
- `codigomunicipioprograma` - Código del municipio donde se ofrece el programa
- `nombremunicipioprograma` - Nombre del municipio donde se ofrece el programa
- `codigoestadoprograma` - Código del estado del programa
- `nombreestadoprograma` - Estado del programa (Activo/Inactivo)

### Información Académica
- `cantidadcreditos` - Número de créditos del programa
- `fechaacreditacion` - Fecha de acreditación
- `codigoareaconocimiento` - Código del área de conocimiento
- `nombreareaconocimiento` - Nombre del área de conocimiento
- `codigometodologia` - Código de la metodología
- `nombremetodologia` - Metodología (Presencial, Virtual, etc.)
- `codigonbc` - Código NBC (Núcleo Básico del Conocimiento)
- `nombrenbc` - Nombre del NBC
- `codigonivelformacion` - Código del nivel de formación
- `nombrenivelformacion` - Nivel de formación (Universitaria, Especialización, etc.)
- `codigonivelacademico` - Código del nivel académico
- `nombrenivelacademico` - Nivel académico (Pregrado, Posgrado)
- `codigoperiodicidad` - Código de periodicidad
- `nombreperiodicidad` - Periodicidad (Semestral, Trimestral, etc.)
- `cantidadperiodos` - Cantidad de períodos
- `numeroresolucionacreditacion` - Número de resolución de acreditación
- `codigotipoacreditacion` - Código del tipo de acreditación
- `nombretipoacreditacion` - Tipo de acreditación
- `aniosacreditados` - Años acreditados
- `nombretituloobtenido` - Título que se obtiene
- `fechacreacion` - Fecha de creación del programa

## Mapeo Propuesto a `programas.csv`

### Mapeos Directos
```
nombreprograma → nombre_programa
nombreinstitucion → institucion
nombrenivelacademico → nivel_academico (Pregrado/Posgrado)
nombreareaconocimiento → area_conocimiento
nombremetodologia → modalidad
nombrenbc → nucleo_basico_conocimiento
nombretituloobtenido → titulo_otorgado
nombreestadoprograma → estado_programa
```

### Transformaciones y Concatenaciones

#### ubicacion
```python
# Concatenar información de ubicación
ubicacion = f"{nombremunicipioinstitucion}"
# Ejemplo: "Medellín
```

#### duracion_semestres
```python
# Calcular duración en semestres basado en periodicidad
if nombreperiodicidad == "Semestral":
    duracion_semestres = cantidadperiodos
elif nombreperiodicidad == "Trimestral":
    duracion_semestres = cantidadperiodos / 2  # Aproximación
elif nombreperiodicidad == "Anual":
    duracion_semestres = cantidadperiodos * 2
else:
    duracion_semestres = "No especificado"
```

#### creditos_academicos
```python
creditos_academicos = cantidadcreditos if cantidadcreditos != "NA" else "No especificado"
```

#### formato
```python
# Basado en metodología
formato = nombremetodologia  # "Presencial", "Virtual", "A distancia", etc.
```

#### tipo_institucion
```python
tipo_institucion = nombreorigeninstitucional  # "Oficial" o "Privado"
```

#### caracter_academico
```python
caracter_academico = nombrecaracteracademico  # "Universidad", "Institución Universitaria", etc.
```

### Campos Derivados y Calculados

#### perfil_ocupacional
```python
# Basado en el área de conocimiento y NBC
perfil_ocupacional = f"Profesional en {nombreareaconocimiento} - {nombrenbc}"
```

#### competencias_desarrollar
```python
# Basado en el área de conocimiento
competencias_map = {
    "Ciencias sociales y humanas": "Competencias en investigación social, análisis crítico, comunicación",
    "Ingeniería, arquitectura, urbanismo y afines": "Competencias técnicas, diseño, innovación tecnológica",
    "Ciencias de la salud": "Competencias clínicas, diagnóstico, atención en salud",
    "Ciencias de la educación": "Competencias pedagógicas, didácticas, formación integral",
    # ... más mapeos según áreas disponibles
}
competencias_desarrollar = competencias_map.get(nombreareaconocimiento, "Competencias específicas del área")
```

#### costo_estimado_cop
```python
# Para educación superior, generalmente no hay costo específico en el archivo
# Se puede categorizar por tipo de institución
if nombreorigeninstitucional == "Oficial":
    costo_estimado_cop = "Matrícula pública según estrato"
else:
    costo_estimado_cop = "Consultar con la institución"
```

#### estilo_aprendizaje
```python
# Basado en metodología
estilo_map = {
    "Presencial": "Presencial - Interacción directa",
    "Virtual": "Virtual - Aprendizaje en línea",
    "A distancia": "A distancia - Estudio autónomo",
    "Dual": "Mixto - Presencial y virtual"
}
estilo_aprendizaje = estilo_map.get(nombremetodologia, nombremetodologia)
```

## Mapeo Detallado (ESTRUCTURA LIMPIA)

| Campo Destino | Campo Fuente | Transformación |
|---------------|--------------|----------------|
| **id_programa** | - | ID secuencial único |
| **nombre_programa** | nombreprograma | Directo |
| **nivel_academico** | nombrenivelformacion | Directo |
| **duracion_info** | cantidadperiodos + nombreperiodicidad | Concatenar: "X períodos semestrales", "Y períodos anuales" |
| **formato** | nombremetodologia | Directo |
| **costo** | nombreorigeninstitucional | "Consultar institución" (sin inventar valores) |
| **ubicacion** | nombremunicipioinstitucion | Directo: solo el municipio |
| **institucion** | nombreinstitucion | Directo |
| **codigo_programa** | codigoprograma | Directo |
| **estado_programa** | nombreestadoprograma | Directo |
| **area_conocimiento** | nombreareaconocimiento | Directo |
| **tipo_institucion** | nombreorigeninstitucional | Mapeo: "OFICIAL" → "Oficial", "PRIVADA" → "Privada" |

## Estructura Final del DataFrame (ESTRUCTURA LIMPIA)

```python
columns = [
    'id_programa',          # Secuencial único
    'nombre_programa',      # nombreprograma
    'nivel_academico',      # nombrenivelacademico
    'duracion_info',        # Información de duración (períodos + periodicidad)
    'formato',              # nombremetodologia
    'costo',                # Categorizado por tipo de institución
    'ubicacion',            # nombremunicipioinstitucion + nombredepartinstitucion
    'institucion',          # nombreinstitucion
    'codigo_programa',      # codigoprograma
    'estado_programa',      # nombreestadoprograma
    'area_conocimiento',    # nombreareaconocimiento
    'tipo_institucion'      # nombreorigeninstitucional
]
```

## Consideraciones Especiales

### Filtros de Calidad
1. **Programas Activos**: Filtrar solo programas con `nombreestadoprograma == "Activo"`
2. **Datos Válidos**: Excluir registros con campos críticos vacíos o "NA"
3. **Duplicados**: Deduplicar por `nombreprograma` + `nombreinstitucion` + `nombremunicipioinstitucion`

### Manejo de Valores Faltantes
- `cantidadcreditos == "NA"` → "No especificado"
- `fechaacreditacion == "NA"` → "Sin acreditación"
- Campos vacíos → "No disponible"

### Diferencias con Programas Técnicos
1. **Mayor complejidad académica**: Más campos relacionados con acreditación y calidad
2. **Estructura institucional**: Información más detallada sobre instituciones
3. **Niveles académicos**: Pregrado, especialización, maestría, doctorado
4. **Créditos académicos**: Sistema de créditos vs. horas de duración
5. **Metodologías diversas**: Presencial, virtual, a distancia, dual

## Próximos Pasos

1. **Implementar función `process_university_programs()`** con este mapeo
2. **Validar datos** con muestras del archivo
3. **Probar integración** con `process_technical_programs()`
4. **Ajustar deduplicación** para manejar ambos tipos de programas
5. **Verificar consistencia** en la estructura final de `programas.csv`