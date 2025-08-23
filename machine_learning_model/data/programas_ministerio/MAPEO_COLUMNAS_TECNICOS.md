# Mapeo de Columnas - Programas Técnicos

## Análisis de Columnas Disponibles

### Archivo Fuente: MEN_PROGRAMAS_EDUCACI_N_PARA_EL_TRABAJO_Y_EL_DESARROLLO_HUMANO_20250811.csv

**Columnas disponibles:**
- Código Secretaria
- Secretaria
- Código Institución
- Nombre Institución
- Código Programa
- Nombre Programa
- Cod Departamento
- Departamento
- Cod Municipio
- Municipio
- Localidad
- Dirección
- Sede
- Estado Programa
- Registro
- Fecha Registro
- Área Desempeño
- Área Desempeño Salud
- Tipo Certificado
- Subtipo Certificado
- Escolaridad
- Jornadas
- Costo
- Duración Horas
- Número Certificación
- Tipo Certificación
- Certificado Calidad
- Estado Certificación
- Entidad Emisora Certificación
- Fecha Otorgamiento
- Fecha Vencimiento
- Latitud
- Longitud
- Año Corte
- Mes Corte
- Fecha Corte

### Archivo Destino: programas.csv (ESTRUCTURA LIMPIA)

**Columnas objetivo (solo datos reales):**
- id_programa
- nombre_programa
- nivel_academico
- duracion_info
- formato
- costo
- ubicacion
- institucion
- codigo_programa
- estado_programa
- area_conocimiento
- tipo_institucion

## Mapeo Propuesto (ESTRUCTURA LIMPIA)

| Campo Destino | Campo Fuente | Transformación/Lógica |
|---------------|--------------|----------------------|
| **id_programa** | - | Generar ID secuencial único |
| **nombre_programa** | Nombre Programa | Directo |
| **nivel_academico** | Tipo Certificado | Mapeo: TÉCNICO LABORAL → "Técnico", CONOCIMIENTOS ACADÉMICOS → "Curso" |
| **duracion_info** | Duración Horas | Directo con unidad: "900 horas", "1200 horas" |
| **formato** | Jornadas | Mapeo: DIURNA/NOCTURNA → "Presencial", FIN DE SEMANA → "Presencial" |
| **costo** | Costo | Limpiar y convertir a número (sin multiplicar) |
| **ubicacion** | Municipio | Directo: solo el municipio |
| **institucion** | Nombre Institución | Directo |
| **codigo_programa** | Código Programa | Directo |
| **estado_programa** | Estado Programa | Directo |
| **area_conocimiento** | Área Desempeño | Directo o "No especificada" si está vacío |
| **tipo_institucion** | - | Valor fijo: "Técnica" |

## Problemas Identificados en el Código Actual

1. **Valores "nan" en campos calculados**: El código actual está generando "nan" en perfil_ocupacional, competencias_desarrollar y areas_conocimiento
2. **Área Desempeño vacía**: Muchos registros tienen el campo "Área Desempeño" vacío, causando los valores "nan"
3. **Costo multiplicado por 10**: El costo parece estar siendo multiplicado incorrectamente (1880000 vs 18800000)

## Soluciones Propuestas

1. **Manejo de valores vacíos**: Usar valores por defecto cuando "Área Desempeño" esté vacío
2. **Validación de datos**: Verificar que los campos no sean None/NaN antes de procesarlos
3. **Corrección de costo**: Revisar la lógica de limpieza del campo Costo
4. **Campos alternativos**: Si "Área Desempeño" está vacío, usar "Tipo Certificado" como alternativa

## Valores por Defecto Sugeridos

- **areas_conocimiento**: "Técnica" si Área Desempeño está vacío
- **perfil_ocupacional**: "Técnico especializado" si no hay área específica
- **competencias_desarrollar**: "Competencias técnicas generales" si no hay área específica
- **metodologia**: "Formación por competencias" (siempre)
- **estilo_aprendizaje**: "Práctico" para técnicos, "Teórico-Práctico" para cursos