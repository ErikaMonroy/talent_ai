# Formato JSON para Predicciones - TalentAI

## Estructura del JSON de Entrada

El endpoint `/api/v1/predictions/predict` requiere el siguiente formato JSON:

### Ejemplo Completo
```json
{
  "user_email": "estudiante@ejemplo.com",
  
  // PUNTAJES ICFES (Rango: 0-500)
  "matematicas": 350.5,
  "lectura_critica": 280.3,
  "ciencias_naturales": 320.1,
  "sociales_ciudadanas": 295.7,
  "ingles": 275.2,
  
  // DIMENSIONES PERSONALES (Rango: 1-5)
  "dimension_1_logico_matematico": 4.2,
  "dimension_2_comprension_comunicacion": 3.8,
  "dimension_3_pensamiento_cientifico": 4.1,
  "dimension_4_analisis_social_humanistico": 3.5,
  "dimension_5_creatividad_innovacion": 4.0,
  "dimension_6_liderazgo_trabajo_equipo": 3.9,
  "dimension_7_pensamiento_critico": 4.3,
  "dimension_8_adaptabilidad_aprendizaje": 3.7,
  
  // CONFIGURACIÓN DEL MODELO
  "model_type": "knn"
}
```

## Descripción de Campos

### 📧 Información del Usuario
- **user_email**: Email del estudiante (requerido)

### 📊 Puntajes ICFES (0-500 puntos)
Estos son los puntajes obtenidos en las pruebas ICFES:
- **matematicas**: Puntaje en matemáticas
- **lectura_critica**: Puntaje en lectura crítica
- **ciencias_naturales**: Puntaje en ciencias naturales
- **sociales_ciudadanas**: Puntaje en sociales y ciudadanas
- **ingles**: Puntaje en inglés

### 🧠 Dimensiones Personales (1-5 escala)
Estas dimensiones evalúan habilidades y características personales:
- **dimension_1_logico_matematico**: Capacidad de razonamiento lógico-matemático
- **dimension_2_comprension_comunicacion**: Habilidades de comprensión y comunicación
- **dimension_3_pensamiento_cientifico**: Capacidad de pensamiento científico
- **dimension_4_analisis_social_humanistico**: Análisis social y humanístico
- **dimension_5_creatividad_innovacion**: Creatividad e innovación
- **dimension_6_liderazgo_trabajo_equipo**: Liderazgo y trabajo en equipo
- **dimension_7_pensamiento_critico**: Pensamiento crítico
- **dimension_8_adaptabilidad_aprendizaje**: Adaptabilidad y capacidad de aprendizaje

### ⚙️ Configuración
- **model_type**: Tipo de modelo a usar ("knn" o "neural_network")

## Respuesta de la API

La API devuelve una predicción con las áreas de conocimiento recomendadas:

```json
{
  "id": 5,
  "user_email": "test@ejemplo.com",
  "predictions": [
    {
      "area": "Ciencias Sociales",
      "percentage": 8.695652173913043,
      "confidence": 0.08695652173913043
    },
    {
      "area": "Artes",
      "percentage": 4.3478260869565215,
      "confidence": 0.043478260869565216
    }
  ],
  "model_type": "ModelType.KNN",
  "model_version": "ModelType.KNN_v20250822_203116",
  "processing_time": 0.0,
  "confidence_score": 0.08695652173913043,
  "created_at": "2025-08-22T22:17:50.215506Z"
}
```

## Comando de Prueba

```bash
curl -X POST http://localhost:8000/api/v1/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_email": "estudiante@ejemplo.com",
    "matematicas": 350.5,
    "lectura_critica": 280.3,
    "ciencias_naturales": 320.1,
    "sociales_ciudadanas": 295.7,
    "ingles": 275.2,
    "dimension_1_logico_matematico": 4.2,
    "dimension_2_comprension_comunicacion": 3.8,
    "dimension_3_pensamiento_cientifico": 4.1,
    "dimension_4_analisis_social_humanistico": 3.5,
    "dimension_5_creatividad_innovacion": 4.0,
    "dimension_6_liderazgo_trabajo_equipo": 3.9,
    "dimension_7_pensamiento_critico": 4.3,
    "dimension_8_adaptabilidad_aprendizaje": 3.7,
    "model_type": "knn"
  }'
```