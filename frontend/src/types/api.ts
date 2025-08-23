// Tipos de API basados en la arquitectura técnica de TalentAI

// Tipos de entrada para predicción
export interface PredictionInput {
  user_email: string;
  matematicas: number;
  lectura_critica: number;
  ciencias_naturales: number;
  sociales_ciudadanas: number;
  ingles: number;
  dimension_1_logico_matematico: number;
  dimension_2_comprension_comunicacion: number;
  dimension_3_pensamiento_cientifico: number;
  dimension_4_analisis_social_humanistico: number;
  dimension_5_creatividad_innovacion: number;
  dimension_6_liderazgo_trabajo_equipo: number;
  dimension_7_pensamiento_critico: number;
  dimension_8_adaptabilidad_aprendizaje: number;
  model_type?: 'knn' | 'neural_network';
}

// Predicción por área
export interface AreaPrediction {
  area_id: number;
  area_name: string;
  percentage: number;
}

// Respuesta de predicción
export interface PredictionResponse {
  id: number;
  user_email: string;
  predictions: AreaPrediction[];
  model_type: string;
  model_version: string;
  processing_time: number;
  confidence_score: number | null;
  created_at: string;
}

// Historial de predicciones
export interface PredictionHistory {
  predictions: PredictionResponse[];
  total: number;
  page: number;
  size: number;
  total_pages: number;
}

// Programa académico
export interface Program {
  id: number;
  name: string;
  area_id: number;
  city: string;
  department: string;
  academic_level: string;
  institution: string;
}

// Área de conocimiento
export interface Area {
  id: number;
  name: string;
  description: string;
  icon: string;
}

// Respuesta de búsqueda de programas
export interface ProgramSearchResponse {
  programs: Program[];
  total: number;
  page: number;
  size: number;
  total_pages: number;
}

// Parámetros de búsqueda de programas
export interface ProgramSearchParams {
  area_id?: number;
  city?: string;
  department?: string;
  academic_level?: string;
  name?: string;
  page?: number;
  size?: number;
}

// Filtros disponibles para búsqueda de programas
export interface ProgramFilters {
  cities: string[];
  departments: string[];
  academic_levels: string[];
  knowledge_areas: {
    id: number;
    name: string;
    code: string;
  }[];
  total_programs: number;
  filter_counts: {
    cities: number;
    departments: number;
    academic_levels: number;
    knowledge_areas: number;
  };
}

// Datos del formulario local
export interface TalentAIFormData {
  sessionId: string;
  email: string;
  timestamp: number;
  isComplete: boolean;
  currentDimension: number;
  currentQuestion: number;
  responses: Record<string, number>;
  dimensionAverages: Record<number, number>;
  icfesScores: {
    matematicas: number;
    lectura_critica: number;
    ciencias_naturales: number;
    sociales_ciudadanas: number;
    ingles: number;
  };
}

// Respuesta de error de API
export interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

// Respuesta genérica de API
export interface ApiResponse<T> {
  data?: T;
  error?: ApiError;
  success: boolean;
}