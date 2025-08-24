import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  PredictionInput,
  PredictionResponse,
  PredictionHistory,
  ProgramSearchResponse,
  ProgramSearchParams,
  ProgramFilters,
  ApiResponse,
  ApiError
} from '../types/api';

// Configuración base de la API
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class TalentAIApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000, // 30 segundos para predicciones ML
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Interceptor para manejo de errores
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        const apiError: ApiError = {
          message: error.response?.data?.message || error.message || 'Error desconocido',
          code: error.response?.status?.toString() || 'UNKNOWN',
          details: error.response?.data
        };
        return Promise.reject(apiError);
      }
    );
  }

  /**
   * Realizar predicción de programas académicos
   */
  async predict(input: PredictionInput): Promise<ApiResponse<PredictionResponse>> {
    try {
      const response = await this.client.post<PredictionResponse>(
        '/api/v1/predictions/predict',
        input
      );
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Obtener historial de predicciones de un usuario
   */
  async getPredictionHistory(
    userEmail: string,
    page: number = 1,
    size: number = 10
  ): Promise<ApiResponse<PredictionHistory>> {
    try {
      const response = await this.client.get<PredictionHistory>(
        `/api/v1/predictions/history/${encodeURIComponent(userEmail)}`,
        {
          params: { page, size }
        }
      );
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Obtener una predicción específica por ID
   */
  async getPredictionById(id: number): Promise<ApiResponse<PredictionResponse>> {
    try {
      const response = await this.client.get<PredictionResponse>(
        `/api/v1/predictions/${id}`
      );
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Buscar programas académicos
   */
  async searchPrograms(params: ProgramSearchParams = {}): Promise<ApiResponse<ProgramSearchResponse>> {
    try {
      const response = await this.client.get<ProgramSearchResponse>(
        '/api/v1/programs/search',
        { params }
      );
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Obtener detalles de un programa específico
   */
  async getProgramById(id: number): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get(`/api/v1/programs/${id}`);
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Obtener áreas de conocimiento disponibles
   */
  async getAreas(): Promise<ApiResponse<any[]>> {
    try {
      const response = await this.client.get<any[]>(
        '/api/v1/areas'
      );
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Obtener áreas de conocimiento con paginación y filtros
   */
  async getKnowledgeAreas(params: {
    area_id?: number;
    limit?: number;
    offset?: number;
  } = {}): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get(
        '/api/v1/programs/areas',
        { params }
      );
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Obtener información detallada de un área específica por ID
   */
  async getAreaById(areaId: number): Promise<ApiResponse<any>> {
    try {
      const response = await this.getKnowledgeAreas({ area_id: areaId, limit: 1 });
      
      if (response.success && response.data?.areas?.length > 0) {
        return {
          data: response.data.areas[0],
          success: true
        };
      }
      
      return {
        error: { message: 'Área no encontrada', code: '404' } as ApiError,
        success: false
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Obtener filtros disponibles para búsqueda de programas
   */
  async getAvailableFilters(): Promise<ApiResponse<ProgramFilters>> {
    try {
      const response = await this.client.get<ProgramFilters>(
        '/api/v1/programs/filters'
      );
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }

  /**
   * Verificar el estado de salud de la API
   */
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    try {
      const response = await this.client.get('/api/v1/health');
      
      return {
        data: response.data,
        success: true
      };
    } catch (error) {
      return {
        error: error as ApiError,
        success: false
      };
    }
  }
}

// Instancia singleton del servicio de API
export const apiService = new TalentAIApiService();

// Funciones de utilidad para preparar datos
export const prepareFormDataForPrediction = (
  email: string,
  icfesScores: Record<string, number>,
  dimensionAverages: Record<number, number>,
  modelType: 'knn' | 'neural_network' = 'knn'
): PredictionInput => {
  // Función para asegurar que los valores de dimensiones estén en el rango 1-5
  const clampDimensionValue = (value: number): number => {
    if (value === 0) return 1; // Si no hay respuestas, usar valor mínimo
    return Math.max(1, Math.min(5, value)); // Asegurar rango 1-5
  };

  // Función para asegurar que los puntajes ICFES estén en rangos válidos
  const clampIcfesScore = (value: number, max: number): number => {
    return Math.max(0, Math.min(max, value));
  };

  return {
    user_email: email,
    matematicas: clampIcfesScore(icfesScores.matematicas || 0, 500),
    lectura_critica: clampIcfesScore(icfesScores.lectura_critica || 0, 400),
    ciencias_naturales: clampIcfesScore(icfesScores.ciencias_naturales || 0, 300),
    sociales_ciudadanas: clampIcfesScore(icfesScores.sociales_ciudadanas || 0, 400),
    ingles: clampIcfesScore(icfesScores.ingles || 0, 500),
    dimension_1_logico_matematico: clampDimensionValue(dimensionAverages[1] || 0),
    dimension_2_comprension_comunicacion: clampDimensionValue(dimensionAverages[2] || 0),
    dimension_3_pensamiento_cientifico: clampDimensionValue(dimensionAverages[3] || 0),
    dimension_4_analisis_social_humanistico: clampDimensionValue(dimensionAverages[4] || 0),
    dimension_5_creatividad_innovacion: clampDimensionValue(dimensionAverages[5] || 0),
    dimension_6_liderazgo_trabajo_equipo: clampDimensionValue(dimensionAverages[6] || 0),
    dimension_7_pensamiento_critico: clampDimensionValue(dimensionAverages[7] || 0),
    dimension_8_adaptabilidad_aprendizaje: clampDimensionValue(dimensionAverages[8] || 0),
    model_type: modelType
  };
};

export default apiService;