import { useState, useCallback } from 'react';
import { apiService, prepareFormDataForPrediction } from '../services/api';
import {
  PredictionResponse,
  PredictionHistory,
  ProgramSearchResponse,
  ProgramSearchParams,
  ApiError
} from '../types/api';
import { useAssessmentStore } from '../store/assessmentStore';
import { toast } from 'sonner';

// Hook para manejar el estado de carga y errores de API
export const useApiState = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const clearError = useCallback(() => setError(null), []);

  return { loading, error, setLoading, setError, clearError };
};

// Hook para predicciones
export const usePrediction = () => {
  const { loading, error, setLoading, setError, clearError } = useApiState();
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const { personalData, responses, getDimensionAverage } = useAssessmentStore();

  const predict = useCallback(async (
    modelType: 'knn' | 'neural_network' = 'knn'
  ) => {
    if (!personalData.email) {
      setError({ message: 'Email es requerido para realizar la predicción' });
      return null;
    }

    setLoading(true);
    clearError();

    try {
      // Preparar datos de dimensiones
      const dimensionAverages: Record<number, number> = {};
      for (let i = 1; i <= 8; i++) {
        dimensionAverages[i] = getDimensionAverage(i);
      }

      // Preparar datos ICFES (por ahora valores por defecto)
      const icfesScores = {
        matematicas: personalData.icfesScores?.matematicas || 0,
        lectura_critica: personalData.icfesScores?.lectura_critica || 0,
        ciencias_naturales: personalData.icfesScores?.ciencias_naturales || 0,
        sociales_ciudadanas: personalData.icfesScores?.sociales_ciudadanas || 0,
        ingles: personalData.icfesScores?.ingles || 0
      };

      const predictionInput = prepareFormDataForPrediction(
        personalData.email,
        icfesScores,
        dimensionAverages,
        modelType
      );

      const response = await apiService.predict(predictionInput);

      if (response.success && response.data) {
        setPrediction(response.data);
        toast.success('Predicción realizada exitosamente');
        return response.data;
      } else {
        setError(response.error || { message: 'Error desconocido' });
        toast.error(response.error?.message || 'Error al realizar la predicción');
        return null;
      }
    } catch (err) {
      const error = err as ApiError;
      setError(error);
      toast.error(error.message || 'Error al realizar la predicción');
      return null;
    } finally {
      setLoading(false);
    }
  }, [personalData, responses, getDimensionAverage, setLoading, setError, clearError]);

  const getPredictionById = useCallback(async (id: number) => {
    setLoading(true);
    clearError();

    try {
      const response = await apiService.getPredictionById(id);

      if (response.success && response.data) {
        setPrediction(response.data);
        return response.data;
      } else {
        setError(response.error || { message: 'Error desconocido' });
        return null;
      }
    } catch (err) {
      const error = err as ApiError;
      setError(error);
      return null;
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError, clearError]);

  return {
    prediction,
    loading,
    error,
    predict,
    getPredictionById,
    clearError
  };
};

// Hook para historial de predicciones
export const usePredictionHistory = () => {
  const { loading, error, setLoading, setError, clearError } = useApiState();
  const [history, setHistory] = useState<PredictionHistory | null>(null);

  const getHistory = useCallback(async (
    userEmail: string,
    page: number = 1,
    size: number = 10
  ) => {
    setLoading(true);
    clearError();

    try {
      const response = await apiService.getPredictionHistory(userEmail, page, size);

      if (response.success && response.data) {
        setHistory(response.data);
        return response.data;
      } else {
        setError(response.error || { message: 'Error desconocido' });
        return null;
      }
    } catch (err) {
      const error = err as ApiError;
      setError(error);
      return null;
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError, clearError]);

  return {
    history,
    loading,
    error,
    getHistory,
    clearError
  };
};

// Hook para búsqueda de programas
export const useProgramSearch = () => {
  const { loading, error, setLoading, setError, clearError } = useApiState();
  const [programs, setPrograms] = useState<ProgramSearchResponse | null>(null);

  const searchPrograms = useCallback(async (params: ProgramSearchParams = {}) => {
    setLoading(true);
    clearError();

    try {
      const response = await apiService.searchPrograms(params);

      if (response.success && response.data) {
        setPrograms(response.data);
        return response.data;
      } else {
        setError(response.error || { message: 'Error desconocido' });
        return null;
      }
    } catch (err) {
      const error = err as ApiError;
      setError(error);
      return null;
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError, clearError]);

  return {
    programs,
    loading,
    error,
    searchPrograms,
    clearError
  };
};

// Hook para verificar conectividad con la API
export const useApiHealth = () => {
  const { loading, error, setLoading, setError, clearError } = useApiState();
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  const checkHealth = useCallback(async () => {
    setLoading(true);
    clearError();

    try {
      const response = await apiService.healthCheck();

      if (response.success) {
        setIsHealthy(true);
        return true;
      } else {
        setIsHealthy(false);
        setError(response.error || { message: 'API no disponible' });
        return false;
      }
    } catch (err) {
      const error = err as ApiError;
      setIsHealthy(false);
      setError(error);
      return false;
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError, clearError]);

  return {
    isHealthy,
    loading,
    error,
    checkHealth,
    clearError
  };
};

// Hook combinado para todas las operaciones de API
export const useApi = () => {
  const prediction = usePrediction();
  const history = usePredictionHistory();
  const programs = useProgramSearch();
  const health = useApiHealth();

  return {
    prediction,
    history,
    programs,
    health
  };
};