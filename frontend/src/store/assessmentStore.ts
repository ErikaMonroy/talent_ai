import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { AssessmentResponse, PersonalData, AssessmentSession, AssessmentStep, Dimension } from '../types/assessment';
import { PredictionResponse, AreaPrediction } from '../types/api';
import { DIMENSIONS, COMPETENCIES } from '../data/dimensions';
import { validatePersonalData, validateResponse, validateCompleteForm } from "../utils/validation";

interface AssessmentStore {
  // Estado
  session: AssessmentSession;
  personalData: PersonalData;
  responses: Record<string, AssessmentResponse>;
  currentStep: AssessmentStep;
  currentDimension: number;
  currentQuestionIndex: number;
  dimensionProgress: Record<number, { completed: number; total: number }>;
  
  // Estado de validación
  validationErrors: string[];
  validationWarnings: string[];
  isValidating: boolean;
  lastValidation: Date | null;
  
  // Estado de predicción
  selectedModelType: 'knn' | 'neural_network';
  isLoadingPrediction: boolean;
  predictionResult: PredictionResponse | null;
  selectedAreaIds: number[];
  predictionError: string | null;
  
  // Acciones
  initializeSession: () => void;
  updatePersonalData: (data: Partial<PersonalData>) => void;
  saveResponse: (competencyId: string, value: number) => void;
  nextQuestion: () => void;
  previousQuestion: () => void;
  nextDimension: () => void;
  previousDimension: () => void;
  goToStep: (step: AssessmentStep) => boolean;
  goToDimension: (dimensionId: number) => void;
  goToPreviousStep: () => void;
  
  // Validación
  validateCurrentStep: () => Promise<boolean>;
  validatePersonalDataStep: () => boolean;
  validateDimensionStep: () => boolean;
  validateCompleteForm: () => boolean;
  clearValidationErrors: () => void;
  addValidationError: (error: string) => void;
  removeValidationError: (error: string) => void;
  
  // Utilidades
  getCurrentDimension: () => Dimension | undefined;
  getCurrentQuestion: () => any;
  getCompletionPercentage: () => number;
  getDimensionAverage: (dimensionId: number) => number;
  calculateDimensionAverage: (dimensionId: number) => number;
  getDimensionProgress: (dimensionId: number) => { completed: number; total: number; percentage: number };
  getDimensionCompletionPercentage: (dimensionId: number) => number;
  calculateDimensionProgress: () => void;
  canProceedToResults: () => boolean;
  getQualityScore: () => number;
  resetAssessment: () => void;
  
  // Acciones de predicción
  setSelectedModelType: (modelType: 'knn' | 'neural_network') => void;
  setLoadingPrediction: (loading: boolean) => void;
  setPredictionResult: (result: PredictionResponse | null) => void;
  setPredictionError: (error: string | null) => void;
  toggleAreaSelection: (areaId: number) => void;
  setSelectedAreas: (areaIds: number[]) => void;
  getTopPredictedAreas: (limit?: number) => AreaPrediction[];
  resetPredictionState: () => void;
}

const initialState = {
  session: {
    id: '',
    startTime: new Date(),
    lastUpdated: new Date(),
    currentStep: 'welcome' as AssessmentStep,
    currentDimension: 1,
    currentQuestionIndex: 0,
    isCompleted: false,
    timeSpent: 0
  },
  personalData: {
    name: '',
    age: 0,
    gender: '',
    education: '',
    workExperience: '',
    workArea: '',
    location: '',
    icfesScores: {
      matematicas: 0,
      lectura_critica: 0,
      ciencias_naturales: 0,
      sociales_ciudadanas: 0,
      ingles: 0
    }
  },
  responses: {},
  currentStep: 'welcome' as AssessmentStep,
  currentDimension: 1,
  currentQuestionIndex: 0,
  dimensionProgress: {},
  
  // Estado de validación
  validationErrors: [],
  validationWarnings: [],
  isValidating: false,
  lastValidation: null,
  
  // Estado de predicción
  selectedModelType: 'knn' as 'knn' | 'neural_network',
  isLoadingPrediction: false,
  predictionResult: null,
  selectedAreaIds: [],
  predictionError: null
};

export const useAssessmentStore = create<AssessmentStore>()(persist(
  (set, get) => ({
    ...initialState,
    
    // Inicializar sesión
    initializeSession: () => {
      const sessionId = `assessment_${Date.now()}`;
      set((state) => ({
        session: {
          ...state.session,
          id: sessionId,
          startTime: new Date()
        }
      }));
    },
    
    // Actualizar datos personales
    updatePersonalData: (data) => {
      set((state) => {
        const newPersonalData = { ...state.personalData, ...data };
        
        // Validar automáticamente los datos personales
        const validation = validatePersonalData(newPersonalData);
        
        return {
          personalData: newPersonalData,
          validationErrors: validation.errors.map(e => e.message),
          validationWarnings: validation.warnings,
          lastValidation: new Date()
        };
      });
    },
    
    // Guardar respuesta
    saveResponse: (competencyId, value) => {
      set((state) => {
        // Validar la respuesta individual
        const responseValidation = validateResponse(value);
        
        const newResponses = {
          ...state.responses,
          [competencyId]: {
            competencyId,
            value,
            score: value, // Por ahora score = value, se puede calcular diferente después
            timestamp: Date.now()
          }
        };
        
        return {
          responses: newResponses,
          validationErrors: responseValidation.errors.map(e => e.message),
          validationWarnings: responseValidation.warnings,
          lastValidation: new Date()
        };
      });
      
      // Recalcular progreso después de guardar respuesta
      get().calculateDimensionProgress();
    },
    
    // Navegación entre preguntas
    nextQuestion: () => {
      const state = get();
      const currentDimension = state.getCurrentDimension();
      
      if (!currentDimension) return;
      
      const totalQuestions = currentDimension.competencies.length;
      
      if (state.currentQuestionIndex < totalQuestions - 1) {
        set({ currentQuestionIndex: state.currentQuestionIndex + 1 });
      } else {
        // Ir a la siguiente dimensión o a resultados
        state.nextDimension();
      }
    },
    
    previousQuestion: () => {
      const state = get();
      
      if (state.currentQuestionIndex > 0) {
        set({ currentQuestionIndex: state.currentQuestionIndex - 1 });
      } else {
        // Ir a la dimensión anterior
        state.previousDimension();
      }
    },
    
    // Navegación entre dimensiones
    nextDimension: () => {
      const state = get();
      const currentIndex = DIMENSIONS.findIndex(d => d.id === state.currentDimension);
      
      if (currentIndex < DIMENSIONS.length - 1) {
        const nextDimension = DIMENSIONS[currentIndex + 1];
        set({
          currentDimension: nextDimension.id,
          currentQuestionIndex: 0
        });
      } else {
        // Ir a resultados
        set({ currentStep: 'results' });
      }
    },
    
    previousDimension: () => {
      const state = get();
      const currentIndex = DIMENSIONS.findIndex(d => d.id === state.currentDimension);
      
      if (currentIndex > 0) {
        const prevDimension = DIMENSIONS[currentIndex - 1];
        const prevDimensionQuestions = prevDimension.competencies.length;
        set({
          currentDimension: prevDimension.id,
          currentQuestionIndex: prevDimensionQuestions - 1
        });
      }
    },
    
    // Navegación entre pasos
    goToStep: (step) => {
      const state = get();
      
      // Validar antes de cambiar de paso
      if (step !== 'welcome' && !state.validateCurrentStep()) {
        return false;
      }
      
      set({ currentStep: step });
      return true;
    },
    
    goToDimension: (dimensionId) => {
      const dimension = DIMENSIONS.find(d => d.id === dimensionId);
      if (dimension) {
        set({
          currentDimension: dimensionId,
          currentQuestionIndex: 0,
          currentStep: 'dimension'
        });
        
        // Limpiar errores de validación al cambiar de dimensión
        get().clearValidationErrors();
      }
    },
    
    goToPreviousStep: () => {
      const state = get();
      
      switch (state.currentStep) {
        case 'personal-data':
          set({ currentStep: 'welcome' });
          break;
        case 'dimension':
          set({ currentStep: 'personal-data' });
          break;
        case 'results':
          set({ currentStep: 'dimension' });
          break;
      }
    },
    
    // Métodos de validación
    validateCurrentStep: async () => {
      const state = get();
      set({ isValidating: true });
      
      try {
        let isValid = false;
        
        switch (state.currentStep) {
          case 'personal-data':
            isValid = state.validatePersonalDataStep();
            break;
          case 'dimension':
            isValid = state.validateDimensionStep();
            break;
          case 'results':
            isValid = state.validateCompleteForm();
            break;
          default:
            isValid = true;
        }
        
        return isValid;
      } finally {
        set({ isValidating: false });
      }
    },
    
    validatePersonalDataStep: () => {
      const state = get();
      const validation = validatePersonalData(state.personalData);
      
      set({
        validationErrors: validation.errors.map(e => e.message),
        validationWarnings: validation.warnings,
        lastValidation: new Date()
      });
      
      return validation.isValid;
    },
    
    validateDimensionStep: () => {
      const state = get();
      const currentDimension = state.getCurrentDimension();
      
      if (!currentDimension) {
        set({
          validationErrors: ['Dimensión no encontrada'],
          validationWarnings: [],
          lastValidation: new Date()
        });
        return false;
      }
      
      const dimensionResponses = Object.values(state.responses)
        .filter(response => {
          const competency = COMPETENCIES.find(c => c.id === response.competencyId);
          return competency?.dimensionId === state.currentDimension;
        });
      
      const isValid = dimensionResponses.length >= currentDimension.competencies.length * 0.5;
      
      set({
        validationErrors: isValid ? [] : ['Completa al menos el 50% de las preguntas de esta dimensión'],
        validationWarnings: [],
        lastValidation: new Date()
      });
      
      return isValid;
    },
    
    validateCompleteForm: () => {
      const state = get();
      const validation = validateCompleteForm(
        state.personalData,
        state.responses
      );
      
      set({
        validationErrors: validation.errors.map(e => e.message),
        validationWarnings: validation.warnings,
        lastValidation: new Date()
      });
      
      return validation.isValid;
    },
    
    clearValidationErrors: () => {
      set({ validationErrors: [], validationWarnings: [] });
    },
    
    addValidationError: (error) => {
      set((state) => ({
        validationErrors: [...state.validationErrors, error]
      }));
    },
    
    removeValidationError: (error) => {
      set((state) => ({
        validationErrors: state.validationErrors.filter(e => e !== error)
      }));
    },
    
    // Utilidades
    getCurrentDimension: () => {
      const state = get();
      return DIMENSIONS.find(d => d.id === state.currentDimension);
    },
    
    getCurrentQuestion: () => {
      const state = get();
      const dimension = state.getCurrentDimension();
      return dimension?.competencies[state.currentQuestionIndex];
    },
    
    getCompletionPercentage: () => {
      const state = get();
      const totalQuestions = COMPETENCIES.length;
      const answeredQuestions = Object.keys(state.responses).length;
      return totalQuestions > 0 ? Math.round((answeredQuestions / totalQuestions) * 100) : 0;
    },
    
    getDimensionAverage: (dimensionId) => {
      const state = get();
      const dimension = DIMENSIONS.find(d => d.id === dimensionId);
      if (!dimension) return 0;
      
      const dimensionResponses = Object.values(state.responses)
        .filter(response => {
          const competency = COMPETENCIES.find(c => c.id === response.competencyId);
          return competency?.dimensionId === dimensionId;
        });
      
      if (dimensionResponses.length === 0) return 0;
      
      const sum = dimensionResponses.reduce((acc, response) => acc + response.value, 0);
      return sum / dimensionResponses.length;
    },

    calculateDimensionAverage: (dimensionId) => {
      return get().getDimensionAverage(dimensionId);
    },

    getDimensionProgress: (dimensionId) => {
      const state = get();
      const dimension = DIMENSIONS.find(d => d.id === dimensionId);
      if (!dimension) return { completed: 0, total: 0, percentage: 0 };
      
      const dimensionResponses = Object.values(state.responses)
        .filter(response => {
          const competency = COMPETENCIES.find(c => c.id === response.competencyId);
          return competency?.dimensionId === dimensionId;
        });
      
      const completed = dimensionResponses.length;
      const total = dimension.competencies.length;
      const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
      
      return { completed, total, percentage };
    },
    
    getDimensionCompletionPercentage: (dimensionId) => {
      const state = get();
      const dimension = DIMENSIONS.find(d => d.id === dimensionId);
      if (!dimension) return 0;
      
      const dimensionResponses = Object.values(state.responses)
        .filter(response => {
          const competency = COMPETENCIES.find(c => c.id === response.competencyId);
          return competency?.dimensionId === dimensionId;
        });
      
      return Math.round((dimensionResponses.length / dimension.competencies.length) * 100);
    },
    
    calculateDimensionProgress: () => {
      const state = get();
      const progress: Record<number, { completed: number; total: number }> = {};
      
      DIMENSIONS.forEach(dimension => {
        const dimensionResponses = Object.values(state.responses)
          .filter(response => {
            const competency = COMPETENCIES.find(c => c.id === response.competencyId);
            return competency?.dimensionId === dimension.id;
          });
        
        progress[dimension.id] = {
          completed: dimensionResponses.length,
          total: dimension.competencies.length
        };
      });
      
      set({ dimensionProgress: progress });
    },
    
    // Utilidades mejoradas
    canProceedToResults: () => {
      const state = get();
      const completionPercentage = state.getCompletionPercentage();
      const hasPersonalData = validatePersonalData(state.personalData).isValid;
      
      // Verificar que todas las 8 dimensiones estén completadas al 100%
      const allDimensionsCompleted = DIMENSIONS.every(dimension => {
        const dimensionCompletion = state.getDimensionCompletionPercentage(dimension.id);
        return dimensionCompletion === 100;
      });
      
      return completionPercentage === 100 && allDimensionsCompleted && hasPersonalData;
    },
    
    getQualityScore: () => {
      const state = get();
      const responses = Object.values(state.responses);
      
      if (responses.length === 0) return 0;
      
      // Calcular calidad basada en:
      // 1. Variabilidad de respuestas (evitar respuestas todas iguales)
      // 2. Tiempo de respuesta (evitar respuestas muy rápidas)
      // 3. Completitud por dimensión
      
      let qualityScore = 0;
      
      // Variabilidad (30% del score)
      const values = responses.map(r => r.value);
      const uniqueValues = new Set(values).size;
      const variabilityScore = Math.min((uniqueValues / 5) * 30, 30);
      
      // Completitud (50% del score)
      const completionScore = (state.getCompletionPercentage() / 100) * 50;
      
      // Consistencia temporal (20% del score)
      const avgResponseTime = responses.length > 1 ? 
        responses.reduce((acc, curr, index) => {
          if (index === 0) return acc;
          const prevTime = responses[index - 1].timestamp;
          const currTime = curr.timestamp;
          return acc + (currTime - prevTime);
        }, 0) / (responses.length - 1) : 5000;
      
      const timeScore = avgResponseTime > 2000 && avgResponseTime < 30000 ? 20 : 10;
      
      qualityScore = variabilityScore + completionScore + timeScore;
      
      return Math.round(Math.min(qualityScore, 100));
    },
    
    resetAssessment: () => {
      set(initialState);
      localStorage.removeItem('assessment-storage');
    },
    
    // Acciones de predicción
    setSelectedModelType: (modelType) => {
      set({ selectedModelType: modelType });
    },
    
    setLoadingPrediction: (loading) => {
      set({ isLoadingPrediction: loading });
    },
    
    setPredictionResult: (result) => {
      set({ predictionResult: result, predictionError: null });
    },
    
    setPredictionError: (error) => {
      set({ predictionError: error, predictionResult: null });
    },
    
    toggleAreaSelection: (areaId) => {
      set((state) => {
        const isSelected = state.selectedAreaIds.includes(areaId);
        const newSelectedAreas = isSelected
          ? state.selectedAreaIds.filter(id => id !== areaId)
          : [...state.selectedAreaIds, areaId];
        return { selectedAreaIds: newSelectedAreas };
      });
    },
    
    setSelectedAreas: (areaIds) => {
      set({ selectedAreaIds: areaIds });
    },
    
    getTopPredictedAreas: (limit = 5) => {
      const { predictionResult } = get();
      if (!predictionResult?.predictions) return [];
      
      return predictionResult.predictions
        .sort((a, b) => b.percentage - a.percentage)
        .slice(0, limit);
    },
    
    resetPredictionState: () => {
      set({
        selectedModelType: 'knn',
        isLoadingPrediction: false,
        predictionResult: null,
        selectedAreaIds: [],
        predictionError: null
      });
    }
  }),
  {
    name: 'assessment-storage',
    version: 1
  }
));