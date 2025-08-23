import { useState, useEffect, useMemo } from 'react';
import { useAssessmentStore } from '../store/assessmentStore';
import {
  validatePersonalData,
  validateDimension,
  validateCompleteForm,
  validateMinimumProgress,
  validateResponseTime,
  type ValidationResult,
  type FormValidationResult,
  type ValidationError
} from '../utils/validation';
import type { PersonalData } from '../types/assessment';

// Hook para validación de datos personales
export const usePersonalDataValidation = (data: PersonalData) => {
  const [validation, setValidation] = useState<ValidationResult>({
    isValid: true,
    errors: [],
    warnings: []
  });

  useEffect(() => {
    const result = validatePersonalData(data);
    setValidation(result);
  }, [data]);

  return validation;
};

// Hook para validación de dimensión específica
export const useDimensionValidation = (dimensionId: number) => {
  const { responses } = useAssessmentStore();
  const [validation, setValidation] = useState<ValidationResult>({
    isValid: true,
    errors: [],
    warnings: []
  });

  useEffect(() => {
    const result = validateDimension(dimensionId, responses);
    setValidation(result);
  }, [dimensionId, responses]);

  return validation;
};

// Hook principal para validación del formulario completo
export const useFormValidation = () => {
  const { personalData, responses, session } = useAssessmentStore();
  const [validation, setValidation] = useState<FormValidationResult>({
    isValid: false,
    errors: [],
    warnings: [],
    completionPercentage: 0,
    missingFields: [],
    canProceed: false
  });

  const [timeValidation, setTimeValidation] = useState<ValidationResult>({
    isValid: true,
    errors: [],
    warnings: []
  });

  // Validación completa del formulario
  useEffect(() => {
    const result = validateCompleteForm(personalData, responses);
    setValidation(result);
  }, [personalData, responses]);

  // Validación de tiempo de respuesta
  useEffect(() => {
    if (Object.keys(responses).length > 0) {
      const result = validateResponseTime(responses, session.startTime);
      setTimeValidation(result);
    }
  }, [responses, session.startTime]);

  // Validación de progreso mínimo
  const progressValidation = useMemo(() => {
    return validateMinimumProgress(responses, 20);
  }, [responses]);

  // Combinar todas las validaciones
  const combinedValidation = useMemo(() => {
    const allErrors = [
      ...validation.errors,
      ...timeValidation.errors,
      ...progressValidation.errors
    ];

    const allWarnings = [
      ...validation.warnings,
      ...timeValidation.warnings,
      ...progressValidation.warnings
    ];

    return {
      ...validation,
      errors: allErrors,
      warnings: allWarnings,
      isValid: allErrors.length === 0
    };
  }, [validation, timeValidation, progressValidation]);

  return {
    validation: combinedValidation,
    progressValidation,
    timeValidation,
    // Funciones de utilidad
    hasErrors: combinedValidation.errors.length > 0,
    hasWarnings: combinedValidation.warnings.length > 0,
    canProceedToResults: validation.canProceed,
    completionPercentage: validation.completionPercentage
  };
};

// Hook para validación en tiempo real de respuestas
export const useResponseValidation = () => {
  const { responses, saveResponse } = useAssessmentStore();
  const [validationErrors, setValidationErrors] = useState<Record<string, ValidationError[]>>({});

  const validateAndSaveResponse = (competencyId: string, value: number) => {
    // Validar la respuesta antes de guardarla
    if (value >= 1 && value <= 5 && Number.isInteger(value)) {
      saveResponse(competencyId, value);
      // Limpiar errores si la validación es exitosa
      setValidationErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[competencyId];
        return newErrors;
      });
    } else {
      // Agregar error de validación
      setValidationErrors(prev => ({
        ...prev,
        [competencyId]: [{
          field: 'response',
          message: 'La respuesta debe estar entre 1 y 5',
          type: 'range'
        }]
      }));
    }
  };

  const getResponseError = (competencyId: string): ValidationError[] => {
    return validationErrors[competencyId] || [];
  };

  const hasResponseError = (competencyId: string): boolean => {
    return (validationErrors[competencyId]?.length || 0) > 0;
  };

  return {
    validateAndSaveResponse,
    getResponseError,
    hasResponseError,
    validationErrors
  };
};

// Hook para validación de navegación
export const useNavigationValidation = () => {
  const { personalData, responses, currentStep, currentDimension } = useAssessmentStore();

  const canNavigateToStep = (targetStep: string): { canNavigate: boolean; reason?: string } => {
    switch (targetStep) {
      case 'personal-data':
        return { canNavigate: true };
      
      case 'dimension':
        const personalValidation = validatePersonalData(personalData);
        if (!personalValidation.isValid) {
          return {
            canNavigate: false,
            reason: 'Debe completar correctamente los datos personales primero'
          };
        }
        return { canNavigate: true };
      
      case 'results':
        const formValidation = validateCompleteForm(personalData, responses);
        if (!formValidation.canProceed) {
          return {
            canNavigate: false,
            reason: `Debe completar al menos 80% del formulario. Progreso actual: ${formValidation.completionPercentage}%`
          };
        }
        return { canNavigate: true };
      
      default:
        return { canNavigate: true };
    }
  };

  const canNavigateToDimension = (targetDimension: number): { canNavigate: boolean; reason?: string } => {
    // Permitir navegación libre entre dimensiones una vez que se inicie la evaluación
    const personalValidation = validatePersonalData(personalData);
    if (!personalValidation.isValid) {
      return {
        canNavigate: false,
        reason: 'Debe completar los datos personales primero'
      };
    }

    return { canNavigate: true };
  };

  const canGoToNextDimension = (): { canNavigate: boolean; reason?: string } => {
    const dimensionValidation = validateDimension(currentDimension, responses);
    
    // Permitir avanzar si al menos 50% de la dimensión está completa
    const completionPercentage = (dimensionValidation.errors.length === 0) ? 100 : 0;
    
    if (completionPercentage < 50) {
      return {
        canNavigate: false,
        reason: 'Complete al menos 50% de las preguntas de esta dimensión antes de continuar'
      };
    }

    return { canNavigate: true };
  };

  return {
    canNavigateToStep,
    canNavigateToDimension,
    canGoToNextDimension
  };
};

// Hook para mostrar alertas de validación
export const useValidationAlerts = () => {
  const { validation } = useFormValidation();
  const [dismissedWarnings, setDismissedWarnings] = useState<Set<string>>(new Set());

  const activeWarnings = validation.warnings.filter(
    warning => !dismissedWarnings.has(warning)
  );

  const dismissWarning = (warning: string) => {
    setDismissedWarnings(prev => new Set([...prev, warning]));
  };

  const dismissAllWarnings = () => {
    setDismissedWarnings(new Set(validation.warnings));
  };

  const resetDismissedWarnings = () => {
    setDismissedWarnings(new Set());
  };

  return {
    errors: validation.errors,
    warnings: activeWarnings,
    allWarnings: validation.warnings,
    hasActiveWarnings: activeWarnings.length > 0,
    hasErrors: validation.errors.length > 0,
    dismissWarning,
    dismissAllWarnings,
    resetDismissedWarnings
  };
};

// Hook para estadísticas de validación
export const useValidationStats = () => {
  const { validation } = useFormValidation();
  const { responses } = useAssessmentStore();

  const stats = useMemo(() => {
    const totalQuestions = 100;
    const completedQuestions = Object.keys(responses).length;
    const errorCount = validation.errors.length;
    const warningCount = validation.warnings.length;
    const missingQuestions = totalQuestions - completedQuestions;

    return {
      totalQuestions,
      completedQuestions,
      missingQuestions,
      completionPercentage: validation.completionPercentage,
      errorCount,
      warningCount,
      isComplete: completedQuestions === totalQuestions,
      canProceed: validation.canProceed,
      qualityScore: Math.max(0, 100 - (errorCount * 10) - (warningCount * 5))
    };
  }, [validation, responses]);

  return stats;
};