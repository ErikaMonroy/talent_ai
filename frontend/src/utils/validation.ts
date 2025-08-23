import type { PersonalData, AssessmentResponse } from '../types/assessment';
import { DIMENSIONS } from '../data/dimensions';
import { DIMENSIONS_CONFIG } from '../types/assessment';

// Tipos para validación
export interface ValidationError {
  field: string;
  message: string;
  type: 'required' | 'format' | 'range' | 'custom';
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: string[];
}

export interface FormValidationResult extends ValidationResult {
  completionPercentage: number;
  missingFields: string[];
  canProceed: boolean;
}

// Validaciones para datos personales
export const validatePersonalData = (data: PersonalData): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: string[] = [];

  // Validar nombre
  if (!data.name || data.name.trim().length < 2) {
    errors.push({
      field: 'name',
      message: 'El nombre debe tener al menos 2 caracteres',
      type: 'required'
    });
  }

  // Validar edad
  if (!data.age || data.age < 16 || data.age > 100) {
    errors.push({
      field: 'age',
      message: 'La edad debe estar entre 16 y 100 años',
      type: 'range'
    });
  }

  // Validar género
  if (!data.gender) {
    errors.push({
      field: 'gender',
      message: 'Debe seleccionar un género',
      type: 'required'
    });
  }

  // Validar educación
  if (!data.education) {
    errors.push({
      field: 'education',
      message: 'Debe seleccionar un nivel educativo',
      type: 'required'
    });
  }

  // Validar ubicación
  if (!data.location || data.location.trim().length < 2) {
    errors.push({
      field: 'location',
      message: 'Debe especificar su ubicación',
      type: 'required'
    });
  }

  // Validaciones opcionales con advertencias
  if (!data.workExperience) {
    warnings.push('La experiencia laboral ayuda a personalizar mejor los resultados');
  }

  if (!data.workArea) {
    warnings.push('El área de trabajo permite comparaciones más precisas');
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

// Validar respuesta individual
export const validateResponse = (value: number): ValidationResult => {
  const errors: ValidationError[] = [];

  if (value < 1 || value > 5) {
    errors.push({
      field: 'response',
      message: 'La respuesta debe estar entre 1 y 5',
      type: 'range'
    });
  }

  if (!Number.isInteger(value)) {
    errors.push({
      field: 'response',
      message: 'La respuesta debe ser un número entero',
      type: 'format'
    });
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings: []
  };
};

// Validar dimensión completa
export const validateDimension = (
  dimensionId: number,
  responses: Record<string, AssessmentResponse>
): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: string[] = [];
  
  const dimensionData = DIMENSIONS.find(d => d.id === dimensionId);
  const competencies = dimensionData?.competencies || [];
  const dimensionResponses = competencies.filter(comp => responses[comp.id]);
  
  // Verificar que todas las competencias tengan respuesta
  const missingCompetencies = competencies.filter(comp => !responses[comp.id]);
  
  if (missingCompetencies.length > 0) {
    errors.push({
      field: 'dimension',
      message: `Faltan ${missingCompetencies.length} respuestas en esta dimensión`,
      type: 'required'
    });
  }

  // Verificar consistencia de respuestas
  if (dimensionResponses.length > 0) {
    const values = dimensionResponses.map(comp => responses[comp.id].value);
    const average = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - average, 2), 0) / values.length;
    
    // Si todas las respuestas son iguales, podría indicar falta de atención
    if (variance === 0 && values.length > 5) {
      warnings.push('Todas las respuestas son iguales. Asegúrese de leer cada pregunta cuidadosamente.');
    }
    
    // Si hay mucha variabilidad, podría indicar inconsistencia
    if (variance > 2) {
      warnings.push('Las respuestas muestran mucha variabilidad. Revise si reflejan su situación real.');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

// Validar formulario completo
export const validateCompleteForm = (
  personalData: PersonalData,
  responses: Record<string, AssessmentResponse>
): FormValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: string[] = [];
  const missingFields: string[] = [];

  // Validar datos personales
  const personalDataValidation = validatePersonalData(personalData);
  errors.push(...personalDataValidation.errors);
  warnings.push(...personalDataValidation.warnings);

  // Validar todas las dimensiones
  DIMENSIONS_CONFIG.forEach(dimension => {
    const dimensionValidation = validateDimension(dimension.id, responses);
    errors.push(...dimensionValidation.errors);
    warnings.push(...dimensionValidation.warnings);
    
    if (!dimensionValidation.isValid) {
      missingFields.push(`Dimensión ${dimension.id}: ${dimension.name}`);
    }
  });

  // Calcular porcentaje de completitud
  const totalQuestions = 100; // Total de competencias
  const completedQuestions = Object.keys(responses).length;
  const completionPercentage = Math.round((completedQuestions / totalQuestions) * 100);

  // Determinar si se puede proceder
  const canProceed = personalDataValidation.isValid && completionPercentage >= 80;

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    completionPercentage,
    missingFields,
    canProceed
  };
};

// Validar progreso mínimo para continuar
export const validateMinimumProgress = (
  responses: Record<string, AssessmentResponse>,
  requiredPercentage: number = 20
): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: string[] = [];

  const totalQuestions = 100;
  const completedQuestions = Object.keys(responses).length;
  const completionPercentage = (completedQuestions / totalQuestions) * 100;

  if (completionPercentage < requiredPercentage) {
    errors.push({
      field: 'progress',
      message: `Debe completar al menos ${requiredPercentage}% del formulario para continuar`,
      type: 'custom'
    });
  }

  if (completionPercentage < 50) {
    warnings.push('Se recomienda completar al menos 50% del formulario para obtener resultados más precisos');
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

// Validar tiempo de respuesta (detectar respuestas muy rápidas)
export const validateResponseTime = (
  responses: Record<string, AssessmentResponse>,
  startTime: Date | string
): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: string[] = [];

  const responseCount = Object.keys(responses).length;
  
  // Handle both Date objects and string dates from localStorage
  const startTimeMs = startTime instanceof Date ? startTime.getTime() : new Date(startTime).getTime();
  const elapsedMinutes = (Date.now() - startTimeMs) / (1000 * 60);
  const averageTimePerResponse = elapsedMinutes / responseCount;

  // Si el promedio es menor a 5 segundos por respuesta, podría ser muy rápido
  if (averageTimePerResponse < 0.083) { // 5 segundos = 0.083 minutos
    warnings.push('Las respuestas parecen muy rápidas. Asegúrese de leer cada pregunta cuidadosamente.');
  }

  // Si el promedio es mayor a 5 minutos por respuesta, podría indicar distracción
  if (averageTimePerResponse > 5) {
    warnings.push('Se detectaron pausas largas. Los resultados podrían verse afectados por interrupciones.');
  }

  return {
    isValid: true, // No bloqueamos por tiempo, solo advertimos
    errors,
    warnings
  };
};

// Utilidad para obtener mensajes de error amigables
export const getErrorMessage = (error: ValidationError): string => {
  const messages: Record<string, string> = {
    'name': 'Nombre',
    'age': 'Edad',
    'gender': 'Género',
    'education': 'Educación',
    'location': 'Ubicación',
    'workExperience': 'Experiencia laboral',
    'workArea': 'Área de trabajo',
    'response': 'Respuesta',
    'dimension': 'Dimensión',
    'progress': 'Progreso'
  };

  const fieldName = messages[error.field] || error.field;
  return `${fieldName}: ${error.message}`;
};

// Utilidad para validar email (si se agrega en el futuro)
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Utilidad para validar teléfono (si se agrega en el futuro)
export const validatePhone = (phone: string): boolean => {
  const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
  return phoneRegex.test(phone.replace(/[\s\-\(\)]/g, ''));
};