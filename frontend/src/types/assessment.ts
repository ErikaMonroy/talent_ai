// Tipos para el sistema de evaluación de competencias TalentAI

export type AssessmentStep = 'welcome' | 'personal-data' | 'icfes' | 'dimension' | 'results';

export interface Dimension {
  id: number;
  name: string;
  shortName: string;
  description: string;
  icon: string;
  color: string;
  competencies: Competency[];
  competencyCount?: number;
}

export interface Competency {
  id: string; // C001, C002, etc.
  title: string;
  description: string;
  dimensionId: number;
}

export interface AssessmentResponse {
  competencyId: string;
  value: number; // 1-5 Likert scale
  score: number;
  timestamp: number;
}

export interface DimensionProgress {
  completed: number;
  total: number;
  average: number;
  percentage: number;
}

export interface PersonalData {
  name: string;
  age: number;
  gender: string;
  education: string;
  location: string;
  workExperience?: string;
  workArea?: string;
  icfesScores: {
    matematicas: number;
    lectura_critica: number;
    ciencias_naturales: number;
    sociales_ciudadanas: number;
    ingles: number;
  };
  email?: string; // Campo opcional para compatibilidad
}

export interface AssessmentSession {
  id: string;
  startTime: Date;
  lastUpdated: Date;
  isCompleted: boolean;
  currentStep: string;
  currentDimension: number;
  currentQuestionIndex: number;
  formData?: {
    personalData: PersonalData;
  };
}

export interface LikertScaleProps {
  value?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export interface QuestionCardProps {
  competency: Competency;
  questionNumber: number;
  totalQuestions: number;
  value?: AssessmentResponse;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export interface DimensionPillProps {
  dimension: Dimension;
  progress: DimensionProgress;
  isActive: boolean;
  isCompleted: boolean;
  onClick: () => void;
}

export interface AssessmentHeaderProps {
  currentDimension: number;
  totalDimensions: number;
  currentCompetency: number;
  totalCompetencies: number;
  overallProgress: number;
  dimensionProgress: number;
  dimensionName: string;
}

export interface NavigationControlsProps {
  onPrevious?: () => void;
  onNext?: () => void;
  onHome?: () => void;
  onReset?: () => void;
  canGoPrevious?: boolean;
  canGoNext?: boolean;
  nextLabel?: string;
  previousLabel?: string;
  showHome?: boolean;
  showReset?: boolean;
  className?: string;
}

// Importar las dimensiones reales con competencias
import { DIMENSIONS } from '../data/dimensions';

// Usar las dimensiones reales en lugar de la configuración vacía
export const DIMENSIONS_CONFIG = DIMENSIONS;

// Labels para la escala Likert
export const LIKERT_LABELS = {
  1: "Muy bajo",
  2: "Bajo", 
  3: "Medio",
  4: "Alto",
  5: "Muy alto"
};

export const LIKERT_DESCRIPTIONS = {
  1: "No tengo experiencia o habilidad en esta área",
  2: "Tengo poca experiencia o habilidad",
  3: "Tengo experiencia y habilidad moderada",
  4: "Tengo buena experiencia y habilidad",
  5: "Tengo excelente experiencia y habilidad"
};