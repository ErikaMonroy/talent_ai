'use client';

import React from 'react';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Clock, User, CheckCircle } from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { DIMENSIONS_CONFIG } from '@/types/assessment';

interface AssessmentHeaderProps {
  currentStep: number;
  totalSteps: number;
  stepName: string;
  completionPercentage: number;
}

export function AssessmentHeader({
  currentStep,
  totalSteps,
  stepName,
  completionPercentage
}: AssessmentHeaderProps) {
  const { 
    personalData, 
    currentDimension, 
    currentQuestionIndex,
    getDimensionProgress,
    calculateDimensionAverage,
    session 
  } = useAssessmentStore();

  const currentDimensionConfig = DIMENSIONS_CONFIG.find(d => d.id === currentDimension);
  const currentDimensionProgress = getDimensionProgress(currentDimension);
  
  const isInDimensionStep = currentStep >= 2 && currentStep <= 9;
  const estimatedTimeRemaining = isInDimensionStep && currentDimensionConfig 
    ? Math.ceil((5 * (1 - (currentDimensionProgress?.completed || 0) / currentDimensionConfig.competencyCount)))
    : 0;

  const formatElapsedTime = () => {
    if (!session?.startTime) return '0 min';
    const elapsed = Math.floor((Date.now() - new Date(session.startTime).getTime()) / 60000);
    return `${elapsed} min`;
  };

  return (
    <div className="bg-card border-b shadow-sm">
      <div className="max-w-4xl mx-auto px-4 py-4">
        {/* Información del usuario y sesión */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <User className="h-4 w-4" />
              <span>{personalData.name || 'Usuario'}</span>
            </div>
            
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Tiempo: {formatElapsedTime()}</span>
            </div>
            
            {isInDimensionStep && currentDimensionConfig && (
              <Badge variant="default" className="text-xs">
                Dimensión {currentDimension}
              </Badge>
            )}
            
            {isInDimensionStep && estimatedTimeRemaining > 0 && (
              <Badge variant="secondary" className="text-xs">
                ~{estimatedTimeRemaining} min restantes
              </Badge>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              Sesión: {session?.id?.slice(-8) || 'N/A'}
            </Badge>
          </div>
        </div>





        {/* Progreso general */}
        <div className="">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Progreso general de la evaluación
            </span>
            <span className="text-sm text-muted-foreground">
              {completionPercentage}% completado
            </span>
          </div>
          
          <Progress value={completionPercentage} className="h-2" />
          
          <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
            <span>Paso {currentStep} de {totalSteps}</span>
            <div className="flex items-center gap-1">
              <CheckCircle className="h-3 w-3" />
              <span>{Math.floor(completionPercentage)} preguntas respondidas de 100</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}