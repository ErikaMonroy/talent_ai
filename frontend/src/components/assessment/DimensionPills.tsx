'use client';

import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CheckCircle, Circle } from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { DIMENSIONS_CONFIG } from '@/types/assessment';
import { cn } from '@/lib/utils';

interface DimensionPillsProps {
  currentDimension: number;
  onDimensionChange?: (dimensionId: number) => void;
  showProgress?: boolean;
  className?: string;
}

export function DimensionPills({
  currentDimension,
  onDimensionChange,
  showProgress = true,
  className
}: DimensionPillsProps) {
  const { dimensionProgress, currentStep } = useAssessmentStore();
  const isInDimensionStep = currentStep === 'dimension';

  const getDimensionStatus = (dimensionId: number) => {
    const progress = dimensionProgress[dimensionId];
    if (!progress) return 'pending';
    
    const completion = (progress.completed / DIMENSIONS_CONFIG[dimensionId].competencyCount) * 100;
    if (completion >= 100) return 'completed';
    if (completion > 0) return 'in-progress';
    return 'pending';
  };

  const getDimensionProgress = (dimensionId: number) => {
    const progress = dimensionProgress[dimensionId];
    if (!progress) return 0;
    return (progress.completed / DIMENSIONS_CONFIG[dimensionId].competencyCount) * 100;
  };

  const handleDimensionClick = (dimensionId: number) => {
    if (onDimensionChange && isInDimensionStep) {
      onDimensionChange(dimensionId);
    }
  };

  return (
    <div className={cn("flex flex-wrap gap-3", className)}>
      {Object.entries(DIMENSIONS_CONFIG).map(([id, config]) => {
        const dimensionId = parseInt(id);
        const status = getDimensionStatus(dimensionId);
        const progress = getDimensionProgress(dimensionId);
        const isActive = currentDimension === dimensionId;
        const isClickable = isInDimensionStep && onDimensionChange;

        return (
          <div key={dimensionId} className="relative pt-2 pr-2 pb-1">
            <Button
              variant={isActive ? "default" : status === 'completed' ? "secondary" : "outline"}
              size="sm"
              onClick={() => handleDimensionClick(dimensionId)}
              disabled={!isClickable}
              className={cn(
                "flex items-center gap-2 transition-all duration-200",
                isActive && "ring-2 ring-primary ring-offset-2",
                status === 'completed' && "bg-green-100 border-green-300 text-green-800 dark:bg-green-900 dark:border-green-700 dark:text-green-200",
                status === 'in-progress' && "bg-yellow-100 border-yellow-300 text-yellow-800 dark:bg-yellow-900 dark:border-yellow-700 dark:text-yellow-200",
                isClickable && "cursor-pointer hover:scale-105",
                !isClickable && "cursor-default"
              )}
            >
              {/* Icono de estado */}
              <span className="text-lg">{config.icon}</span>
              
              {/* Nombre de la dimensión */}
              <span className="font-medium text-xs sm:text-sm">
                {config.shortName || config.name}
              </span>
              
              {/* Indicador de completado */}
              {status === 'completed' && (
                <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
              )}
              {status === 'in-progress' && (
                <Circle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
              )}
            </Button>
            
            {/* Barra de progreso mini */}
            {showProgress && status !== 'pending' && (
              <div className="absolute bottom-1 left-1 right-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className={cn(
                    "h-full transition-all duration-300 rounded-full",
                    status === 'completed' ? "bg-green-500" : "bg-yellow-500"
                  )}
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
            
            {/* Badge con número de dimensión */}
            <Badge 
              variant="secondary" 
              className="absolute -top-2 -right-2 h-5 w-5 p-0 flex items-center justify-center text-xs font-semibold"
            >
              {dimensionId}
            </Badge>
          </div>
        );
      })}
    </div>
  );
}

// Versión compacta para móviles
export function DimensionPillsCompact({
  currentDimension,
  onDimensionChange,
  className
}: DimensionPillsProps) {
  const { dimensionProgress, currentStep } = useAssessmentStore();
  const isInDimensionStep = currentStep === 'dimension';

  const getDimensionStatus = (dimensionId: number) => {
    const progress = dimensionProgress[dimensionId];
    if (!progress) return 'pending';
    
    const completion = (progress.completed / DIMENSIONS_CONFIG[dimensionId].competencyCount) * 100;
    if (completion >= 100) return 'completed';
    if (completion > 0) return 'in-progress';
    return 'pending';
  };

  const handleDimensionClick = (dimensionId: number) => {
    if (onDimensionChange && isInDimensionStep) {
      onDimensionChange(dimensionId);
    }
  };

  return (
    <div className={cn("flex gap-1 overflow-x-auto pb-2", className)}>
      {Object.entries(DIMENSIONS_CONFIG).map(([id, config]) => {
        const dimensionId = parseInt(id);
        const status = getDimensionStatus(dimensionId);
        const isActive = currentDimension === dimensionId;
        const isClickable = isInDimensionStep && onDimensionChange;

        return (
          <Button
            key={dimensionId}
            variant={isActive ? "default" : "outline"}
            size="sm"
            onClick={() => handleDimensionClick(dimensionId)}
            disabled={!isClickable}
            className={cn(
              "flex-shrink-0 w-12 h-12 p-0 flex flex-col items-center justify-center gap-1",
              isActive && "ring-2 ring-primary ring-offset-1",
              status === 'completed' && "bg-green-100 border-green-300 dark:bg-green-900 dark:border-green-700",
              status === 'in-progress' && "bg-yellow-100 border-yellow-300 dark:bg-yellow-900 dark:border-yellow-700"
            )}
          >
            <span className="text-sm">{config.icon}</span>
            <span className="text-xs font-medium">{dimensionId}</span>
            {status === 'completed' && (
              <CheckCircle className="absolute -top-1 -right-1 h-3 w-3 text-green-600 dark:text-green-400" />
            )}
          </Button>
        );
      })}
    </div>
  );
}